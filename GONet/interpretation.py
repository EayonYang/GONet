


import os
# import args
import torch
import random
import numpy as np
import itertools
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics

from scipy.stats import spearmanr, rankdata
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns


from torch_geometric.explain import Explainer, CaptumExplainer, GNNExplainer, AttentionExplainer, PGExplainer
from captum.attr import IntegratedGradients, DeepLift, GradientShap, Saliency, InputXGradient
from torch_geometric.utils import unbatch, unbatch_edge_index
import json
from captum.attr import *
from captum.metrics import infidelity, sensitivity_max

################################## interpretation analysis ######################
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_baseline(sub_x, baseline_type, seed):
    set_seed(seed)
    if baseline_type == 'zeros':
        return torch.zeros_like(sub_x)
    elif baseline_type == 'uniform':
        return torch.rand_like(sub_x)  
    elif baseline_type == 'gaussian':
        return torch.randn_like(sub_x) 
    elif baseline_type == 'mean':
        return sub_x.mean(dim=0, keepdim=True).repeat(sub_x.size(0), 1)
    elif baseline_type == 'shuffle':
        return sub_x[torch.randperm(sub_x.size(0))]
    else:
        raise ValueError(f"Unsupported baseline_type: {baseline_type}")


def normalize(attr, method='zscore', eps=1e-8):
    if method == 'minmax':
        min_val = attr.min(dim=0, keepdim=True).values
        max_val = attr.max(dim=0, keepdim=True).values
        return (attr - min_val) / (max_val - min_val + eps)
    elif method == 'zscore':
        mean = attr.mean(dim=0, keepdim=True)
        std = attr.std(dim=0, keepdim=True) + eps
        return (attr - mean) / std
    elif method == 'absrank':
        abs_attr = torch.abs(attr)
        ranks = torch.argsort(torch.argsort(abs_attr, dim=0, descending=True), dim=0).float()
        return ranks / (attr.size(0) - 1) 
    else:
        raise ValueError("Invalid normalization method")

def interpret_with_method(method, sub_forward, sub_inputs, sub_baseline, sub_batch, n_steps, target):
    if method == 'IG':
        attr_method = IntegratedGradients(sub_forward)
        attrs, delta = attr_method.attribute(sub_inputs, sub_baseline, target=target,  internal_batch_size=len(sub_batch), n_steps=n_steps, return_convergence_delta=True)
        return attrs, delta
    elif method == 'Saliency':
        attrs = Saliency(sub_forward).attribute(sub_inputs, target=target)
        return attrs, torch.tensor(0.0)
    elif method == 'InputXGradient':
        attrs = InputXGradient(sub_forward).attribute(sub_inputs, target=target)
        return attrs, torch.tensor(0.0)
    else:
        raise ValueError(f'Unsupport method:{method}')


def compute_fidelity(sub_forward, sub_inputs, sub_baselines, sub_attributions, sub_target_idx, original_target_prob, is_positive, fraction=0.5):
        attr_x, attr_ae_x = sub_attributions
        signed_x = attr_x.view(attr_x.size(0), -1).sum(dim=1)
        signed_ae = attr_ae_x.view(-1)

        k_x = max(1, int(len(signed_x) * fraction))
        k_ae = max(1, int(len(signed_ae) * fraction))

        if is_positive:
            mask_idx_x = torch.topk(signed_x, k_x).indices
            mask_idx_ae = torch.topk(signed_ae, k_ae).indices
        else:
            mask_idx_x = torch.topk(-signed_x, k_x).indices
            mask_idx_ae = torch.topk(-signed_ae, k_ae).indices

        perturbed_x = sub_inputs[0].clone()
        perturbed_ae = sub_inputs[1].clone().view(-1)

        perturbed_x[mask_idx_x] = sub_baselines[0][mask_idx_x]
        perturbed_ae[mask_idx_ae] = sub_baselines[1].clone().view(-1)[mask_idx_ae]
        perturbed_ae = perturbed_ae.view_as(sub_inputs[1])

        with torch.no_grad():
            perturbed_probs = F.softmax(sub_forward(perturbed_x, perturbed_ae), dim=1)
        return (original_target_prob - perturbed_probs[0, sub_target_idx]).item()



##################################### Model interpretation #######################

def explain_attribution(model, x, edge_index, y, batch, ae_x, intra_edge_index,
                             cross_edge_index, omics_mapping, patient_names, n_steps=50, local_seed=42, 
                             norm_method='minmax', baseline_type='zeros',  explain_method='IG'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)  # Move model to GPU
    x = x.to(device)
    edge_index = edge_index.to(device)
    y =  y.to(device)
    batch = batch.to(device)
    intra_edge_index = intra_edge_index.to(device)
    cross_edge_index = cross_edge_index.to(device)
    
    batch_size = len(batch.unique())
    model.eval()
    
    # Unbatch (operations on GPU)
    x_list = unbatch(x, batch)
    ae_x_list = [torch.tensor(ae_x.loc[name].values).unsqueeze(0).to(device) for name in patient_names]
    edge_index_list = unbatch_edge_index(edge_index, batch)
    intra_edge_index_list = unbatch_edge_index(intra_edge_index, batch)
    cross_edge_index_list = unbatch_edge_index(cross_edge_index, batch)
    y_list = y.view(-1).cpu().tolist() 
    batch_list = [torch.zeros(len(sub_x), dtype=torch.long, device=device) for sub_x in x_list]
    omics_mapping_list = omics_mapping 
    
    results = {
        'per_patient':[],
        'summary':{'attribution_ae':None,'attribution_x':[], 'delta':[], 'fidelity_plus':[], 'fidelity_minus':[]}}
    for i in range(batch_size):
        set_seed(local_seed)
        sub_x = x_list[i].clone().detach().requires_grad_(True)
        sub_ae_x = ae_x_list[i].clone().detach().requires_grad_(True)
        sub_inputs = (sub_x, sub_ae_x)

        sub_x_baselines = get_baseline(sub_x, baseline_type, local_seed)
        sub_ax_baselines = get_baseline(sub_ae_x, baseline_type, local_seed)
        sub_baselines = (sub_x_baselines, sub_ax_baselines)

        sub_edge_index, sub_intra_edge_index, sub_cross_edge_index = edge_index_list[i], intra_edge_index_list[i], cross_edge_index_list[i]
        sub_batch, sub_y = batch_list[i], torch.tensor([y_list[i]], device=device)
        
        def sub_forward(sx, sae): 
            return model(sx, sub_edge_index, sub_intra_edge_index, sub_cross_edge_index, sub_y, sae, sub_batch, omics_mapping_list, status='explain')
        
        with torch.no_grad():
            original_logits = sub_forward(*sub_inputs)
            original_probs = F.softmax(original_logits, dim=1)
            sub_target_idx = original_probs.argmax(dim=1).item()
            original_target_prob = original_probs[0, sub_target_idx].item()

        sub_attributions, sub_delta = interpret_with_method(explain_method, sub_forward, sub_inputs, sub_baselines, sub_batch, n_steps, sub_target_idx)
        sub_fid_plus = compute_fidelity(sub_forward, sub_inputs, sub_baselines, sub_attributions, sub_target_idx, original_target_prob, True)
        sub_fid_minus = compute_fidelity(sub_forward, sub_inputs, sub_baselines, sub_attributions, sub_target_idx, original_target_prob, False)
        attr_x, attr_ae = sub_attributions if isinstance(sub_attributions, (tuple, list)) else (sub_attributions, None)

        patient_result = {
            'patient_name': patient_names[i],
            'attribution_x': attr_x.detach().cpu().numpy(),
            'attribution_ae': attr_ae.detach().cpu().numpy() if attr_ae is not None else None,
            'delta': float(sub_delta) if sub_delta is not None else None,
            'fidelity_plus': float(sub_fid_plus),
            'fidelity_minus': float(sub_fid_minus),
            'target_prob': original_target_prob,
            'target_idx': sub_target_idx
        }
        results['per_patient'].append(patient_result)

    fid_plus_list = [p['fidelity_plus'] for p in results['per_patient']]
    fid_minus_list = [p['fidelity_minus'] for p in results['per_patient']]
    delta_list = [p['delta'] for p in results['per_patient']]
    results['summary'] = {
        'mean_delta': float(np.mean(delta_list)),
        'mean_fid_plus': float(np.mean(fid_plus_list)),
        'std_fid_plus': float(np.std(fid_plus_list)),
        'mean_fid_minus': float(np.mean(fid_minus_list)),
        'std_fid_minus': float(np.std(fid_minus_list)),
        'n_patients': batch_size
    }
    return results

import pickle
def interpretation_parameter(model, x, edge_index, y, batch, ae_x, intra_edge_index, cross_edge_index, omics_mapping, 
                            methods=('IG', 'Saliency', 'InputXGradient'), baselines_types=('zeros', 'uniform', 'gaussian'), seed=42, n_steps_list=(50, 200, 300, 400, 500, 600), patient_names=None, feature_names=None, cancer='BRCA', data='data1'):
    all_results = {}
    for method in methods:
        for baseline_type in baselines_types:
            for n_steps in n_steps_list:
                key = f'{method}_{baseline_type}_{n_steps}'
                print( f'*************running {key} *************' )
                res = explain_attribution(model, x, edge_index, y, batch, ae_x, intra_edge_index,
                             cross_edge_index, omics_mapping, patient_names, n_steps=n_steps, local_seed=42,
                             norm_method='minmax', baseline_type=baseline_type,  explain_method=method)
                all_results[f'{key}'] = res

                with open(f"./results/interpretation_results/{cancer}_{data}_results_{key}.pkl", "wb") as f:
                    pickle.dump(res, f)
    
    print("=== All interpretation parameter combinations completed ===")
    for key, res in all_results.items():
        summary = res['summary']
        print(f"{key}: Mean Delta +: {summary['mean_delta']:.4f}, Mean Fidelity +: {summary['mean_fid_plus']:.4f} ± {summary['std_fid_plus']:.4f}, Mean Fidelity -: {summary['mean_fid_minus']:.4f} ± {summary['std_fid_minus']:.4f}")

    # with open(f"parameter_sensitivity_results_{data}.pkl", "wb") as f:
    #     pickle.dump(all_results, f)

    return all_results
    


####################################################################################
#                                                                                  #
#                                                                                  # 
#                                   3. plot function                               #
#                                                                                  #
#                                                                                  #
####################################################################################
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

def plot_global_importance(ig_scores, topk, xlim=(0.1,0.4), palette='tab20',_save=False, path='./'):
    plt.rcParams.update({
        'font.family' : 'Arial',     
        'axes.labelsize': 12,        
        'axes.titlesize': 14,       
        'xtick.labelsize': 10,      
        'ytick.labelsize': 10,       
        'legend.fontsize': 10,       
        'figure.dpi': 300,            
        'savefig.dpi': 300,        
        'axes.linewidth': 0.5,       
        'lines.markersize': 4,        
        'legend.frameon': False,       
    })
    plt.figure(figsize=(12, 6))
    ig_top = ig_scores.iloc[:topk]
    
    fig, ax = plt.subplots(figsize=(14,12))
    palette = sns.color_palette(palette)
    sns.barplot(x='Scores', y='Features', hue='Type', hue_order=['mRNA', "prot"], data=ig_top, palette=palette, ax=ax, width=0.85, dodge=False)
    ax.set_xlim(left=xlim[0],right=xlim[1])
    ax.set_xlabel('Importance Score', fontsize=14)
    ax.set_ylabel('Features', fontsize=14)
    ax.tick_params(axis='y', labelsize=10)  
    ax.grid(True, axis='x', linestyle='--', alpha=0.7)  
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=10)
    plt.legend(title='Feature Type', fontsize=12, title_fontsize=14)
    
    plt.tight_layout()
    plt.title(f'Top {topk} Feature Importance by Integrated Gradients')

    if _save:
        plt.savefig(os.path.join(path, 'ig_feature_importance100.pdf'))
    plt.show()
    plt.close()


def plot_category_distribution(df, type_column='Type', title='Category Distribution', 
                               save_path=None, fig_size=(8, 8), explode_factor=0.05,
                               shadow=True, cmap='tab20', autopct='%1.1f%%'):
    """
    Plots an advanced pie chart for category distribution in the given DataFrame.
    Parameters:
    - df: pandas DataFrame containing the data.
    - type_column: str, column name containing the categories (default: 'Type').
    - title: str, title of the plot (default: 'Category Distribution').
    - save_path: str or None, path to save the figure (e.g., 'output.pdf'). If None, shows the plot.
    - fig_size: tuple, figure size (default: (8, 8)).
    - explode_factor: float, factor to explode each slice (default: 0.05).
    - shadow: bool, whether to add shadow to the pie (default: True).
    - cmap: str, colormap for pie slices (default: 'tab20').
    - autopct: str, format for percentage display (default: '%1.1f%%').

    Returns:
    None (displays or saves the plot).
    """
    plt.rcParams.update({
        'font.family' : 'Arial',      
        'axes.labelsize': 12,         
        'axes.titlesize': 14,      
        'xtick.labelsize': 10,    
        'ytick.labelsize': 10,       
        'legend.fontsize': 10,      
        'figure.dpi': 300,         
        'savefig.dpi': 300,        
        'axes.linewidth': 0.5,       
        'lines.markersize': 4,  
        'legend.frameon': False,    
    })
    category_counts = df[type_column].value_counts()
    explode = [explode_factor] * len(category_counts)
    colors = cm.get_cmap(cmap)(range(len(category_counts)))
    fig, ax = plt.subplots(figsize=fig_size)
    wedges, texts, autotexts = ax.pie(
        category_counts, 
        labels=category_counts.index, 
        autopct=autopct, 
        startangle=90, 
        explode=explode, 
        shadow=shadow, 
        colors=colors,
    )

    plt.setp(autotexts, size=10, weight="bold", color="white")
    plt.setp(texts, size=12)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.axis('equal') 
    centre_circle = plt.Circle((0, 0), 0.30, fc='white')
    fig.gca().add_artist(centre_circle)
    # Save or show
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(f'{save_path}/pie_chart_{title}.pdf', bbox_inches='tight', dpi=300)
        plt.show()
        plt.close(fig)
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()



def plot_scores_boxplot(df, x_col='Type', y_col='Scores', title='Boxplot of Scores by Omics Type', 
                        save_path=None, fig_size=(8, 6), palette='tab10', rotation=45,
                        grid=True, show_stats=True,stat='median',outlier_props=None, whisker_props=None):
    """
    Plots an advanced boxplot for scores distribution across different categories.
    
    Parameters:
    - df: pandas DataFrame containing the data.
    - x_col: str, column name for the x-axis categories (default: 'Type').
    - y_col: str, column name for the y-axis scores (default: 'Scores').
    - title: str, title of the plot (default: 'Boxplot of Scores by Omics Type').
    - save_path: str or None, path to save the figure (e.g., 'output.pdf'). If None, shows the plot.
    - fig_size: tuple, figure size (default: (8, 6)).
    - palette: str, seaborn color palette for boxes (default: 'tab10').
    - rotation: int, rotation angle for x-tick labels (default: 45).
    - grid: bool, whether to show grid lines (default: True).
    - outlier_props: dict or None, properties for outliers (e.g., {'marker': 'o', 'color': 'red'}).
    - whisker_props: dict or None, properties for whiskers (e.g., {'linestyle': '--', 'color': 'black'}).
    
    Returns:
    None (displays or saves the plot).
    """
    plt.rcParams.update({
        'font.family' : 'Arial',    
        'axes.labelsize': 12,       
        'axes.titlesize': 14,    
        'xtick.labelsize': 10,     
        'ytick.labelsize': 10,     
        'legend.fontsize': 10,      
        'figure.dpi': 300,       
        'savefig.dpi': 300,           
        'axes.linewidth': 0.5,       
        'lines.markersize': 4,       
        'legend.frameon': False,   
    })
    fig, ax = plt.subplots(figsize=fig_size)
    sns.set_style("whitegrid") if grid else sns.set_style("white")
    sns.boxplot(
        data=df, 
        x=x_col, 
        y=y_col, 
        palette=palette, 
        ax=ax,
        flierprops=outlier_props or {'marker': 'o', 'markersize': 5, 'markerfacecolor': 'lightblue', 'markeredgecolor': 'gray'},
        whiskerprops=whisker_props or {'linewidth': 1.5, 'color': 'black'},
        boxprops={'edgecolor': 'black', 'linewidth': 1.2},
        medianprops={'color': 'red', 'linewidth': 2}
    )
    
    plt.xticks(rotation=rotation, ha='right')
    ax.set_xlabel(x_col, fontsize=12, fontweight='bold')
    ax.set_ylabel(y_col, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold')
    
    if grid:
        ax.grid(True, linestyle='--', alpha=0.7)
    if show_stats:
        grouped = df.groupby(x_col)[y_col]
        categories = df[x_col].unique()
        pos = range(len(categories))
        
        for i, cat in enumerate(categories):
            data = grouped.get_group(cat)
            if stat == 'median' or stat == 'both':
                median = data.median()
                ax.text(pos[i], median, f'{median:.4f}', 
                        ha='center', va='bottom', fontweight='bold')
            if stat == 'mean' or stat == 'both':
                mean = data.mean()
                ax.text(pos[i], mean, f'{mean:.4f}', 
                        ha='center', va='top', fontweight='bold')
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(f'{save_path}/{title}.pdf', bbox_inches='tight', dpi=300)
        plt.show()
        plt.close(fig)  # Close to avoid displaying if saving
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()


def plot_feature_heatmap(test_data, top_features, test_labels, topk=50, title='Heatmap of Top Gene Importance in Global Module',
                          cmap='YlGnBu', annot=False, fmt='.2f', cbar_label='mean_importance', 
                          save_path=None, fig_size=(10, 10), rotation=90, grid=False, 
                          vmin=None, vmax=None, center=None, linewidths=0.5, linecolor='white',
                          sort_by_label=True
                        ):
    """
    Plots an advanced heatmap for top feature importance across samples, sorted by labels.
    
    Parameters:
    - test_data: pandas DataFrame containing the feature data.
    - top_features: pandas DataFrame with 'Features' column listing top features.
    - test_labels: pandas Series or array-like, labels for sorting samples.
    - topk: int, number of top features to include in the heatmap (default: 50).
    - title: str, title of the plot (default: 'Heatmap of Top Gene Importance in Global Module').
    - cmap: str, colormap for the heatmap (default: 'YlGnBu').
    - annot: bool, whether to annotate cells with values (default: False).
    - fmt: str, format for annotations (default: '.2f').
    - cbar_label: str, label for the colorbar (default: 'mean_importance').
    - save_path: str or None, path to save the figure (e.g., 'output.pdf'). If None, shows the plot.
    - fig_size: tuple, figure size (default: (10, 10)).
    - rotation: int, rotation for x-tick labels (default: 90).
    - grid: bool, whether to show grid lines (default: False).
    - vmin: float or None, minimum value for colormap normalization (default: None).
    - vmax: float or None, maximum value for colormap normalization (default: None).
    - center: float or None, center value for divergent colormaps (default: None).
    - linewidths: float, width of lines between cells (default: 0.5).
    - linecolor: str, color of lines between cells (default: 'white').
    - label_col: str, name for the label column (default: 'label').
    - sort_by_label: bool, whether to sort rows by labels (default: True).
    - font_family: str, font family for text (default: 'Arial').
    
    Returns:
    None (displays or saves the plot).
    """
    # Update rcParams for font
    plt.rcParams.update({
        'font.family' : 'Arial',        
        'axes.labelsize': 12,          
        'axes.titlesize': 14,     
        'xtick.labelsize': 10,   
        'ytick.labelsize': 10,    
        'legend.fontsize': 10,     
        'figure.dpi': 300,         
        'savefig.dpi': 300,         
        'axes.linewidth': 0.5,       
        'lines.markersize': 4,     
        'legend.frameon': False,     
    })
    # Prepare data
    feat_list = top_features['Features'].tolist()[:topk]  # Select topk features
    global_exp = test_data[feat_list].copy()
    global_exp['label'] = test_labels
    if sort_by_label:
        global_exp = global_exp.sort_values('label')
    heatmap_data = global_exp[feat_list]  # Drop label for heatmap
    fig, ax = plt.subplots(figsize=fig_size)
    
    sns.set_style("whitegrid") if grid else sns.set_style("white")
    
    sns.heatmap(
        heatmap_data,
        annot=annot,
        fmt=fmt,
        cmap=cmap,
        cbar_kws={'label': cbar_label, 'shrink': 0.8},
        ax=ax,
    )
    
    plt.xticks(rotation=rotation, ha='right', fontsize=8)
    plt.yticks(fontsize=8)
    ax.set_xlabel('Features', fontsize=12, fontweight='bold')
    ax.set_ylabel('Samples (sorted by label)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold')
    
    
    if sort_by_label:
        unique_labels = sorted(global_exp['label'].unique())
        n_labels = len(unique_labels)
        label_colors = sns.color_palette('tab10', n_labels) if n_labels <= 10 else sns.color_palette('husl', n_labels)
        label_map = dict(zip(unique_labels, label_colors))
        
        # Add color sidebar on the left
        bar_width = 1
        for i in range(len(global_exp)):
            label = global_exp.iloc[i]['label']
            color = label_map[label]
            ax.add_patch(plt.Rectangle((-bar_width-0.2, i), bar_width, 1, color=color, clip_on=False, linewidth=0))
        
        ax.set_xlim(-bar_width, len(feat_list))
        
        handles = [plt.Rectangle((0, 0), 1, 1, color=label_map[label]) for label in unique_labels]
        legend_labels = [str(label) for label in unique_labels]
        ax.legend(handles, legend_labels, title='Label', loc='upper left', 
                  bbox_to_anchor=(1.05, 1.05), fontsize=10, title_fontsize=12)

    plt.tight_layout()
    
    # Save or show
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(f'{save_path}/{title}.pdf', bbox_inches='tight', dpi=300)
        plt.show()
        plt.close(fig) 
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()



def plt_tsne_embedding(data1_pca, data2_pca, data3_pca, data4_pca, final_embedding, _save=False, result_path='./'):

    plt.figure(figsize=(10, 7))
    sns.set(style="whitegrid", palette="muted", font_scale=1.2)

    plot_config = {
        "final": {"color": "#E64B35", "marker": "o", "s": 80},
        "global": {"color": "#4DBBD5", "marker": "s", "s": 80},
        "gnn": {"color": "#00A087", "marker": "^", "s": 80},
        "omics": {"color": "#3C5488", "marker": "D", "s": 80}
    }

    for label, config in plot_config.items():
        data = eval(f"data{['final','global','gnn','omics'].index(label)+1}_pca")
        plt.scatter(data[:,0], data[:,1],
                    c=config["color"],
                    marker=config["marker"],
                    s=config["s"],
                    edgecolor='w',
                    linewidth=0.5,
                    alpha=0.8,
                    label=label)

    plt.xlabel("Principal Component 1", fontsize=14, labelpad=10)
    plt.ylabel("Principal Component 2", fontsize=14, labelpad=10)
    plt.title("Multi-view Feature Space Projection", fontsize=16, pad=20)

    legend = plt.legend(title="Model Type",
                    title_fontsize=12,
                    fontsize=11,
                    frameon=True,
                    loc='upper right',
                    bbox_to_anchor=(1.18, 1),
                    labelspacing=1.2,
                    borderpad=1)

    legend.get_frame().set_facecolor('#F5F5F5')
    legend.get_frame().set_edgecolor('#404040')

    plt.grid(True, linestyle='--', alpha=0.6)
    sns.despine(offset=10, trim=True)

    plt.tight_layout()


    if  not os.path.exists(result_path):
        os.makedirs(result_path)
    if _save:
        save_name = os.path.join(result_path, 'pca_projection.pdf')
        plt.savefig(save_name, dpi=300, bbox_inches='tight')
    plt.show()


