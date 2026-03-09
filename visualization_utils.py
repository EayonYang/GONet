#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/2/21 15:30
# @Author  : Shark
# @Site    : 
# @File    : visualization_utils.py
# @Software: PyCharm
# @logit:


import torch
import re
import glob
import os
import pandas as pd
import numpy as np
import seaborn as sns

from colorsys import rgb_to_hls, hls_to_rgb
from scipy.stats import ttest_ind
import scipy.stats as stats
import scikit_posthocs as sp 
from scipy.stats import friedmanchisquare, wilcoxon, ttest_ind
from statannotations.Annotator import Annotator
from statsmodels.stats.multitest import multipletests

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize  # 用于bar颜色渐变
from matplotlib import gridspec 
from matplotlib.patches import Patch
from matplotlib.table import table,Table
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from matplotlib.colors import to_rgb
from matplotlib.patches import Polygon
from matplotlib.path import Path
from matplotlib.spines import Spine
from matplotlib.projections.polar import PolarAxes
import matplotlib.colors as mcolors
from matplotlib.projections import register_projection
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.patches as patches
from scipy.stats import ttest_ind, mannwhitneyu, wilcoxon

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap
from sklearn.impute import KNNImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.metrics import (accuracy_score, roc_auc_score,average_precision_score, precision_recall_curve,
                             roc_curve, classification_report,
                             precision_score, recall_score, f1_score,silhouette_score, davies_bouldin_score)

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'Arial'


_PALETTES = dict(
    tight_9      = ['#a1a9d0', "#add0d9", "#B2D4F8","#f6c4e1",  "#B777C8","#d5eae7","#d8c1e4", "#ffd8d3","#87C8E2","#CCC9E6", '#E4C755','#476D87'],
    zeileis_28   = ["#023fa5", "#7d87b9", "#bec1d4", "#d6bcc0", "#bb7784", 
                    "#8e063b", "#4a6fe3", "#8595e1", "#b5bbe3", "#e6afb9", "#e07b91", "#d33f6a", "#11c638", "#8dd593", "#c6dec7", "#ead9c6", "#f0b98d", "#ef9708", "#0fcfc0", "#9cded6", "#d5eae7", "#f3e1eb", "#f6c4e1", "#f79cd4",'#7f7f7f', "#c7c7c7", "#1CE6FF", "#336600"],
    ting_36     =  ['#E5D2DD', '#53A85F', '#F1BB72', '#F3B1A0', '#D6E7A3',
                    '#57C3F3', '#476D87', '#E95C59', '#E59CC4', '#AB3282', '#23452F', '#BD956A', '#8C549C', '#585658', '#9FA3A8', "#F0C39F", '#5F3D69', '#C5DEBA', '#58A4C3', '#E4C755', '#F7F398', '#AA9A59', '#E63863', '#E39A35', '#C1E6F3', '#6778AE', '#91D0BE', '#B53E2B', '#712820', '#DCC1DD', 
                    '#CCE0F5',  '#CCC9E6', '#625D9E', '#68A180', '#3A6963','#968175']
)



def selected_data(df, cancer_col='cancer'):
    """统一数据预处理流程"""
    filtered_data = []
    cancer_list = df[cancer_col].unique().tolist()
    for cancer in cancer_list:
        subset = df[df[cancer_col] == cancer]
        subset['Cancer Type'] = cancer
        filtered_data.append(subset)
    return pd.concat(filtered_data, axis=0)


def format_table(df, col, index):
    summary = (
        df.groupby([index, col])['roc_auc']
        .agg(['mean', 'std'])
        .reset_index()
    )
    summary['mean_std'] = summary['mean'].round(4).astype(str) + ' ± ' + summary['std'].round(4).astype(str)
    result_table = summary.pivot(index=index, columns=col, values='mean_std')
    result_table = result_table.reset_index()
    return result_table

def format_mean_std(x):
    mean_val = x.mean()
    std_val = x.std()
    return f"{mean_val:.6f}±{std_val:.6f}" 

def summarize_method_performance(df, method_col='methods', f1_col='f1', auc_col='roc_auc'):
    """calculate F1 and AUC on average for each methods"""
    summary = (
        df.groupby(method_col)
        .agg(
            f1_mean=(f1_col, 'mean'),
            f1_sem=(f1_col, 'sem'),
            auc_mean=(auc_col, 'mean'),
            auc_sem=(auc_col, 'sem')
        )
        .reset_index()
    )
    return summary

def create_marker_color_maps(methods, palette, markers=None):
    """generate shape and color for each methods """
    if markers is None:
        markers = ['o', 's', '^', 'D', 'P', '*', 'x', 'h', '>']
    return (
        {m: mk for m, mk in zip(methods, markers)},
        {m: c for m, c in zip(methods, palette)}
    )

################################# plot function ###############################

def adjust_lightness(color, factor):
    r, g, b = to_rgb(color)
    h, l, s = rgb_to_hls(r, g, b)
    l = max(0, min(1, l * factor))
    r_new, g_new, b_new = hls_to_rgb(h, l, s)
    return (r_new, g_new, b_new)


def create_visualization(
    data,
    plot_types,
    group_name='module',
    y_name='roc_AE',
    save_path = './',
    module_order=None,
    mode_colors=None,
    background_colors=None,
    _annotate_sigification=True,
    _save = False
):
    """
    Create a multi-panel visualization with customizable plot types for each module.
    
    Parameters:
    - data_path (str): Path to the CSV file with columns 'module', 'mode', 'auc'.
    - plot_types (dict): Dictionary mapping module names to plot types ('box' or 'hist').
    - module_order (list, optional): Order of modules for plotting.
    - mode_colors (dict, optional): Colors for each mode.
    - p_values (dict, optional): P-values to display on each panel.
    - background_colors (dict, optional): Background colors for each panel.
    
    Returns:
    - fig: Matplotlib figure object.
    """
    NATURE_STYLE = {
        'font.sans-serif': 'Arial',
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'axes.linewidth': 1.0,
        'lines.linewidth': 1.0,
        'pdf.fonttype': 42,
        'figure.dpi': 300,
    }
    
    plt.rcParams.update(NATURE_STYLE)
    if module_order is None:
        module_order = data[group_name].unique()
    modes = data['mode'].unique()
    
    if mode_colors is None:
        mode_colors = dict(zip(modes, sns.color_palette('husl', len(modes))))
    sns.set_style('white')
    sns.set_context('paper', font_scale=1.2)
    
    modules = data[group_name].unique()
    fig, axes = plt.subplots(1, len(module_order), figsize=(len(modules) * 5, 6), sharey=False)
    if len(module_order) == 1:
        axes = [axes]
        
    for i, module in enumerate(module_order):
        ax = axes[i]
        module_data = data[data[group_name] == module]
        modes = module_data['mode'].unique()
        mode_order = sorted(modes)
        plot_type = plot_types.get(module, 'box') 
        
        # ==== 1. 添加渐变背景色 ====
        if background_colors and module in background_colors:
            base_color = background_colors[module]
        else:
            base_color = '#DDDDDD'
        gradient = np.linspace(1.5, 0.8, 256).reshape(-1, 1)
        cmap_colors = [adjust_lightness(base_color, f) for f in gradient.flatten()]
        gradient_rgb = np.array(cmap_colors).reshape(-1, 1, 3)
        ax.imshow(gradient_rgb, aspect='auto', extent=(0, 1, 0, 1), transform=ax.transAxes, zorder=0)
        
        # ==== 2. 主图绘制 ====
        if plot_type == 'box':
            # Box plot for each mode
            sns.boxplot(x='mode', y=y_name, data=module_data, ax=ax, 
                        palette=mode_colors,
                        order=mode_order,
                        boxprops=dict(edgecolor='black'), 
                        whiskerprops=dict(color='black'),
                        capprops=dict(color='black'), 
                        medianprops=dict(color='black'))
        elif plot_type == 'bar':
            grouped = module_data.groupby('mode')[y_name].agg(['mean', 'sem']).reset_index()
            grouped['mode'] = pd.Categorical(grouped['mode'], categories=mode_order, ordered=True)
            grouped = grouped.sort_values('mode').reset_index(drop=True)
            ax.bar(
                x=grouped['mode'],
                height=grouped['mean'],
                yerr=grouped['sem'],
                color=[mode_colors[mode] for mode in grouped['mode']],
                alpha=0.8,
                edgecolor='black',
                capsize=5
            )
        elif plot_type == 'violin':
            sns.violinplot(x='mode', y=y_name, data=module_data, ax=ax,
                           palette=mode_colors, inner='box', order=mode_order)
            
        # ==== 3. 标题和彩色条 ====
        ax.set_title(module.upper(), pad=18, fontsize=14, weight='bold')
        # bar_color = background_colors.get(module, "#E8A6A6") if background_colors else "#CA3535"
        rect = Rectangle((0, 1.05), 1, 0.08, transform=ax.transAxes,
                         color=background_colors[module], clip_on=False)
        ax.add_patch(rect)
        
        # —— 3. 坐标轴 & 背景 —— #
        ax.set_xlabel('')
        ax.set_ylabel('AUROC')
        y_max_val = module_data[y_name].max()
        y_min_val = module_data[y_name].min()
        y_padding = (y_max_val - y_min_val) * 0.15
        ax.set_ylim(y_min_val, y_max_val + y_padding)
        
        if background_colors and module in background_colors:
            ax.set_facecolor(background_colors[module])
        edge_col = background_colors.get(module, 'black') if background_colors else 'black'
        ax.patch.set_edgecolor(edge_col)
        ax.patch.set_linewidth(1.5)

        if background_colors and module in background_colors:
            ax.patch.set_edgecolor(background_colors[module])
            ax.patch.set_linewidth(2)
        else:
            ax.patch.set_edgecolor('black')
            ax.patch.set_linewidth(1)
            

        if module=='Global':
            all_base = 'AE'
        elif module=='Local_molecular' or module=='Local-molecular':
            all_base = 'DGI'
        elif module=='Local_omics' or module=='Local-omics':
            all_base = 'Omics'
        elif module=='Fusion':
            all_base = 'GRU_fusion'
        else:
            all_base = 'all'
        mean_auc = module_data[module_data['mode'] == all_base][y_name].mean()
        table_data = []
        for mode in mode_order:
            mode_data = module_data[module_data['mode'] == mode][y_name]
            mode_mean, mode_val = mode_data.mean(),mode_data.std()
            delta = mode_mean - mean_auc
            table_data.append((mode, delta, f'{round(mode_mean,4)}±{round(mode_val,4)}'))
        print('table_data',table_data)
        if table_data:
            table_vals = [
                [f'{delta:.4f}' for _, delta, _ in table_data],
                [f'{n_val}' for _, _, n_val in table_data]
            ]
            colors = []
            for row in range(3):
                row_colors = []
                for col in range(len(table_data)):
                    delta_val = table_data[col][1]
                    factor = 0.9 - min(abs(delta_val), 1) * 0.5
                    shade_color = adjust_lightness(base_color, factor)
                    row_colors.append(shade_color if row == 0 else 'white')
                colors.append(row_colors)
                
            table = Table(ax, bbox=[0, -0.35, 1, 0.2])  # 调整 bbox 以控制表格位置
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            num_cols = len(table_data)
            cell_width = 0.85 / num_cols
            for (row, col), val in np.ndenumerate(table_vals):
                color = colors[row][col]
                table.add_cell(row, col, width=cell_width, height=0.05, text=val, loc='center', facecolor=color)
            ax.add_table(table)
            ax.text(-0.02, -0.20, 'delta', ha='right', va='center', transform=ax.transAxes) # fontsize=10
            ax.text(-0.02, -0.30, 'value', ha='right', va='center', transform=ax.transAxes) # fontsize=10
        
        # Beautify: Remove top and right spines, adjust ticks
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', which='major') # labelsize=10
        
    # Adjust layout for better spacing
    plt.tight_layout(pad=2.0)
    fig.subplots_adjust(bottom=0.3)
    if _save:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig




def plot_circular_bar_chart(df, y_name, group_name, mode_colors, cancer_colors, gap_factor=1.2, save_path='./', _save=False):
    """
    Plot a circular bar chart with gaps between cancer regions and outer bars for distinction.
    
    Parameters:
    - df: DataFrame with 'cancer', 'mode', and 'auc' columns
    - colors: dict mapping modes to colors
    - cancer_colors: dict mapping cancer types to colors for outer bars
    - gap_factor: float to adjust gap size between cancer regions
    """
    # Get unique cancer types and modes
    # cancer_df = df[df['mode'] == 'all'][y_name].mean()
    # cancer_df = df.groupby('mode')[y_name].agg(['mean', 'sem']).reset_index()
    # cancers = cancer_df['cancer'].unique()
    # modes = cancer_df['mode'].unique()
    # num_cancers = len(cancers)
    # num_modes = len(modes)
    plt.rcParams['font.family'] = 'Arial'
    df_grouped = df.groupby(['cancer', group_name])[y_name].mean().reset_index()
    # print(df_grouped)
    
    cancers = df_grouped['cancer'].unique()
    modes = df_grouped[group_name].unique()
    num_cancers = len(cancers)
    num_modes = len(modes)

    # Calculate angles with gaps
    theta = np.linspace(0, 2 * np.pi, num_cancers, endpoint=False)
    width = 2 * np.pi / (num_cancers + gap_factor)  # Adjusted width for gaps

    # Create polar plot
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(6, 6))
    # cancer_colors = [plt.cm.tab20(j % 20) for j in range(num_cancers)]
    
    # Plot each cancer region
    for i, cancer in enumerate(cancers):
        ax.bar(theta[i], 1, width=width, color=cancer_colors[cancer], alpha=0.3, edgecolor='none')
        # Get data for this cancer
        cancer_data = df_grouped[df_grouped['cancer'] == cancer]
        
        # Calculate bar positions within the cancer region
        bar_width = width / num_modes
        for j, (_, row) in enumerate(cancer_data.iterrows()):
        # for j, row in cancer_data.iterrows():
            # mode = row.mode
            # auc = row.roc_auc
            mode = row[group_name]
            auc = row[y_name] 
            bar_theta = theta[i] - (num_modes / 2 - 0.5 - j) * bar_width
            # Plot inner bar for AUC
            ax.bar(bar_theta, auc, width=bar_width * 0.9, color=mode_colors[mode], edgecolor='black', linewidth=0.5)
            ax.text(bar_theta, auc + 0.05, f'{auc:.2f}', ha='center', va='bottom', fontsize=10)

        # Add outer bar to distinguish cancer type
        ax.bar(theta[i], 0.1, width=width, bottom=1.05, color=cancer_colors[cancer], edgecolor='black', linewidth=0.5)
        
        # Add cancer label
        ax.text(theta[i], 1.1, cancer, ha='center', va='center', fontsize=10, rotation=np.degrees(theta[i]) - 90)

    # Customize plot
    ax.set_ylim(0, 1.3)
    ax.set_xticks([])  # Hide angular ticks
    ax.set_yticklabels([])
    ax.spines['polar'].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=mode_colors[mode]) for mode in modes]
    
    fig.legend(handles, modes, loc='upper right', title='Modes', fontsize=10)
    plt.title('Cancer Types with Mean-AUROC Values by Mode', pad=20, fontsize=12)
    plt.tight_layout()
    plt.show()
    if _save:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')


def plot_grouped_boxplot_with_significance(
    df,
    group_col='cancer',
    category_col='mode',
    value_col='auc',
    ref_category='all',
    save_path = './',
    palette='deep',
    figsize=(12, 10),
    point_jitter=0.25,
    _annotations_signation=False,
    _save = False
):

    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 10
    
    # 确保 ref_category 存在
    if ref_category not in df[category_col].unique():
        raise ValueError(f"`ref_category={ref_category}` is not existed in `{category_col}` ")
    
    cancers = df[group_col].unique()
    modes = df[category_col].unique()
    num_modes = len(modes)

    mode_colors = palette#dict(zip(modes, pal))
    
    fig = plt.figure(figsize=figsize)
    ax = sns.boxplot(
        x=group_col, y=value_col, hue=category_col,
        data=df, palette=mode_colors, fliersize=0,
        linewidth=1.2
    )
    
    for i, cancer in enumerate(cancers):
        sub = df[df[group_col] == cancer]
        for j, mode in enumerate(modes):
            vals = sub[sub[category_col] == mode][value_col].values
            if len(vals) == 0:
                continue
            x = i + (j - (len(modes) - 1) / 2) * point_jitter
            mean_val = vals.mean()
            plt.scatter(
                np.full_like(vals, x), vals,
                color=mode_colors[mode], 
                s=45, alpha=0.85, zorder=10, linewidth=0.3
            )

            ax.plot([x],[mean_val], marker='D', color='darkred', markersize=6, label='mean' if i==0 and j==0 else "", zorder=11)
            
            ax.text(x, mean_val + 0.002, f'{mean_val:.2f}', ha='center', va='bottom', fontsize=10, color='dimgray',fontweight='bold', zorder=12)
              
    for idx in range(1, len(cancers)):
        ax.axvline(x=idx-0.5, color="#CFA877", linestyle='--', linewidth=1)
        # ax.plot([mean_val], [j], marker='D', color='darkred', markersize=6, label='mean' if i==0 and j==0 else "")
    
    ax.set_xlabel("Cancer Type", fontsize=12)
    ax.set_ylabel("AUC", fontsize=12)
    ax.set_ylim(bottom=0.15)
    ax.yaxis.grid(False)
    ax.tick_params(axis='x', labelrotation=0, labelsize=10)
    ax.tick_params(axis='y', labelsize=10)

    sns.despine()
    ax.legend(title=category_col, fontsize=10, title_fontsize=10,
              loc='upper right', frameon=False)

    plt.tight_layout()
    plt.show()
    if _save:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')



def plot_grouped_barplot_with_stats(
    df,
    group_col='cancer',
    category_col='mode',
    value_col='auc',
    ref_category='all',
    save_path = './',
    palette='Set2',
    figsize=(12, 10),
    bar_width=0.18,
    group_spacing=0.2,  # ✅ 控制组间间隔
    show_delta=True,
    show_signification=False,
    _save=False
):
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 10
    sns.set(style="whitegrid")
    fig = plt.figure(figsize=figsize)

    groups = df[group_col].unique()
    categories = df[category_col].unique()
    num_categories = len(categories)
    category_idx = {cat: i for i, cat in enumerate(categories)}
    category_colors=palette

    # 创建x轴位置
    x_pos = []
    tick_labels = []
    base = 0
    for group in groups:
        for j in range(num_categories):
            x_pos.append(base + j * bar_width)
            tick_labels.append(group)
        base += num_categories * bar_width + group_spacing 

    ax = plt.gca()
    x_centers = []

    for i, group in enumerate(groups):
        sub = df[df[group_col] == group]
        x0 = i * (num_categories * bar_width + group_spacing)
        x_group_center = []
        ax.axvline(x=x0+0.4, color="#CFA877", linestyle='--', linewidth=1)

        for j, cat in enumerate(categories):
            xpos = x0 + j * bar_width
            sub_vals = sub[sub[category_col] == cat][value_col].values
            if len(sub_vals) == 0:
                continue

            mean = np.mean(sub_vals)
            std = np.std(sub_vals)
            ax.bar(xpos, mean, yerr=std, width=bar_width * 0.9,
                   color=category_colors[cat], edgecolor='black', capsize=3, label=cat if i == 0 else "")
            ax.scatter(np.full_like(sub_vals, xpos), sub_vals, color=category_colors[cat], s=30, zorder=10, alpha=0.7)
            ax.text(xpos, mean + std + 0.01, f'{mean:.2f}', ha='center', va='bottom', fontsize=10, zorder=11)
            x_group_center.append(xpos)

        x_centers.append(np.mean(x_group_center))

        # 显著性 + Δ 标注
        if ref_category not in sub[category_col].values:
            continue
        ref_vals = sub[sub[category_col] == ref_category][value_col].values
        x_ref = x0 + category_idx[ref_category] * bar_width

        for cat in categories:
            if cat == ref_category or cat not in sub[category_col].values:
                continue
            comp_vals = sub[sub[category_col] == cat][value_col].values
            if len(ref_vals) > 1 and len(comp_vals) > 1:
                stat, pval = ttest_ind(ref_vals, comp_vals, equal_var=False)
                star = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else 'ns'

                x1 = x0 + category_idx[ref_category] * bar_width
                x2 = x0 + category_idx[cat] * bar_width
                y1 = np.max(ref_vals)
                y2 = np.max(comp_vals)
                height = max(y1, y2) + 0.05

                # 显著性线
                if show_signification:
                    ax.plot([x1, x1, x2, x2], [height-0.005, height, height, height-0.005], color='black', lw=1)
                    ax.text((x1 + x2)/2, height + 0.005, star, ha='center', va='bottom', fontsize=10)

                # Δ 标注在两者最大值中点上
                if show_delta:
                    delta = np.mean(comp_vals) - np.mean(ref_vals)
                    delta_y = height + 0.02
                    ax.text((x1 + x2)/2, delta_y, f'Δ={delta:.2f}', ha='center', fontsize=10, color='dimgray')
    
    # x轴标签
    ax.set_xticks(x_centers)
    ax.set_xticklabels(groups, fontsize=12)

    # 其余美化
    ax.set_ylabel(value_col.upper(), fontsize=12)
    ax.set_xlabel(group_col.capitalize(), fontsize=12)
    ax.tick_params(axis='y', labelsize=10)
    ax.yaxis.grid(False)
    sns.despine()
    ax.grid(False)

    
        
    # 图例
    ax.legend(title=category_col.capitalize(), fontsize=10, title_fontsize=11, frameon=False)
    plt.tight_layout()
    plt.show()
    if _save:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')



def compare_cancer_specific_methods_performance(df, _save=False, path='./'):
    """
    绘制不同癌症类型下各方法的性能对比条形图，包含误差条和均值标签。
    
    参数:
    - GNN_methods_selected_cancer_specific: 包含 'cancer', 'methods', 'accuracy' 列的 DataFrame。
    """
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 10
    sns.set(style="whitegrid")

    # 计算每个cancer-method组合的均值和标准差
    grouped = df.groupby(['cancer', 'methods']).agg(mean_accuracy=('accuracy', 'mean'), std_accuracy=('accuracy', 'std')).reset_index()

    # 动态获取唯一的方法和癌症列表
    unique_methods = sorted(grouped['methods'].unique())
    unique_cancers = sorted(grouped['cancer'].unique())

    # 准备绘图数据
    pivot_mean = grouped.pivot(index='cancer', columns='methods', values='mean_accuracy')
    pivot_std = grouped.pivot(index='cancer', columns='methods', values='std_accuracy')

    # 定义颜色方案，使用husl调色板为每个方法分配不同颜色
    colors = sns.color_palette("Set2", len(unique_methods))

    # 创建条形图
    x = np.arange(len(unique_cancers))  # cancer位置
    num_methods = len(unique_methods)
    width = 0.8 / num_methods  # 动态计算条形宽度以适应多个方法

    fig, ax = plt.subplots(figsize=(12, 6))  # 增大figure大小以适应更多方法

    # 动态绘制每个方法的条形，并添加误差条
    for i, method in enumerate(unique_methods):
        offset = (i - num_methods / 2 + 0.5) * width
        rects = ax.bar(x + offset, pivot_mean[method], width, yerr=pivot_std[method], label=method, capsize=5, color=colors[i])
        
        # 在条形上添加数值标签（均值）
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.4f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    # 添加标签、标题等
    ax.set_xlabel('Cancers')
    ax.set_ylabel('Accuracy')
    ax.set_title('Performance of Methods on Different Cancers (with 5 Replicates)')
    ax.set_xticks(x)
    ax.set_xticklabels(unique_cancers)
    ax.set_ylim(0, 1)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))  # 将图例放在右上角外部以避免重叠

    plt.tight_layout()
    plt.show()
    if _save:
        fig.savefig(path, dpi=300, bbox_inches='tight')




def plot_advanced_radar(
    df,
    group_col='cancer',
    category_col='mode',
    value_col='auc',
    modes_to_plot=None,
    palette='Accent',
    cancer_palette='tab10',
    figsize=(8, 6),
    fill_alpha=0.2,
    line_width=3,
    point_size=80,
    save_path='./',
    _save=False
    
):
    """
    复杂高级雷达图：
      1. 隐藏径向度数
      2. 轴线从中心延伸并标注癌症名称
      3. 点上显示具体均值
      4. 渐变背景圆环，阴影，双线风格
    """

    # 准备数据
    cancers = df[group_col].unique().tolist()
    if modes_to_plot is None:
        modes_to_plot = df[category_col].unique()[:2]
    assert len(modes_to_plot) == 2, "请指定正好两个 mode"

    N = len(cancers)
    # 角度
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # 封闭

    # 颜色板
    pal = palette#sns.color_palette(palette, len(modes_to_plot))
    fig, ax = plt.subplots(1, 1, subplot_kw=dict(polar=True), figsize=figsize)
    ax.set_facecolor("#f7f7f7")
    
    # 1. 渐变背景圆环
    for i, (r_frac, color) in enumerate(zip([0.2, 0.4, 0.6, 0.8, 1.0],
                                            ["#e0f0ff", "#d6e8ff", "#f9eded", "#fbe4e4", "#f8dede"])):
        circle = plt.Circle((0, 0), r_frac, transform=ax.transData._b, color=color, alpha=0.2, zorder=0)
        ax.add_artist(circle)
    max_val = df[value_col].max() * 1.05

    # 2. 每个轴线 & 标签
    for angle, cancer in zip(angles[:-1], cancers):
        ax.plot([angle, angle], [0, max_val], color='gray', lw=0.7, linestyle='--', alpha=0.6)
        # ax.text(angle, max_val * 1.05, cancer, ha='center', va='center', fontsize=11,
        #         fontweight='bold', rotation=np.degrees(angle) - 90, rotation_mode='anchor')
        x = angle
        y = max_val * 1.05
        label_circle_color = cancer_palette[cancer]

        # 添加圆背景（fancy）
        bbox_props = dict(boxstyle="circle,pad=0.35", fc=label_circle_color, ec="none", alpha=0.9)
        ax.text(x, y, cancer, color='dimgray',
                ha='center', va='center', fontsize=10, fontweight='bold',
                rotation=np.degrees(angle) - 90,
                rotation_mode='anchor',
                bbox=bbox_props)

    # 3. 绘制两条雷达线和填充
    for idx, mode in enumerate(modes_to_plot):
        # 提取当前 mode 下所有 cancer 的均值序列
        vals = [df[(df[group_col]==c) & (df[category_col]==mode)][value_col].mean() for c in cancers]
        data = vals + vals[:1]
        print(data)
        
        # 主线
        ax.plot(angles, data, color=pal[mode], linewidth=line_width, 
                linestyle='--' if idx==0 else '-', label=f"{mode} (mean={np.mean(vals[:-1]):.3f})")
        # 渐变填充（两种模式不同透明度）
        ax.fill(angles, data, color=pal[mode], alpha=fill_alpha*(1 - idx*0.3))

        # 4. 在每个点上显示具体数值
        for angle, val in zip(angles, vals):
            ax.scatter(angle, val, s=point_size, color=palette[mode], zorder=5)
            ax.text(angle, val, f'{val:.2f}', ha='center', va='bottom',
                    fontsize=13.5, color=palette[mode], fontweight='bold')

    # 5. 美化：隐藏径向度数，保留同心网格
    ax.set_yticklabels([])
    ax.set_xticks([])

    ax.spines['polar'].set_visible(False)
    ax.grid(color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
    ax.set_ylim(0, max_val * 1.15)


    ax.legend(loc='upper right', bbox_to_anchor=(1., 1.), fontsize=10, title=category_col)
    plt.tight_layout()
    plt.show()
    # fig.savefig(, dpi=300, bbox_inches='tight')
    if _save:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')


def plot_horizontal_grouped_box(df, value_col='auc', group_col='cancer', category_col='mode', palette='Set2', cancer_palette='tab10', save_path='./', _save=False):
    sns.set(style='whitegrid', context='talk')
    
    cancers = df[group_col].unique()
    n = len(cancers)
    
    fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(8, n  ), sharex=True)
    if n == 1:
        axes = [axes]

    mode_order = df[category_col].unique()
    cancer_type = df['cancer'].unique()
    # box_colors = palette # sns.color_palette(palette, len(mode_order))
    cancer_colors = sns.color_palette(cancer_palette, len(cancer_type))

    for i, cancer in enumerate(cancers):
        ax = axes[i]
        sub_df = df[df[group_col] == cancer]
        omics_colors = sns.light_palette(cancer_colors[i], n_colors=len(sub_df['omics'].unique()) + 2, reverse=False)[1:-1]
        
        # 横向箱线图
        sns.boxplot(data=sub_df, y=category_col, x=value_col, ax=ax, 
                    order=mode_order,
                    palette=omics_colors, width=0.5, fliersize=0, boxprops=dict(linewidth=1), saturation=0.9)
        
        # 添加点图（重复实验）
        sns.stripplot(data=sub_df, y=category_col, x=value_col, ax=ax, 
                      order=mode_order,
                      color='black', size=3, jitter=0.15, alpha=0.8)

        # 添加均值线
        for j, mode in enumerate(mode_order):
            mode_vals = sub_df[sub_df[category_col] == mode][value_col]
            mean_val = mode_vals.mean()
            ax.plot([mean_val], [j], marker='D', color='darkred', markersize=6, label='mean' if i==0 and j==0 else "")
            ax.text(mean_val + 0.01, j + 0.1, f'{mean_val:.2f}', color='darkred', fontsize=8)

        # 显著性检验（两组之间）
        if len(mode_order) == 2:
            vals1 = sub_df[sub_df[category_col] == mode_order[0]][value_col]
            vals2 = sub_df[sub_df[category_col] == mode_order[1]][value_col]
            stat, pval = ttest_ind(vals1, vals2)
            sig_label = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else 'ns'

            # 连线 + 显著性标注
            y1, y2 = 0, 1
            max_val = max(sub_df[value_col].max(), 1.0)
            line_y = max_val + 0.05
            ax.plot([vals1.mean(), vals2.mean()], [y1 + 0.25, y2 + 0.25], lw=1.5, c='gray')
            # ax.text((vals1.mean() + vals2.mean()) / 2, y2 + 0.3, sig_label, ha='center', fontsize=10, color='black')

        ax.annotate(f" {cancer} ",
            xy=(0.5, 1.05), xycoords='axes fraction',
            ha='center', va='bottom', color='#AAAAAA',
            fontsize=11, weight='bold',
            backgroundcolor='white', zorder=5)
        # 样式调整
        # ax.set_title(f"{cancer}", fontsize=12, loc='left', pad=10, weight='bold')
        ax.set_ylabel('')
        ax.set_xlabel('')
        ax.grid(False)
        ax.set_yticklabels(mode_order, fontsize=10)
        ax.set_facecolor('white') # 设置子图干净的背景

        # 去掉顶部和右侧线条
        # sns.despine(ax=ax, left=False, bottom=False)
        ymin, ymax = ax.get_ylim()
        ax.spines['top'].set_position(('outward', 9))
        ax.spines['left'].set_bounds(ymin, ymax-0.2)
        ax.spines['right'].set_bounds(ymin, ymax-0.2)
        ax.spines['left'].set_visible(True)
        ax.spines['right'].set_visible(True)
        for spine in ['top', 'left', 'right', 'bottom']:
            ax.spines[spine].set_color('#AAAAAA')  # 自定义颜色
            ax.spines[spine].set_linewidth(1)
        
    # X轴设置
    axes[-1].set_xlabel(value_col.upper(), fontsize=14)
    
    # legend
    mean_patch = mlines.Line2D([], [], color='darkred', marker='D', linestyle='None', markersize=6, label='Mean')
    plt.legend(handles=[mean_patch], loc='lower right', frameon=False)

    plt.tight_layout(h_pad=1)
    plt.show()
    if _save:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')



##################################################################
#                                                                #
#                       以下是针对omics部分的绘图代码                #
#                                                                #  
##################################################################


def plot_upset(df, all_omics_types, palette='husl', y_col = 'roc_auc',save_path='./', SG='mRNA_protein_miRNA',  _save=False):
    """
    Plots model recall across cancer types and omics combinations with a combination matrix.
    
    Parameters:
    - df: DataFrame containing 'cancer', 'omics', and 'auc' columns.
    - palette: Color palette to use for cancer types (default is 'husl').
    """
    # Define all possible omics types
    
    # Group data to calculate mean and std for auc
    grouped = df.groupby(['cancer', 'omics'])[y_col].agg(['mean', 'std']).reset_index()
    
    # Set Seaborn theme and font
    sns.set_theme(style="white")
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    
    # Create FacetGrid
    g = sns.FacetGrid(grouped, col='cancer', sharey=False, height=5, aspect=0.8, col_wrap=4)
    g.set_titles("")
    g.set_yticklabels("")
    # Get unique cancer types and assign colors
    cancer_types = grouped['cancer'].unique()
    cancer_colors = sns.color_palette(palette, len(cancer_types))
    
    # Plot each subplot
    for i, ax in enumerate(g.axes.flat):
        cancer_type = cancer_types[i]
        sub_df = grouped[grouped['cancer'] == cancer_type]
        sub_df_raw = df[df['cancer'] == cancer_type]
        
        # Generate shades for omics combinations
        omics_colors = sns.light_palette(cancer_colors[i], n_colors=len(sub_df['omics'].unique()) + 2, reverse=True)[1:-1]
        
        # Plot bars with error bars
        x_coords = np.arange(len(sub_df))
        y_coords = sub_df['mean']
        errors = sub_df['std']
        max_y = 0
        for j, (x, y, err, omic) in enumerate(zip(x_coords, y_coords, errors, sub_df['omics'])):
            ax.bar(x, y, color=omics_colors[j], edgecolor='black', linewidth=1)
            ax.errorbar(x, y, yerr=err, fmt='none', c='black', capsize=3)
            ax.text(x, y + err + 0.005, f'{y:.3f}', ha='center', va='bottom', fontsize=10, color='black')
            max_y = max(max_y, y + err + 0.01)

        ax.set_ylim(0, max_y + 0.05)
        ax.set_ylabel('mean auc', fontsize=14)
        ax.axhline(y=0.07, color='grey', linestyle='--', linewidth=1)
        
        # Remove grid lines
        ax.grid(False)
        
        # Hide x-axis labels and ticks
        ax.set_xlabel('')
        ax.set_xticks(x_coords)
        ax.set_xticklabels([])
        ax.tick_params(axis='x', length=0)
        
        # Draw combination matrix
        y_positions = np.linspace(-0.05, -0.3, num=len(all_omics_types))
        ax.set_ylim(bottom=min(y_positions) - 0.01)
        for y_pos, omic_name in zip(y_positions, all_omics_types):
            ax.text(-0.8, y_pos, omic_name, ha='right', va='center', fontsize=11, weight='bold')
        combinations = sub_df['omics'].unique()
        for x_pos, combo_str in enumerate(combinations):
            active_omics = combo_str.split('_')
            active_y_coords = []
            for y_pos, omic_name in zip(y_positions, all_omics_types):
                if omic_name in active_omics:
                    ax.plot(x_pos, y_pos, 'o', color='black', markersize=8, solid_capstyle='round')
                    active_y_coords.append(y_pos)
                else:
                    ax.plot(x_pos, y_pos, 'o', color='lightgray', markersize=7)
            if len(active_y_coords) > 1:
                ax.plot([x_pos, x_pos], [min(active_y_coords), max(active_y_coords)], 
                        color='black', linewidth=2, zorder=0)
        
        rect_height = 0.08
        rect_y = 1.12
        rect = Rectangle((0, rect_y), 1, rect_height, transform=ax.transAxes,
                         color=cancer_colors[i], clip_on=False)
        ax.add_patch(rect)
        ax.text(0.5, 0.03+rect_y, cancer_type, ha='center', va='center', fontsize=14, color='white', 
                weight='bold', transform=ax.transAxes)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    g.fig.suptitle('Model Mean-AUC Across Cancer Types and Single Omics Combinations', fontsize=20, weight='bold')
    plt.show()
    if _save:
        g.savefig(save_path, dpi=300, bbox_inches='tight')
    

def plot_multi_panel_bar_charts(df, y_name, group_name, mode_colors, cancer_colors, gap_factor=1.2, save_path='./', _save=False):
    """
    Plot a multi-panel bar chart where each panel is a bar chart comparing modes for one cancer.
    
    Parameters:
    - df: DataFrame with 'cancer', 'mode', and 'auc' columns
    - mode_colors: dict mapping modes to colors
    - cancer_colors: dict mapping cancer types to colors for panel backgrounds
    - gap_factor: float (kept for compatibility, but not used)
    - save_path: str path to save the figure
    - _save: bool whether to save the figure
    """
    plt.rcParams['font.family'] = 'Arial'
    
    # Group by cancer and mode, compute mean AUC
    df_grouped = df.groupby(['cancer', group_name])[y_name].mean().reset_index()
    
    # Get unique cancers and modes
    cancers = df_grouped['cancer'].unique()
    modes = df_grouped[group_name].unique()
    num_cancers = len(cancers)
    num_modes = len(modes)
    
    # Determine grid layout: aim for 3 columns
    ncols = 7
    nrows = (num_cancers + ncols - 1) // ncols
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 5 * nrows), sharey=True)
    axs = axs.flatten()
    
    for i, cancer in enumerate(cancers):
        ax = axs[i]
        # Set background color using cancer_colors
        bg_color = mcolors.to_rgba(cancer_colors.get(cancer, 'lightgray'), alpha=0.3)
        ax.set_facecolor(bg_color)
        # ax.set_facecolor(cancer_colors.get(cancer, 'lightgray'))
        
        # Get data for this cancer
        cancer_data = df_grouped[df_grouped['cancer'] == cancer]
        modes_list = cancer_data[group_name].values
        aucs = cancer_data[y_name].values
        
        # Plot bars
        x = np.arange(len(modes_list))
        ax.bar(x, aucs, color=[mode_colors[m] for m in modes_list], edgecolor='black', linewidth=0.5)
        
        # Add AUC text labels
        for j, auc in enumerate(aucs):
            ax.text(x[j], auc + 0.01, f'{auc:.2f}', ha='center', va='bottom', fontsize=10)
        
        # Customize axis
        ax.set_xticks(x)
        ax.set_xticklabels(modes_list, rotation=45, ha='right', fontsize=10)
        ax.set_ylabel('Mean AUROC', fontsize=12)
        ax.set_title(cancer, fontsize=14)
        ax.set_ylim(0, 1.1)  # Adjust based on AUC range
    
    # Hide unused axes if any
    for j in range(num_cancers, len(axs)):
        axs[j].axis('off')
    
    # Add legend for modes
    handles = [plt.Rectangle((0, 0), 1, 1, color=mode_colors[mode]) for mode in modes]
    fig.legend(handles, modes, loc='upper right', title='Modes', fontsize=12, title_fontsize=14)
    
    # Overall title and layout
    fig.suptitle('Mean AUROC by Mode for Each Cancer Type', fontsize=16, y=1.02)
    plt.tight_layout()
    
    if _save:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
   


##################################################################
#                                                                #
#                       以下是针对methods部分的绘图代码                #
#                                                                #  
##################################################################

def methods_organization(methods_df, omics_type):
    methods_df['omics'] = omics_type
    methods_df['methods'] = methods_df['methods'].replace({
        'MOGONET': 'MOGONET',
        'Synomics': 'SynOmics',
        'Multimodal': "Li's method",
        'mogcn': 'MOGCN',
        'RandomForest':'RF',
        'LogisticRegression':'LR',
        'mlp':'MLP'
        })
    # NonGNN_methods = methods_df[~(methods_df['methods'].isin(['MOGONET','SynOmics', 'SynOmics_BRCA',"Li's method", 'MOGCN']))]
    # NonGNN_methods = NonGNN_methods[~(NonGNN_methods['cancer'].isin(['MOGONET_BRCA', 'SynOmics_BRCA', 'TCGA_BRCA', 'BRCA', 'COAD', 'HNSCC', 'CCRCC', 'PDAC', 'LUAD', 'LSCC']))]
    NonGNN_methods_selected = selected_data(methods_df)
    return NonGNN_methods_selected


def methods_comparison_bubble(metrics_cols, row_order, colormaps, df, _save, save_path ='./R4D_CPTAC_data_m_mi-pro.pdf'):    
    # --- 1. 设置绘图布局 ---
    # 创建一个图和两个子图 (axes)，左侧 (ax_bubble) 宽，右侧 (ax_bar) 窄
    fig, (ax_bubble, ax_bar) = plt.subplots(
        1, 2, 
        figsize=(13, 7.5),
        # figsize=(7,3.5)
        # gridspec_kw={'width_ratios': [4, 1.6], 'wspace': 0.05}
        gridspec_kw={'width_ratios': [0.5, 0.3], 'wspace': 0.05}
    )
    fig.suptitle(' methods comparison', fontsize=18, fontweight='bold', y=1.02)
    
    # --- 2. 绘制左侧的气泡热图 ---
    # 定义每个指标（列）的色系
    df = df.reindex(row_order)
    metrics_data = df[metrics_cols]
    # df['mean'] = df.mean(1)
    
    n_rows, n_cols = metrics_data.shape
    global_min = metrics_data.min().min()
    global_max = metrics_data.max().max()
    min_markersize = 25
    max_markersize = 40
    ax_bubble.set_xlim(-0.5, n_cols - 0.5)
    ax_bubble.set_ylim(n_rows - 0.5, -0.5) # y轴反转，使'Seurat'在顶部
    ax_bubble.set_xticks(range(n_cols))
    ax_bubble.set_xticklabels(
        metrics_data.columns, 
        fontsize=12, 
        fontweight='bold', 
        rotation=45,          # 旋转45度
        ha='left',            # 设置水平对齐方式
        rotation_mode='anchor' # 以锚点（刻度）为中心旋转
    )
    ax_bubble.set_yticks(range(n_rows))
    ax_bubble.set_yticklabels(metrics_data.index, fontsize=12, fontweight='bold')

    ax_bubble.xaxis.set_ticks_position('top')    # 将X轴标签（指标名称）移到顶部
    ax_bubble.xaxis.set_label_position('top')
    ax_bubble.spines['top'].set_visible(False)
    ax_bubble.spines['right'].set_visible(False)
    ax_bubble.spines['bottom'].set_visible(False)
    ax_bubble.spines['left'].set_visible(False)
    ax_bubble.tick_params(axis='both', which='both', length=0)
    ax_bubble.grid(axis='y', linestyle='dotted', color='gray')


    for i in range(n_rows):
        for j in range(n_cols):
            metric = metrics_data.columns[j]
            value = metrics_data.iloc[i, j]
        
            # 1. 颜色：根据列的min/max进行归一化，以确定颜色深浅
            cmap = colormaps[j] 
            norm_color = mcolors.Normalize(vmin=metrics_data[metric].min(), 
                                        vmax=metrics_data[metric].max())
            color = cmap(norm_color(value))
            
            # 2. 文本颜色：自动判断背景亮度
            # 计算颜色的感知亮度
            r, g, b = to_rgb(color)
            luminance = (0.299 * r + 0.587 * g + 0.114 * b)
            text_color = 'black' if luminance > 0.5 else 'white'
            
            # 3. 绘制圆圈
            norm_size_val = (value - global_min) / (global_max - global_min)
            current_markersize = min_markersize + norm_size_val * (max_markersize - min_markersize)-10
            # ax_bubble.plot(j, i, 'o', markersize=28, color=color)
            ax_bubble.plot(j, i, 'o', markersize=current_markersize, color=color)
            
            # 4. 绘制文本
            ax_bubble.text(j, i, f"{value:.4f}", 
                        ha='center', va='center', 
                        color=text_color, fontsize=7)

    # --- 3. 绘制右侧的均值条形图 ---
    y_pos = range(n_rows)
    means = metrics_data.mean(axis=1)
    stds  = metrics_data.std(axis=1)
    # print(metrics_data)
    bar_color = '#0072B2'
    error_color = 'black'
    capsize     = 5
    capthick    = 2
    elinewidth  = 1.8

    # 绘制水平条形图
    ax_bar.barh(y_pos, means, color=bar_color, align='center')
    ax_bar.grid(axis='y', linestyle='dotted', color='gray')
    for i, v in enumerate(means):
        ax_bar.text(v - 0.02, i, f"{v:.4f}", 
                    color='white', ha='right', va='center', 
                    fontsize=8, fontweight='bold')
    ax_bar.set_ylim(n_rows - 0.5, -0.5)
    ax_bar.set_yticks([])
    ax_bar.set_yticklabels([])
    ax_bar.xaxis.set_ticks_position('top')
    ax_bar.xaxis.set_label_position('top')
    ax_bar.set_title('Mean', fontsize=12, fontweight='bold', pad=20)

    ax_bar.set_xlim(means.min() - 0.1, means.max() + 0.01)
    ax_bar.tick_params(axis='x', length=0)

    ax_bar.spines['top'].set_visible(False)
    ax_bar.spines['right'].set_visible(False)
    ax_bar.spines['bottom'].set_visible(False)
    ax_bar.spines['left'].set_visible(True)
    ax_bar.spines['left'].set_linewidth(2)
    ax_bar.spines['left'].set_color('black')

    plt.tight_layout(rect=[0, 0.03, 1, 0.92])
    if _save:
        plt.savefig(save_path)
    plt.show()



def methods_comparison_bubble_v3(metrics_cols, row_order, colormaps, df, _save, save_path='./comparison_v3.pdf'):
    # --- 0. 准备数据 ---
    df = df.reindex(row_order)
    metrics_data = df[metrics_cols]
    n_rows, n_cols = metrics_data.shape
    
    # 确定分割点：前3列和剩余列
    split_col_index = 3
    metrics_left_df = metrics_data.iloc[:, :split_col_index]
    metrics_right_df = metrics_data.iloc[:, split_col_index:] 
    n_cols_left = metrics_left_df.shape[1]
    n_cols_right = metrics_right_df.shape[1]

    global_min = metrics_data.min().min()
    global_max = metrics_data.max().max()
    min_markersize = 25
    max_markersize = 40
    
    # **【核心修改 A】计算局部均值和标准差**
    # 1. 左侧条形图 (对应前 3 列)
    means_l = metrics_left_df.mean(axis=1)
    stds_l = metrics_left_df.std(axis=1)
    
    # 2. 右侧条形图 (对应剩余列)
    means_r = metrics_right_df.mean(axis=1)
    stds_r = metrics_right_df.std(axis=1)
    
    # 调整 X 轴范围以容纳误差棒文本 (使用左侧和右侧的最大/最小值)
    max_x_l = (means_l + stds_l).max() + 0.05
    min_x_l = (means_l - stds_l).min() - 0.05
    
    max_x_r = (means_r + stds_r).max() + 0.05
    min_x_r = (means_r - stds_r).min() - 0.05
    
    # --- 1. 设置绘图布局 ---
    fig, (ax_bar_l, ax_bubble_l, ax_names, ax_bubble_r, ax_bar_r) = plt.subplots(
        1, 5, 
        figsize=(18, 7.5),
        gridspec_kw={
            'width_ratios': [1.3, 4 * (n_cols_left / n_cols), 0.7, 4 * (n_cols_right / n_cols), 1.3],
            'wspace': 0.0 
        }
    )
    fig.suptitle('Methods Comparison', fontsize=18, fontweight='bold', y=0.9)
    row_pos = np.arange(n_rows)

    # --- 2. 绘制左侧平均结果柱状图 (ax_bar_l) - 右对齐 ---
    bar_color = '#0072B2'
    # **【核心修改 B】使用 means_l 绘制**
    ax_bar_l.barh(row_pos, means_l, color=bar_color, align='center')
    ax_bar_l.set_ylim(n_rows - 0.5, -0.5)

    for i, (m, s) in enumerate(zip(means_l, stds_l)):
        text = f"{m:.4f}" #±{s:.4f}
        ax_bar_l.text(min_x_l + 0.2, i, text, 
                      color='black', ha='left', va='center', 
                      fontsize=10, fontweight='bold')
                      
    # 格式化
    ax_bar_l.set_yticks([]) 
    ax_bar_l.xaxis.set_ticks_position('top')
    ax_bar_l.xaxis.set_label_position('top')
    # **【核心修改 D】修改标题以反映数据范围**
    ax_bar_l.set_title(f"Mean $\pm$ Std\n({metrics_left_df.columns[0]} to {metrics_left_df.columns[-1]})", fontsize=10, fontweight='bold', pad=20)
    ax_bar_l.set_xlim(min_x_l, means_l.max() + 0.01) 
    ax_bar_l.tick_params(axis='x', length=0)
    
    ax_bar_l.spines['top'].set_visible(False)
    ax_bar_l.spines['right'].set_visible(True) 
    ax_bar_l.spines['right'].set_linewidth(2)
    ax_bar_l.spines['right'].set_color('black')
    ax_bar_l.spines['bottom'].set_visible(False)
    ax_bar_l.spines['left'].set_visible(False)
    ax_bar_l.invert_xaxis() 

    # --- 3. 绘制左侧气泡热图 (ax_bubble_l) ---
    ax_bubble_l.set_xlim(-0.5, n_cols_left - 0.5)
    ax_bubble_l.set_ylim(n_rows - 0.5, -0.5) 
    
    # 设置 X 轴标签（数据集名称）
    ax_bubble_l.set_xticks(range(n_cols_left))
    ax_bubble_l.set_xticklabels(
        metrics_left_df.columns, fontsize=12, fontweight='bold', 
        rotation=45, ha='left', rotation_mode='anchor'
    )
    
    # Y 轴标签完全隐藏
    ax_bubble_l.set_yticks([]) 
    ax_bubble_l.tick_params(axis='y', length=0, labelright=False, labelleft=False) 
    
    # 格式化
    ax_bubble_l.xaxis.set_ticks_position('top')
    ax_bubble_l.xaxis.set_label_position('top')
    for spine in ax_bubble_l.spines.values(): spine.set_visible(False)
    
    ax_bubble_l.tick_params(axis='x', which='both', length=0) 
    ax_bubble_l.grid(axis='y', linestyle='dotted', color='gray')
    # ax_bubble_l.set_title('Dataset Results', fontsize=12, fontweight='bold', pad=20)

    # 绘制左侧气泡和文本（绘图逻辑不变）
    for i in range(n_rows):
        for j in range(n_cols_left):
            metric = metrics_left_df.columns[j]
            value = metrics_left_df.iloc[i, j]
            
            # 颜色和大小计算
            cmap = colormaps[j] 
            norm_color = mcolors.Normalize(vmin=metrics_data[metric].min(), 
                                        vmax=metrics_data[metric].max())
            color = cmap(norm_color(value))
            r, g, b = to_rgb(color)
            luminance = (0.299 * r + 0.587 * g + 0.114 * b)
            text_color = 'black' if luminance > 0.5 else 'white'
            norm_size_val = (value - global_min) / (global_max - global_min)
            current_markersize = min_markersize + norm_size_val * (max_markersize - min_markersize) - 10
            
            # 绘制
            ax_bubble_l.plot(j, i, 'o', markersize=current_markersize, color=color)
            ax_bubble_l.text(j, i, f"{value:.4f}", 
                        ha='center', va='center', 
                        color=text_color, fontsize=7)


    # --- 3.5. 绘制方法名子图 (ax_names) ---
    ax_names.set_xlim(0, 1)
    ax_names.set_ylim(n_rows - 0.5, -0.5)

    method_names = metrics_left_df.index
    for i, name in enumerate(method_names):
        ax_names.text(
            0.5, i, name, 
            ha='center', 
            va='center', 
            fontsize=11, 
            fontweight='bold'
        ) 

    ax_names.set_yticks([]) 
    ax_names.set_xticks([])
    ax_names.set_title('', pad=20)
    
    # 边框设置
    for spine in ax_names.spines.values(): spine.set_visible(False)
    ax_names.spines['left'].set_visible(True)
    ax_names.spines['left'].set_linewidth(2)
    ax_names.spines['left'].set_color('black')
    ax_names.spines['right'].set_visible(True)
    ax_names.spines['right'].set_linewidth(2)
    ax_names.spines['right'].set_color('black')


    # --- 4. 绘制右侧气泡热图 (ax_bubble_r) ---
    ax_bubble_r.set_xlim(-0.5, n_cols_right - 0.5)
    ax_bubble_r.set_ylim(n_rows - 0.5, -0.5)
    
    # X 轴标签（数据集名称）
    ax_bubble_r.set_xticks(range(n_cols_right))
    ax_bubble_r.set_xticklabels(
        metrics_right_df.columns, fontsize=12, fontweight='bold', 
        rotation=45, ha='left', rotation_mode='anchor'
    )
    # 确保 Y 轴标签被隐藏
    ax_bubble_r.set_yticks([]) 
    
    # 格式化
    ax_bubble_r.xaxis.set_ticks_position('top')
    ax_bubble_r.xaxis.set_label_position('top')
    for spine in ax_bubble_r.spines.values(): spine.set_visible(False)
    ax_bubble_r.tick_params(axis='both', which='both', length=0)
    ax_bubble_r.grid(axis='y', linestyle='dotted', color='gray')
    # ax_bubble_r.set_title('Dataset Results', fontsize=12, fontweight='bold', pad=20)


    # 绘制右侧气泡和文本（绘图逻辑不变）
    for i in range(n_rows):
        for j in range(n_cols_right):
            j_total = split_col_index + j 
            metric = metrics_right_df.columns[j]
            value = metrics_right_df.iloc[i, j]
            
            # 颜色和大小计算
            cmap = colormaps[j_total] 
            norm_color = mcolors.Normalize(vmin=metrics_data[metric].min(), 
                                        vmax=metrics_data[metric].max())
            color = cmap(norm_color(value))
            r, g, b = to_rgb(color)
            luminance = (0.299 * r + 0.587 * g + 0.114 * b)
            text_color = 'black' if luminance > 0.5 else 'white'
            norm_size_val = (value - global_min) / (global_max - global_min)
            current_markersize = min_markersize + norm_size_val * (max_markersize - min_markersize) - 10
            
            # 绘制
            ax_bubble_r.plot(j, i, 'o', markersize=current_markersize, color=color)
            ax_bubble_r.text(j, i, f"{value:.4f}", 
                        ha='center', va='center', 
                        color=text_color, fontsize=7)

    # --- 5. 绘制右侧平均结果柱状图 (ax_bar_r) - 左对齐 ---
    bar_color = '#0072B2'
    # **【核心修改 E】使用 means_r 绘制**
    ax_bar_r.barh(row_pos, means_r, color=bar_color, align='center')
    ax_bar_r.set_ylim(n_rows - 0.5, -0.5)

    for i, (m, s) in enumerate(zip(means_r, stds_r)):
        text = f"{m:.4f}" #±{s:.4f}
        ax_bar_r.text(max_x_r - 0.2, i, text, 
                      color='black', ha='right', va='center', 
                      fontsize=10, fontweight='bold')
                      
    # 格式化
    ax_bar_r.set_yticks([])
    ax_bar_r.xaxis.set_ticks_position('top')
    ax_bar_r.xaxis.set_label_position('top')
    # **【核心修改 G】修改标题以反映数据范围**
    ax_bar_r.set_title(f"Mean $\pm$ Std\n({metrics_right_df.columns[0]} to {metrics_right_df.columns[-1]})", fontsize=10, fontweight='bold', pad=20)
    ax_bar_r.set_xlim(means_r.min() - 0.01, max_x_r) 
    ax_bar_r.tick_params(axis='x', length=0)
    
    # ... (边框设置保持不变) ...
    ax_bar_r.spines['top'].set_visible(False)
    ax_bar_r.spines['right'].set_visible(False)
    ax_bar_r.spines['bottom'].set_visible(False)
    ax_bar_r.spines['left'].set_visible(True) 
    ax_bar_r.spines['left'].set_linewidth(2)
    ax_bar_r.spines['left'].set_color('black')


    # --- 6. 调整布局和保存 ---
    plt.tight_layout(rect=[0, 0.03, 1, 0.92])
    if _save:
        plt.savefig(save_path)
    plt.show()
    
  
def plot_auc_heatmap(
    df,
    group_col='methods',
    metric='AUC',
    figsize=(12, 10),
    cmap='Blues',
    title='Heatmap of Average AUC by Method and Cancer Type',
    _save=False,
    order=None,
    col_order=None
):
    NATURE_STYLE = {
        'font.sans-serif': 'Arial',
        'axes.labelsize': 10,
        'axes.titlesize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'axes.linewidth': 10,
        'lines.linewidth': 10,
        'pdf.fonttype': 42,
        'figure.dpi': 300,
        'figure.figsize': figsize
    }
    
    # 计算均值透视表
    pivot = (
        df
        .groupby(['Cancer Type', group_col])[metric]
        .mean()
        .unstack()
    )
    hm = pivot.T.reindex(order)

    fig = plt.figure(figsize=figsize)
    ax = sns.heatmap(
        hm,
        cmap=cmap,              
        annot=True, fmt=".4f", 
        linewidths=0.8,        
        linecolor='white',
        square=False,           
        annot_kws={           
            'size': 12,        
            'weight': 'normal' 
        },
        cbar_kws={
            'shrink': 0.8,     
            'pad': 0.02,        
            'label': 'Mean AUC'
        }
    )
    ax.set_xlabel('Cancer Type', fontsize=10)
    ax.set_ylabel('Method', fontsize=10)
    ax.set_title(title, fontsize=14, pad=10)


    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=8)

    plt.tight_layout()
    # plt.savefig('./logs/The Heatmap of Average ROC AUC by Method and Cancer Type.svg')
    plt.show()
    if _save:
        fig.savefig(f'./{title}.pdf', dpi=300, bbox_inches='tight')
    
    df_results = (
        df
        .groupby(['Cancer Type', group_col])[metric]
        .agg(format_mean_std)
        .unstack()
    )
    hm_df_results = df_results.T.reindex(order)
    return hm_df_results



def format_mean_std(x):
    return f"{x.mean():.4f} ± {x.std():.4f}"

def plot_auc_heatmap_advance(
    df,
    group_col='methods',
    metric='AUC',
    figsize=(18, 10),  
    cmap='Purples',  
    title='Heatmap of Average AUC by Method and Cancer Type',
    _save=False,
    save_path='./',
    order=None,
    col_order=None
):
    NATURE_STYLE = {
        'font.sans-serif': 'Arial',
        'axes.labelsize': 12,  
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'axes.linewidth': 1.0,
        'lines.linewidth': 1.0,
        'pdf.fonttype': 42,
        'figure.dpi': 300,
        'figure.figsize': figsize
    }
    
    plt.rcParams.update(NATURE_STYLE)
    
    pivot = (
        df
        .groupby(['Cancer Type', group_col])[metric]
        .mean()
        .unstack()
    )
    hm = pivot.T.reindex(order)
    if col_order is not None:
        hm = hm[col_order]
    
    # 计算每个方法的整体均值，不排序，以匹配热图Y轴顺序
    method_means = hm.mean(axis=1)
    
    # 创建子图：左侧热图，中间bar，最右侧colorbar
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(1, 3, width_ratios=[5, 1, 0.2], wspace=0.15)  # 增加wspace以调整间距
    ax_heatmap = fig.add_subplot(gs[0])
    ax_bar = fig.add_subplot(gs[1], sharey=ax_heatmap)
    ax_cbar = fig.add_subplot(gs[2])
    
    # 绘制热图，并指定cbar_ax为最右侧轴
    sns.heatmap(
        hm,
        ax=ax_heatmap,
        cmap=cmap,
        annot=True, fmt=".4f",
        linewidths=0.8,
        linecolor='white',
        square=False,
        annot_kws={'size': 12, 'weight': 'normal'},
        cbar_ax=ax_cbar,  # 指定colorbar轴
        cbar_kws={
            'shrink': 0.8,
            'label': 'Mean AUC',
            'orientation': 'vertical'
        }
    )
    ax_heatmap.set_xlabel('Cancer Type', fontsize=12)
    ax_heatmap.set_ylabel('Method', fontsize=12)
    ax_heatmap.set_title(title, fontsize=14, pad=15)  # 增大pad
    ax_heatmap.set_xticklabels(ax_heatmap.get_xticklabels(), rotation=45, ha='right', fontsize=10)
    
    y_ticks = np.arange(len(hm.index)) + 0.5
    ax_heatmap.set_yticks(y_ticks)
    ax_heatmap.set_yticklabels(hm.index, fontsize=10, rotation=0)
    
    ax_heatmap.grid(which='major', linestyle='-', linewidth=0.5, color='gray', alpha=0.3)
    
    for spine in ax_heatmap.spines.values():
        spine.set_visible(False)
    
    # 绘制中间bar：每个方法的均值，匹配热图Y轴顺序
    norm = Normalize(vmin=hm.min().min(), vmax=hm.max().max())  # 归一化以匹配热图颜色
    colors = plt.cm.get_cmap(cmap)(norm(method_means.values))  # bar颜色渐变
    y_positions = np.arange(len(method_means))  # 与热图行数匹配
    ax_bar.barh(y_positions + 0.5, method_means.values, color=colors, edgecolor='black', height=0.8)  # 调整Y位置到中间
    ax_bar.set_xlabel('Overall Mean AUC', fontsize=12)
    ax_bar.set_xlim(0, method_means.max() * 1.1)  # 动态扩展以容纳标签


    for i, v in enumerate(method_means.values):
        ax_bar.text(v + 0.01, i + 0.5, f"{v:.4f}", va='center', fontsize=10)  # 调整标签Y位置
    for spine in ['top', 'right', 'left', 'bottom']:
        ax_bar.spines[spine].set_visible(False)
    
    # colorbar轴美化：移除不必要边框
    for spine in ax_cbar.spines.values():
        spine.set_visible(False)
    
    # 调整整体布局以确保左侧Y轴标签可见
    fig.subplots_adjust(left=0.15)  # 增加左侧边距以容纳Y轴标签
    
    plt.tight_layout()
    plt.show()
    
    if _save:
        fig.savefig(f'{save_path}.pdf', dpi=300, bbox_inches='tight')
    
    df_results = (
        df
        .groupby(['Cancer Type', group_col])[metric]
        .agg(format_mean_std)
        .unstack()
    )
    hm_df_results = df_results.T.reindex(order)
    return hm_df_results
  
  

def plot_delta_auc_visualization(
    delta_auc,  # DataFrame: rows=methods, columns=cancer types
    positive_ratio,  # Series: index=methods, values=positive ratios
    figsize=None,
    cmap='coolwarm',  # To highlight positive (red) and negative (blue) changes
    title='Delta AUC Heatmap: Performance Improvement After Adding Proteomics',
    _save=False,
    save_path='./',
    cancer_order=None  # Optional: list of cancer types to order columns
):
    NATURE_STYLE = {
        'font.sans-serif': 'Arial',
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'axes.linewidth': 1.0,
        'lines.linewidth': 1.0,
        'pdf.fonttype': 42,
        'figure.dpi': 300,
    }
    
    plt.rcParams.update(NATURE_STYLE)
    
    # Ensure delta_auc is a DataFrame
    if not isinstance(delta_auc, pd.DataFrame):
        raise ValueError("delta_auc must be a pandas DataFrame with methods as rows and cancer types as columns.")
    
    # Sort rows by average delta_auc descending to highlight top improvements
    avg_delta = delta_auc.mean(axis=1).sort_values(ascending=False)
    delta_auc_sorted = delta_auc.loc[avg_delta.index]
    positive_ratio_sorted = positive_ratio.loc[avg_delta.index]
    
    if cancer_order is not None:
        delta_auc_sorted = delta_auc_sorted[cancer_order]
    
    num_methods = len(delta_auc_sorted.index)
    if figsize is None:
        figsize = (18, max(10, num_methods * 0.5))
    NATURE_STYLE['figure.figsize'] = figsize
    
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(1, 4, width_ratios=[5, 1, 1, 0.2], wspace=0.15)
    ax_heatmap = fig.add_subplot(gs[0])
    ax_avg_bar = fig.add_subplot(gs[1], sharey=ax_heatmap)
    ax_ratio_bar = fig.add_subplot(gs[2], sharey=ax_heatmap)
    ax_cbar = fig.add_subplot(gs[3])
    
    # Draw heatmap for delta_auc
    sns.heatmap(
        delta_auc_sorted,
        ax=ax_heatmap,
        cmap=cmap,
        annot=True, fmt=".4f",
        linewidths=0.8,
        linecolor='white',
        square=False,
        annot_kws={'size': 12, 'weight': 'normal'},
        cbar_ax=ax_cbar,
        cbar_kws={
            'shrink': 0.8,
            'label': 'ΔAUC',
            'orientation': 'vertical'
        },
        center=0  # Center colormap at 0 to highlight positive/negative
    )
    ax_heatmap.set_xlabel('Cancer Type', fontsize=12)
    ax_heatmap.set_ylabel('Method', fontsize=12)
    ax_heatmap.set_title(title, fontsize=14, pad=15)
    ax_heatmap.set_xticklabels(ax_heatmap.get_xticklabels(), rotation=45, ha='right', fontsize=10)
    
    # Adjust Y-ticks to row centers and display method names
    y_ticks = np.arange(len(delta_auc_sorted.index)) + 0.5
    ax_heatmap.set_yticks(y_ticks)
    ax_heatmap.set_yticklabels(delta_auc_sorted.index, fontsize=10, rotation=0)
    
    # Add light gridlines
    ax_heatmap.grid(which='major', linestyle='-', linewidth=0.5, color='gray', alpha=0.3)
    
    # Remove unnecessary spines
    for spine in ax_heatmap.spines.values():
        spine.set_visible(False)
    
    # Draw middle bar: average delta_auc
    norm = Normalize(vmin=delta_auc_sorted.min().min(), vmax=delta_auc_sorted.max().max())
    colors_avg = plt.cm.get_cmap(cmap)(norm(avg_delta.values))
    y_positions = np.arange(len(avg_delta))
    ax_avg_bar.barh(y_positions + 0.5, avg_delta.values, color=colors_avg, edgecolor='black', height=0.8)
    ax_avg_bar.set_xlabel('Avg ΔAUC', fontsize=12)
    ax_avg_bar.set_xlim(min(0, avg_delta.min() * 1.1), avg_delta.max() * 1.1)  # Handle negative values
    ax_avg_bar.invert_yaxis()
    # ax_avg_bar.set_yticks([])  # 隐藏avg_bar的Y轴标签，只在热图显示
    # ax_avg_bar.set_yticklabels([])
    
    # Add value labels for avg bar
    for i, v in enumerate(avg_delta.values):
        ax_avg_bar.text(v + 0.01 if v >= 0 else v - 0.05, i + 0.5, f"{v:.4f}", va='center', fontsize=10, ha='left' if v >= 0 else 'right')
    
    for spine in ax_avg_bar.spines.values():
        spine.set_visible(False)
    
    # Draw right bar: positive_ratio
    colors_ratio = plt.cm.get_cmap('Greens')(Normalize(0, 1)(positive_ratio_sorted.values))  # Green gradient for ratios (0-1)
    ax_ratio_bar.barh(y_positions + 0.5, positive_ratio_sorted.values, color=colors_ratio, edgecolor='black', height=0.8)
    ax_ratio_bar.set_xlabel('Positive Ratio', fontsize=12)
    ax_ratio_bar.set_xlim(0, 1.1)
    ax_ratio_bar.invert_yaxis()
    
    # Add value labels for ratio bar
    for i, v in enumerate(positive_ratio_sorted.values):
        ax_ratio_bar.text(v + 0.01, i + 0.5, f"{v:.2f}", va='center', fontsize=10)
    
    for spine in ax_ratio_bar.spines.values():
        spine.set_visible(False)
    
    # Colorbar beautification
    for spine in ax_cbar.spines.values():
        spine.set_visible(False)
    
    # Adjust layout to ensure left Y-labels are visible, 增大左侧边距
    fig.subplots_adjust(left=0.2)  # 增加到0.2以容纳长方法名
    
    plt.tight_layout()
    plt.show()
    
    if _save:
        fig.savefig(f'{save_path}{title.replace(" ", "_")}.pdf', dpi=300, bbox_inches='tight')


def plot_delta_auc_bars(
    delta_auc,  # DataFrame: rows=methods, columns=cancer types
    positive_ratio,  # Series: index=methods, values=positive ratios
    figsize=(10, 12),  # 调整为垂直布局，更高一些
    cmap_delta='coolwarm',  # For delta_auc (positive red, negative blue)
    cmap_ratio='Greens',    # For positive_ratio (green gradient)
    title='Performance Improvement After Adding Proteomics',
    _save=False,
    save_path='./',
    order=None  # 新增: 自定义Y轴顺序（方法列表）
):
    NATURE_STYLE = {
        'font.sans-serif': 'Arial',
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'axes.linewidth': 1.0,
        'lines.linewidth': 1.0,
        'pdf.fonttype': 42,
        'figure.dpi': 300,
        'figure.figsize': figsize
    }
    
    plt.rcParams.update(NATURE_STYLE)
    
    # Ensure delta_auc is a DataFrame
    if not isinstance(delta_auc, pd.DataFrame):
        raise ValueError("delta_auc must be a pandas DataFrame with methods as rows and cancer types as columns.")
    
    # 计算平均delta_auc
    avg_delta = delta_auc.mean(axis=1)
    
    # 根据order调整顺序；如果None，按avg_delta降序排序
    if order is not None:
        avg_delta = avg_delta.reindex(order).dropna()
        positive_ratio_sorted = positive_ratio.reindex(order).dropna()
    else:
        avg_delta = avg_delta.sort_values(ascending=False)
        positive_ratio_sorted = positive_ratio.loc[avg_delta.index]
    
    # Create figure with two subplots, arranged vertically
    fig, (ax_avg_bar, ax_ratio_bar) = plt.subplots(2, 1, figsize=figsize, sharex=False, gridspec_kw={'height_ratios': [1, 1], 'hspace': 0.3})
    fig.suptitle(title, fontsize=14, y=1.02)
    
    # Upper bar: average delta_auc (horizontal barh)
    norm_delta = Normalize(vmin=avg_delta.min(), vmax=avg_delta.max())
    colors_delta = plt.cm.get_cmap(cmap_delta)(norm_delta(avg_delta.values))
    y_positions = np.arange(len(avg_delta))
    ax_avg_bar.barh(y_positions, avg_delta.values, color=colors_delta, edgecolor='black', height=0.8)
    ax_avg_bar.set_yticks(y_positions)
    ax_avg_bar.set_yticklabels(avg_delta.index, fontsize=10, rotation=0)
    ax_avg_bar.set_xlabel('Average ΔAUC', fontsize=12)
    ax_avg_bar.set_ylabel('Method', fontsize=12)
    ax_avg_bar.set_xlim(min(0, avg_delta.min() * 1.1), avg_delta.max() * 1.1)  # Handle negative values
    ax_avg_bar.invert_yaxis()  # Top to bottom descending
    
    # Add value labels for avg bar, 调整位置偏移以避免重叠
    for i, v in enumerate(avg_delta.values):
        # offset = 0.02 if v >= 0 else -0.08  # 增大负值偏移以确保在柱子外
        offset = 0.0005 if v >= 0 else -0.0005  # 增大负值偏移以确保在柱子外
        ha_align = 'left' if v >= 0 else 'right'
        ax_avg_bar.text(v + offset if v >= 0 else v + offset, i, f"{v:.4f}", va='center', fontsize=10, ha=ha_align)
    
    # Remove unnecessary spines
    for spine in ax_avg_bar.spines.values():
        spine.set_visible(False)
    
    # Lower bar: positive_ratio (horizontal barh)
    norm_ratio = Normalize(vmin=0, vmax=1)
    colors_ratio = plt.cm.get_cmap(cmap_ratio)(norm_ratio(positive_ratio_sorted.values))
    ax_ratio_bar.barh(y_positions, positive_ratio_sorted.values, color=colors_ratio, edgecolor='black', height=0.8)
    ax_ratio_bar.set_yticks(y_positions)
    ax_ratio_bar.set_yticklabels(avg_delta.index, fontsize=10, rotation=0)  # Reuse same labels
    ax_ratio_bar.set_xlabel('Positive ΔAUC Ratio', fontsize=12)
    ax_ratio_bar.set_ylabel('Method', fontsize=12)
    ax_ratio_bar.set_xlim(0, 1.1)
    ax_ratio_bar.invert_yaxis()
    
    # Add value labels for ratio bar, 调整位置偏移
    for i, v in enumerate(positive_ratio_sorted.values):
        ax_ratio_bar.text(v + 0.02, i, f"{v:.2f}", va='center', fontsize=10)  # 增大偏移以避免重叠
    
    for spine in ax_ratio_bar.spines.values():
        spine.set_visible(False)
    fig.subplots_adjust(left=0.25)  # Increase left margin for method names
    plt.tight_layout()
    plt.show()
    
    if _save:
        fig.savefig(f'{save_path}.pdf', dpi=300, bbox_inches='tight')
        
        
        
def plot_method_comparison_scatter(df, summary_df, method_col='methods',
                                    f1_col='f1', auc_col='roc_auc',
                                    run_col='run', color_map=None, marker_map=None,
                                    title='Method Comparison: F1 vs AUC', figsize=(8, 6),
                                    _save=False):
    """绘制方法散点 + 平均气泡图"""
    fig = plt.figure(figsize=figsize)

    methods = summary_df[method_col].unique()
    runs = df[run_col].unique()

    # 1. 每个 run 的散点（透明）
    for method in methods:
        for run in runs:
            subset = df[(df[method_col] == method) & (df[run_col] == run)]
            if subset.empty:
                continue
            f1_mean = subset[f1_col].mean()
            auc_mean = subset[auc_col].mean()
            plt.scatter(
                f1_mean, auc_mean,
                s=30,
                color=color_map[method],
                alpha=0.3,
                marker=marker_map[method],
                edgecolor='none'
            )

    # 2. 方法整体平均值气泡（大 + 有边框）
    for _, row in summary_df.iterrows():
        m = row[method_col]
        f1_avg = row['f1_mean']
        auc_avg = row['auc_mean']
        # size = 200 * (auc_avg+f1_avg)  # 气泡大小与 AUC 平均值相关
        plt.scatter(
            f1_avg, auc_avg,
            # s=size,
            color=color_map[m],
            marker=marker_map[m],
            label=m,
            edgecolor='k',
            linewidth=0.8,
            zorder=3
        )
        plt.text(
            f1_avg + 0.001, auc_avg + 0.001,
            m, fontsize=10, ha='left', va='bottom', zorder=4
        )
    plt.xlabel('Mean F1', fontsize=12)
    plt.ylabel('Mean AUC', fontsize=12)
    plt.title(title, fontsize=14)
    # plt.legend(title='Method', bbox_to_anchor=(0, 1), loc='upper left', borderaxespad=0.2)
    plt.legend(title='Method', bbox_to_anchor=(1, 0), loc='lower right', borderaxespad=0.2)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()
    if _save:
        fig.savefig(f'./{title}.pdf', dpi=300, bbox_inches='tight')



def plot_CD(df, _save=False):
    NATURE_STYLE = {
        'font.sans-serif': 'Arial',
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'axes.linewidth': 1.0,
        'lines.linewidth': 1.0,
        'pdf.fonttype': 42,
        'figure.dpi': 300
    }
    
    plt.rcParams.update(NATURE_STYLE)
    
    df_run = df.groupby(['Cancer Type', 'methods'])[['roc_auc']].mean().reset_index()
    df_run['rank'] = df_run.groupby('Cancer Type')['roc_auc'].rank(ascending=False, method='average')
    mean_rank = df_run.groupby('methods')['rank'].mean().reset_index(name='mean_rank')
    mean_rank = mean_rank.sort_values('mean_rank') 
    
    # (1) plot bar figure
    fig1 = plt.figure(figsize=(8, len(mean_rank) * 0.5))
    ax1 = fig1.add_subplot(111)
    norm = plt.Normalize(mean_rank['mean_rank'].min(), mean_rank['mean_rank'].max())
    colors = plt.cm.viridis(norm(mean_rank['mean_rank'].values))  
    sns.barplot(
        x='mean_rank', y='methods',
        data=mean_rank,
        palette=colors,
        dodge=False,
        ax=ax1
    )

    for i, row in enumerate(mean_rank.iterrows()):
        _, row_data = row
        ax1.text(
            row_data['mean_rank'] + 0.05, 
            i,
            f"{row_data['mean_rank']:.2f}",
            va='center',
            fontsize=10
        )
    
    ax1.set_xlabel('Average Rank (lower is better)', fontsize=12)
    ax1.set_ylabel('Method', fontsize=12)
    ax1.set_title('Overall Model Ranking Across Datasets', fontsize=14, pad=15)
    ax1.set_xlim(0.5, mean_rank['mean_rank'].max() + 0.5)
    
    ax1.grid(which='major', axis='x', linestyle='--', linewidth=0.5, color='gray', alpha=0.5)
    for spine in ax1.spines.values():
        spine.set_visible(False)
    
    plt.tight_layout()
    plt.show()
    if _save:
        fig1.savefig('./fig4E_CD_rank.pdf', dpi=300, bbox_inches='tight')
    
    # (2) plot siginficant heatmap
    fig2 = plt.figure(figsize=(8, len(mean_rank) * 0.5))
    ax2 = fig2.add_subplot(111)
    test_results = sp.posthoc_conover_friedman(
        df_run,
        melted=True,
        block_col='Cancer Type',
        group_col='methods',
        y_col='roc_auc',
    )
    
    sns.heatmap(
        test_results,
        ax=ax2,
        cmap='RdYlGn', 
        annot=True,
        fmt=".3f",
        linewidths=0.5,
        linecolor='white',
        cbar_kws={'shrink': 0.8, 'pad': 0.05, 'label': 'p-value'},
        annot_kws={'size': 10}
    )
    
    ax2.set_title('Conover-Friedman Post-Hoc Test (p-values)', fontsize=14, pad=15)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
    ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0)

    for spine in ax2.spines.values():
        spine.set_visible(False)
    
    plt.tight_layout()
    plt.show()
    if _save:
        fig2.savefig('./fig4E_CD_sig.pdf', dpi=300, bbox_inches='tight')
    
    # (3) plot CD figure：adjust and refine
    ranks = pd.Series(mean_rank['mean_rank'])
    ranks.index = mean_rank['methods'].tolist()
    
    fig3 = plt.figure(figsize=(12, 3), dpi=300)  
    ax3 = fig3.add_subplot(111)
    ax3.set_title('Critical Difference Diagram of Average Score Ranks', fontsize=14, pad=15)
    
    sp.critical_difference_diagram(ranks, test_results, ax=ax3, label_fmt_left='{label}', label_fmt_right='{label}')
    
    ax3.grid(which='major', axis='x', linestyle='--', linewidth=0.5, color='gray', alpha=0.5)
    ax3.set_xlabel('Average Rank', fontsize=12)
    for spine in ['top', 'right', 'left', 'bottom']:
        ax3.spines[spine].set_visible(False)
    ax3.spines['bottom'].set_visible(True)
    
    plt.tight_layout()
    plt.show()
    if _save:
        fig3.savefig('./fig4E_CD_sig_CDline.pdf', dpi=300, bbox_inches='tight')