



import os
import torch
import gc
import optuna
import joblib
import json
import pandas as pd
import numpy as np
import itertools
from optuna.pruners import SuccessiveHalvingPruner
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DenseDataLoader

import argparse
from GONet.trainer import cross_validate_training
from GONet.dataset import load_data
from GONet._utils_ import seed_everything, set_environment

class get_args:
    def __init__(self, param_list):
        self.cancer_type = param_list['cancer_type']
        self.workshop_path=f'./data/{cancer_type}'
        self.result_path='./results'
        self.num_classes = param_list['num_classes']

        self.omics = param_list['omics']
        self.interaction_type = 'rela'  # 'combined'\rela\corr

        if self.interaction_type == 'combined':
            self.omics_data_path = os.path.join(self.workshop_path, f'{self.omics}_combined_df.csv')
            self.omics_data_test_path = os.path.join(self.workshop_path, f'{self.omics}_combined_df_test.csv')
        else:
            self.omics_data_path = os.path.join(self.workshop_path, f'{self.omics}_df.csv')
            self.omics_data_test_path = os.path.join(self.workshop_path, f'{self.omics}_df_test.csv')

        ## 1, model structure args
        self.AElayer_numer = param_list['AElayer_numer']
        self.GNNlayer_numer = param_list['GNNlayer_numer']
        self.gnn_out_dim = param_list['gnn_out_dim']
        self.gnn_type = param_list['gnn_type']
        self.head_num = param_list['head_num']
        self.dropout = param_list['dropout']
        self.temperature = param_list['temperature']
        self.omics_head = param_list['omics_head']

        ## 2,model training
        self.num_epoches = param_list['num_epoches']
        self.batch_size = 10
        self.weight_decay = param_list['weight_decay']
        self.lr = param_list['lr']
        self.cls_w = param_list['cls_w']
        self.ae_w = param_list['ae_w']
        self.gnn_w = param_list['gnn_w']
        self.cont_w = param_list['cont_w']
    
        ## 3, model ablation args
        self.AE_module = param_list['AE_module']
        self.GNN_module = param_list['GNN_module']
        self.omics_module = param_list['omics_module']
        self.fusion = param_list['fusion']

        self.seed = 42
        self.flag = 'test'
        self.status = 'training'
        self.contrast_loss = True
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def optuna_train(args):
    workshop_path, omics_data_path, omics_data_test_path = args.workshop_path, args.omics_data_path, args.omics_data_test_path
    omics_graph_path = os.path.join(workshop_path, f'{args.omics}_{args.interaction_type}_graph.pt')
    omics_label_path = os.path.join(workshop_path, f'{args.omics}_clinic.csv')
    omics_label_test_path = os.path.join(workshop_path, f'{args.omics}_clinic_test.csv')

    # Load and preprocess data
    # _, train_graph_list, train_labels, _, test_graph_list, _ = load_data(omics_data_path, omics_graph_path, omics_label_path, omics_data_test_path, omics_label_test_path)
    _, train_graph_list, train_labels, _, test_graph_list, test_labels, omics_mapping = load_data(workshop_path, omics_data_path, omics_graph_path, omics_label_path, omics_data_test_path, omics_label_test_path, omics)

    ##############################################! 2.model training
    all_metrics = cross_validate_training(args, omics_mapping, train_graph_list, train_labels, n_splits=5)
    precision_mean = np.mean(all_metrics['precision'])
    accuracy_mean = np.mean(all_metrics['accuracy'])
    auprc_mean = np.mean(all_metrics['auprc'])
    recall_mean = np.mean(all_metrics['recall'])
    f1_mean = np.mean(all_metrics['f1'])
    roc_auc_mean = np.mean(all_metrics['roc_auc'])
    return precision_mean, recall_mean, f1_mean, accuracy_mean, roc_auc_mean, auprc_mean
    
def objective(trial):
    ## 1,define hypermater space
    param_list = {
        'cancer_type':cancer_type,
        'omics': omics,
        'num_classes':num_classes,
        'AElayer_numer': trial.suggest_int('AElayer_numer',2,5), 
        'GNNlayer_numer' : trial.suggest_int('GNNlayer_numer', 2, 5),
        'gnn_out_dim': trial.suggest_categorical('gnn_out_dim',  [16, 32, 64]),
        'gnn_type': trial.suggest_categorical('gnn_type', ['DGI_GCN']),
        'head_num':  trial.suggest_categorical('head_num', [2,4,8]),
        'temperature':trial.suggest_float('temperature', 0.01, 1),
        'omics_head': trial.suggest_categorical('omics_head', [2,4,8]),
        'dropout':trial.suggest_float('dropout', 0.1, 0.5),
        'weight_decay':trial.suggest_float('weight_decay', 1e-5, 1e-1, log=True),
        'lr':trial.suggest_float('lr', 1e-5, 1e-1, log=True),
        'AE_module': trial.suggest_categorical('AE_module', ['ae']),
        'GNN_module':trial.suggest_categorical('GNN_module', ['all']),
        'omics_module':trial.suggest_categorical('omics_module', ['att']),
        'fusion':trial.suggest_categorical('fusion', ['all']),
        'num_epoches':trial.suggest_int('num_epoches', 50, 500, log=True),

        'cls_w':trial.suggest_float('cls_w', 2, 5, log=True),
        'ae_w':trial.suggest_float('ae_w', 1, 3, log=True),
        'gnn_w':trial.suggest_float('gnn_w', 1, 3, log=True),
        'cont_w':trial.suggest_float('cont_w', 0.01, 1, log=True),
                    }
    gc.collect()
    torch.cuda.empty_cache()
    seed_everything(42)
    set_environment(42)

    ## 2,model training
    # try:
    args = get_args(param_list)
    print('current_args',param_list)
    # 得到当前参数下五折交叉验证的最终结果
    precision, recall, f1, accuracy, roc_auc, auprc = optuna_train(args)

    # save_results_to_csv(trial.number, param_list, precision, recall, f1, accuracy, roc_auc)
    return roc_auc

def save_study_callback(study, trail):
    df = study.trials_dataframe() #将所有试验结果转化为DF形式
    df.to_csv(f'./optuna_results/att_optuna_{cancer_type}_mRNA_miRNA_study_results.csv', index=False)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cancer_type", type=str, required=True, help="cancer_type")
    parser.add_argument("--omics", type=str, required=True, help="omics type")
    parser.add_argument("--num_classes", type=str, required=True, help="num_classes type")

    sh_args = parser.parse_args()
    cancer_type = sh_args.cancer_type.strip()
    omics = sh_args.omics.strip()
    num_classes = int(sh_args.num_classes.strip())


    study = optuna.create_study(direction='maximize') # minimize, pruner=SuccessiveHalvingPruner()
    study.optimize(objective, n_trials=50, callbacks=[save_study_callback])
    print('best trial:')
    print(study.best_params) # 最佳参数组合
    print(study.best_value) # 最优性能指标

    if not os.path.exists('./optuna_results'):
        os.makedirs('./optuna_results', exist_ok=True)
    joblib.dump(study, f'./optuna_results/att_study_{cancer_type}.pkl')
    with open(f'./optuna_results/att_best_params_{cancer_type}.json', 'w') as f:
        json.dump(study.best_params, f)