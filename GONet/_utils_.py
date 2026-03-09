#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/3/22 17:14
# @Author  : Eayon
# @Site    : 
# @File    : _utils_.py.py
# @Software: PyCharm
# @logit: model related auxiliary function

import os
# import args
import torch
import random
import numpy as np
import itertools
import pandas as pd
from tqdm import tqdm
import torch.nn as nn

from sklearn.metrics import accuracy_score, roc_auc_score, auc, roc_curve, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import (accuracy_score, roc_auc_score,average_precision_score, precision_recall_curve,
                             roc_curve, classification_report,
                             precision_score, recall_score, f1_score)
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, LabelEncoder, LabelBinarizer, OneHotEncoder, label_binarize 

from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap
import re
import glob
from itertools import chain, repeat
from typing import Dict, List, Optional
from matplotlib.patches import Patch
import gc
import argparse
import optuna
import joblib
import json
from optuna.pruners import SuccessiveHalvingPruner
from sklearn.model_selection import train_test_split

def set_environment(random_seed):
    # Set random seed for reproducibility
    seed_everything(random_seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    torch.use_deterministic_algorithms(True,warn_only=True)   #这个可以解决某些包的版本太新，一般的固定随机数无效，不能复现结果的情况。os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # 或者 ":16:8"这个命令必须在加载torch和cuda相关参数之前指定

########################################### data processing auxiliary functions
def min_max_normalization(tensor):
    min_val = torch.min(tensor)
    max_val = torch.max(tensor)
    normalized_data = (tensor-min_val)/(max_val-min_val)
    return normalized_data


def calculate_metrics(y_true, all_probs):
    """
    calculate metrics based on true and predicted label
    :param y_true: true label list
    :param all_probs: predicted probability
    :return: metric index
    """
    is_multiclass = all_probs.shape[1] > 2
    if is_multiclass:
        y_pred = np.argmax(all_probs, axis=1)
        precision = precision_score(y_true, y_pred, average='weighted',zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        accuracy = accuracy_score(y_true, y_pred)

        y_true_bin = label_binarize(y_true, classes=np.unique(y_true))
        
        auroc = roc_auc_score(y_true_bin, all_probs, multi_class='ovr', average='macro')
        auprc_list = []
        for i in range(y_true_bin.shape[1]):
            ap = average_precision_score(y_true_bin[:, i], all_probs[:, i])
            auprc_list.append(ap)
        auprc = np.nanmean(auprc_list)
    else:        
        if all_probs.ndim == 2 and all_probs.shape[1] == 2:
            y_prob = all_probs[:, 1]
        else:
            y_prob = all_probs.ravel()
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        youden_idx = np.argmax(tpr-fpr)
        best_thred = thresholds[youden_idx]
        y_pred_opt = (y_prob >= best_thred).astype(int)
        precision = precision_score(y_true, y_pred_opt,zero_division=0) 
        recall = recall_score(y_true, y_pred_opt, zero_division=0)
        f1 = f1_score(y_true, y_pred_opt, zero_division=0)
        accuracy = accuracy_score(y_true, y_pred_opt)

        auroc = roc_auc_score(y_true, y_prob)
        auprc = average_precision_score(y_true, y_prob)
    return precision, recall, f1, accuracy, auroc, auprc

def Plot_multi_ROC(all_labels, all_probs, args, save=True):
    # Plot ROC Curve for multi classification
    print('all_labels:', list(all_labels))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    y_test_onehot = torch.eye(args.num_classes)[all_labels % args.num_classes]
    for i in range(args.num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_onehot[:, i], all_probs[:, i])  # all_probs is array类型的list
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_onehot.ravel(), all_probs.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    plt.figure(figsize=(8, 6))

    # plot roc curve for each class
    for i in range(args.num_classes):
        plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')
    # plot micro curve
    plt.plot(fpr["micro"], tpr["micro"], label=f'Micro-average (AUC = {roc_auc["micro"]:.2f})', color='orange',
             linestyle='-', linewidth=2)
    # plot diag line as reference
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Multi-class Classification')
    plt.legend(loc="lower right")
    
    model_params_flag = args.cancer_type+'_'+args.omics+'_AE_'+ str(args.AElayer_numer) +'_'+args.gnn_type+str(args.GNNlayer_numer) +'_head_'+str(args.head_num)
    modules = {"AE": args.AE_module,"GNN": args.GNN_module,"omics": args.omics_module}
    module_flag = "_" + "_".join(f"{'no' if not status else ''}{name}"
                                 for name, status in modules.items())

    file_name = model_params_flag + module_flag+'_multi_classification_ROC_curve_figure.png'
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path, exist_ok=True)
    save_name = os.path.join(args.result_path, file_name)
    os.makedirs(os.path.dirname(save_name), exist_ok=True)
    if save:
        plt.savefig(save_name)
    plt.close()
    return roc_auc["micro"]

def plot_training_curves(loss_history, lr_history):
    # epochs = range(1, len(loss_history) + 1)
    
    # 绘制总损失曲线
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    epochs = range(1, len(loss_history['total_loss']) + 1)
    
    # plt.figure(figsize=(10, 6))
    plt.plot(epochs, loss_history['cls_loss'], label='cls_loss', marker='o')
    plt.plot(epochs, loss_history['AE_recon_loss'], label='AE_recon_loss', marker='o')
    plt.plot(epochs, loss_history['gnn_loss'], label='gnn_loss', marker='o')
    plt.plot(epochs, loss_history['contra_loss'], label='contra_loss', marker='o')
    plt.plot(epochs, loss_history['total_loss'], label='total_loss', marker='o')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('loss curve')
    plt.legend()
    # plt.grid(True)
    plt.show()
    
    # 绘制学习率曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, lr_history, label="Learning Rate")
    plt.xlabel("Epochs")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Curve")
    plt.legend()
    plt.show()

