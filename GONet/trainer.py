#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/1/29 12:36
# @Author  : Eayon
# @Site    :
# @File    : trainer.py
# @Software: PyCharm
# @logit:

import os
from torch.nn.utils import clip_grad_norm_
from torch_geometric.graphgym import optim
from torch.optim import Adam, RAdam
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split, StratifiedKFold
from torch_geometric.loader import DataLoader
from sklearn.metrics import (accuracy_score, roc_auc_score,
                             roc_curve, classification_report,
                             precision_score, recall_score, f1_score)
from . import dataset
from . import _utils_
from .GONet_model import GONet
from ._utils_ import Plot_multi_ROC, calculate_metrics, plot_training_curves


class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_score is None:  # 第一次无条件更新
            self.best_score = val_loss
        elif val_loss < self.best_score - self.delta:
            self.counter = 0
            self.best_score = val_loss
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

class Trainer_early(object):
    def __init__(self, args, model):
        self.args = args
        self.device = self.args.device
        self.model = model.to(self.args.device)
        self.weight_decay = self.args.weight_decay
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr, weight_decay=self.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=10, verbose=True)

        self.cls_w = self.args.cls_w
        self.ae_w = self.args.ae_w
        self.gnn_w = self.args.gnn_w
        self.cont_w = self.args.cont_w

    def compute_loss(self, cls_loss, AE_recon_loss, gnn_loss, contra_loss):

        loss_cls = self.cls_w*cls_loss
        loss_ae = self.ae_w*AE_recon_loss
        loss_gnn = self.gnn_w*gnn_loss
        loss_contra = self.cont_w*contra_loss
        total_loss = loss_cls + loss_ae + loss_gnn + loss_contra

        loss_dict = {
        'cls_loss': loss_cls.item(),
        'AE_recon_loss': loss_ae.item(),
        'gnn_loss': loss_gnn.item(),
        'contra_loss': loss_contra.item(),
        'total_loss': total_loss.item()}
        # print('loss_dict',loss_dict)
        
        return total_loss, loss_dict

    def train(self, train_loader, omics_mapping, epoch):
        # print('---------------------training-----------------------------------')
        self.model.train()
        total_loss = 0.0
        all_probs, y_true = [],[]
        batch_loss_dict = {'cls_loss': [], 'AE_recon_loss': [], 'gnn_loss': [], 'contra_loss': [], 'total_loss': []}
        batch_omics_weights, batch_mode_weights = [],[]
        batch_omics, batch_global, batch_gnn, batch_embedding, batch_labels = [],[],[],[],[]

        for current_batch, data in enumerate(train_loader, start=1): 
            x, edge_index, edge_weight, y, batch, ae_x, intra_edge_index, cross_edge_index, intra_mask, cross_mask = self._prepare_data(data)
            self.optimizer.zero_grad()
            global_x, final_x, gnn_x, omics_x, omics_weights, gate_weights, cls_loss, class_scores, AE_recon_loss, contra_loss, gnn_loss = self.model(
                x, edge_index, intra_edge_index, cross_edge_index, y, ae_x, batch, omics_mapping, status='training', intra_edge_weight=intra_mask, cross_edge_weight=cross_mask)
            
            global_x = torch.zeros_like(torch.tensor(x)) if global_x is None else global_x
            omics_x = torch.zeros_like(torch.tensor(x)) if omics_x is None else omics_x
            omics_weights = torch.zeros_like(torch.tensor(x)) if omics_weights is None else omics_weights
            batch_omics_weights.append(torch.mean(omics_weights,dim=0))
            batch_mode_weights.append(torch.mean(gate_weights,dim=0))
            batch_embedding.append(final_x.detach().cpu().numpy())
            batch_global.append(global_x.detach().cpu().numpy())
            batch_gnn.append(gnn_x.detach().cpu().numpy())
            batch_omics.append(omics_x.detach().cpu().numpy())
            all_probs.append(class_scores.detach().cpu().numpy())
            
            loss, loss_dict = self.compute_loss(cls_loss, AE_recon_loss, gnn_loss, contra_loss)
            loss.backward()
            clip_grad_norm_(self.model.parameters(), max_norm=5.0)  # gradient clipping
            self.optimizer.step()
            
            total_loss += loss.item()  
            
            y_true.append(y.cpu().numpy())
            for key in batch_loss_dict:
                batch_loss_dict[key].append(loss_dict[key])
            if current_batch % 5 == 0: 
                total_norm = 0
                for p in self.model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                # print(f"Gradient Norm: {total_norm:.4f}")
                print(f"Epoch {epoch}, Batch {current_batch}: {loss_dict}")
        avg_loss_dict = {key: np.mean(values) for key, values in batch_loss_dict.items()}        
        return total_loss / len(train_loader), self.model, avg_loss_dict, batch_global, batch_embedding, batch_gnn, batch_omics, batch_mode_weights, batch_omics_weights, all_probs

    def evaluate(self, model, test_loader, status, omics_mapping):
        # print('---------------------evaluating-----------------------------------')
        model.eval()
        total_loss = 0.0
        all_probs, y_true = [],[]

        with torch.no_grad():
            for _, data in enumerate(test_loader, start=1):
                x, edge_index, edge_weight, y, batch, ae_x, intra_edge_index, cross_edge_index, intra_mask, cross_mask = self._prepare_data(data)
                global_x, final_x, gnn_x, omics_x, omics_weights, gate_weights, cls_loss, class_scores, AE_recon_loss, contra_loss, gnn_loss = model(
                    x, edge_index, intra_edge_index, cross_edge_index, y, ae_x, batch, omics_mapping, status, intra_edge_weight=intra_mask, cross_edge_weight=cross_mask)
                global_x = torch.zeros_like(torch.tensor(x)) if global_x is None else global_x
                omics_x = torch.zeros_like(torch.tensor(x)) if omics_x is None else omics_x
                omics_weights = torch.zeros_like(torch.tensor(x)) if omics_weights is None else omics_weights
                
                loss, _ = self.compute_loss(cls_loss, AE_recon_loss, gnn_loss, contra_loss)
                total_loss += loss.item()
                all_probs.append(class_scores.cpu().numpy())
                y_true.append(y.cpu().numpy())
        avg_loss = total_loss / len(test_loader)
        all_probs = np.vstack(all_probs)
        y_true = np.concatenate(y_true).astype(int)
        
        precision, recall, f1, accuracy, roc_auc, auprc = calculate_metrics(y_true, all_probs)
        if status == 'test':
            roc_auc = Plot_multi_ROC(y_true, all_probs, self.args, save=True)

        return avg_loss, global_x, final_x, gnn_x, omics_x, omics_weights, gate_weights, precision, recall, f1, accuracy, roc_auc, auprc, all_probs
    
    def _prepare_data(self, data):
        x, edge_index, edge_weight, y, batch, intra_edge_index, cross_edge_index, intra_mask, cross_mask = (
            data.x.to(self.device), data.edge_index.to(self.device), data.edge_attr.to(self.device), 
            data.y.to(self.device), data.batch.to(self.device), data.intra_edge_index.to(self.device), data.cross_edge_index.to(self.device), data.intra_mask.to(self.device), data.cross_mask.to(self.device)
        )
        patient = [str(patient) for patient in data.patient_name]
        ae_x = pd.DataFrame(dataset.batch_data_matrix(x, data.batch, len(y)), index=patient)
        return x, edge_index, edge_weight, y, batch, ae_x, intra_edge_index, cross_edge_index, intra_mask, cross_mask

def train_on_full_training_set(args, omics_mapping, train_graph_list, test_graph_list):
    train_loader = DataLoader(train_graph_list, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_graph_list, batch_size=len(test_graph_list), shuffle=False)

    ## ! 1.构建最优参数的模型
    omics_dim = train_graph_list[0].x.size(0)
    best_model = GONet(
        args, omics_dim, args.AElayer_numer, args.gnn_out_dim, args.gnn_type, args.AE_module, args.head_num, args.GNN_module, args.omics_module, args.dropout, args.GNNlayer_numer, args.temperature, args.omics_head).to(args.device)
    model_trainer = Trainer_early(args, best_model)
    _utils_.seed_everything(args.seed)

    # 2.Train and evaluate
    early_stopping = EarlyStopping(patience=5, delta=0.01) #
    lr_list = []
    best_val_loss, best_model_state, best_val_auc = float('inf'), None, -np.inf
    epoch_weights = {'omics_weights': [], 'gate_weights': []}
    epoch_embeddings = {'epoch_final':[], 'epoch_global':[], 'epoch_gnn':[], 'epoch_omics':[], 'epoch_probs':[] }
    epoch_loss_dict = {'cls_loss': [], 'AE_recon_loss': [], 'gnn_loss': [], 'contra_loss': [], 'total_loss': []}

    for epoch in tqdm(range(args.epoches)):
        train_loss, model_done, avg_loss_dict, global_x, final_x, gnn_x, omics_x, gate_weights, omics_weights, all_probs_train = model_trainer.train(train_loader, omics_mapping, epoch)
        epoch_embeddings['epoch_final'].append(np.concatenate(final_x, axis=0))
        epoch_embeddings['epoch_global'].append(np.concatenate(global_x, axis=0))
        epoch_embeddings['epoch_gnn'].append(np.concatenate(gnn_x, axis=0))
        epoch_embeddings['epoch_omics'].append(np.concatenate(omics_x, axis=0))
        epoch_embeddings['epoch_probs'].append(np.concatenate(all_probs_train, axis=0))
        epoch_weights['gate_weights'].append(torch.mean(torch.stack(gate_weights), dim=0).detach().cpu().numpy())   
        epoch_weights['omics_weights'].append(torch.mean(torch.stack(omics_weights), dim=0).detach().cpu().numpy())
        weights_np = {
            key: [t.cpu().numpy() if isinstance(t, torch.Tensor) else t for t in value]
            for key, value in epoch_weights.items()}
        
        if train_loss < best_val_loss:
            best_val_loss = train_loss
            best_model_state = model_done.state_dict()
        model_trainer.scheduler.step(train_loss)
        early_stopping(train_loss)

        if early_stopping.early_stop:
            print(f"Early stopping at epoch [{epoch+1}]")
            break
            
        lr_list.append(model_trainer.optimizer.param_groups[0]['lr'])
        for key in avg_loss_dict:
            epoch_loss_dict[key].append(avg_loss_dict[key])
    plot_training_curves(epoch_loss_dict,lr_list)
    # plot_grad(epoch_gradient_stats)
    
    # 加载最优模型并保存
    save_dir = f"./results/embeddings_{args.cancer_type}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    if (args.AE_module == 'ae') and (args.GNN_module == 'all') and (args.omics_module == 'att') and (args.fusion == 'all'):
        best_model.load_state_dict(best_model_state)
        torch.save(best_model.state_dict(), os.path.join(save_dir,f'best_model_{args.cancer_type}.pkl'))

    test_loss, global_x_te, final_x_te, gnn_x_te, omics_x_te, omics_weights_te, gate_weights_te, precision, recall, f1, accuracy, roc_auc, auprc, all_probs = model_trainer.evaluate(model_done, test_loader, 'test', omics_mapping)

    np.savez(os.path.join(save_dir,f'test_{args.cancer_type}_{args.omics}_{args.AE_module}_{args.GNN_module}_{args.omics_module}_{args.fusion}_embedding.npz'),
                    final_x_tra=epoch_embeddings['epoch_final'],
                    global_x_tra=epoch_embeddings['epoch_global'],
                    gnn_x_tra=epoch_embeddings['epoch_gnn'],
                    omics_x_tra=epoch_embeddings['epoch_omics'],
                    probs=epoch_embeddings['epoch_probs'],
                    # gate_weights=gate_weights,
                    # omics_weights=omics_weights,
                    **weights_np,

                    final_x_te=final_x_te.cpu(),
                    global_x=global_x_te.cpu(),
                    gnn_x=gnn_x_te.cpu(),
                    omics_x=omics_x_te.cpu(),
                    omics_weights_te=omics_weights_te.cpu(),
                    gate_weights_te=gate_weights_te.cpu(),
                    all_probs=all_probs,
                    labels=[test_i.y for test_i in test_graph_list]
                )

    return precision, recall, f1, accuracy, roc_auc, auprc, model_done, global_x, final_x, gnn_x, omics_x, gate_weights, omics_weights, all_probs

def cross_validate_training(args, omics_mapping, data_list, labels, n_splits):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=args.seed)
    all_metrics = {"precision": [], "recall": [], "f1": [], "accuracy": [], "roc_auc": [],"auprc": [],'fold':[]}

    for fold, (train_idx, val_idx) in enumerate(skf.split(data_list, labels)):
        print(f'Fold {fold + 1}/{n_splits}')
        # 1.data loading: Split the data into training and validation sets and Create data loaders
        train_data = [data_list[i] for i in train_idx]
        val_data = [data_list[i] for i in val_idx]
        train_loader = DataLoader(train_data, batch_size=args.batch_size,
                                 shuffle=True, pin_memory=True)
        val_loader = DataLoader(val_data, batch_size=args.batch_size, 
                             shuffle=False, pin_memory=True)

        # 2. Initialize model and trainer
        omics_dim = train_data[0].x.size(0)#*(len(omics_mapping)+1) 

        gnn_net = GONet(
            args, omics_dim, args.AElayer_numer, args.gnn_out_dim, args.gnn_type, args.AE_module, args.head_num, args.GNN_module, args.omics_module, args.dropout, args.GNNlayer_numer, args.temperature, args.omics_head).to(args.device)

        model_trainer = Trainer_early(args, gnn_net)
        _utils_.seed_everything(args.seed)
        best_metrics = {"precision":0, "recall":0, "f1":0, "accuracy":0, "roc_auc":0, "auprc":0}

        # 3.Train and evaluate on this fold by training on multi epoches
        train_loss_list, val_loss_list = [],[]
        early_stopping = EarlyStopping(patience=5)

        for epoch in tqdm(range(args.num_epoches), desc=f'Training fold {fold+1}'):
            train_loss, model_done, _, _, _, _, _, _, _, _ = model_trainer.train(train_loader, omics_mapping, epoch)
            val_loss, _, _, _, _, _, _, precision, recall, f1, accuracy, val_rocauc, auprc, all_probs = model_trainer.evaluate(model_done, val_loader, 'evaluation', omics_mapping)
            train_loss_list.append(train_loss)
            val_loss_list.append(val_loss)

            print(f'epoch [{epoch+1}/{args.num_epoches}], train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}')

            if val_rocauc > best_metrics["roc_auc"]:
                best_metrics.update({
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "accuracy": accuracy,
                    "roc_auc": val_rocauc,
                    "auprc": auprc
                })
            model_trainer.scheduler.step(train_loss)
            early_stopping(val_loss)

            if early_stopping.early_stop:
                print(f"Early stopping at epoch [{epoch+1}]")
                break
            print(f'currrent results in {fold}-{epoch}:', precision, recall, f1, accuracy, auprc, val_rocauc)

        for metric in ['precision','recall','f1','accuracy','roc_auc','auprc']:
            all_metrics[metric].append(best_metrics[metric])
        all_metrics['fold'].append(fold+1)

    for metric, values in all_metrics.items():
        if metric != 'fold':
            print(f'{metric.capitalize()}: {np.mean(values):.4f} ± {np.std(values):.4f}')
    return all_metrics


