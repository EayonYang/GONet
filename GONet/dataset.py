#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/1/30 16:39
# @Author  : Eayon
# @Site    : 
# @File    : dataset.py
# @Software: PyCharm
# @logit:

import os
import numpy as np
import torch
import pandas as pd
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DenseDataLoader, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from . import _utils_

def dataset(graph_data, omics_data, label_data, omics_mapping):
    """
    :param graph_data:
    :param omics_data: (sample,feature),row is the patient, column is the omics features
    :param label_data:
    :return: each patient's graph data, which includes x[each node corresponds a one-dimensional feature], edge_index[interaction between nodes], y[patient label] and patientName
    """
    edge_index, edge_attr = graph_data.edge_index, graph_data.edge_attr
    reverse_dict = {value: key for key, values in omics_mapping.items() for value in values}
    src_type = [reverse_dict.get(tensor_value.item(), "unknown") for tensor_value in edge_index[0]]
    dst_type = [reverse_dict.get(tensor_value.item(), "unknown") for tensor_value in edge_index[1]]

    type_mapping = {values:ind for ind, values in enumerate(omics_mapping.keys())}
    src_types = torch.tensor([type_mapping[type] for type in src_type])
    dst_types = torch.tensor([type_mapping[type] for type in dst_type])
    same_type_edges = src_types==dst_types
    edge_type = torch.where(same_type_edges, torch.tensor(1, dtype=torch.int), torch.tensor(0, dtype=torch.int))
    intra_mask = (edge_type == 1)
    cross_mask = (edge_type == 0)
    intra_indices = torch.nonzero(intra_mask).squeeze()
    cross_indices = torch.nonzero(cross_mask).squeeze()
    intra_edge_index = edge_index[:, intra_mask]
    cross_edge_index = edge_index[:, cross_mask]

    omics_onehot = torch.zeros(len(reverse_dict), len(omics_mapping))
    for omics_type, indices in omics_mapping.items():
        omics_onehot[indices, type_mapping[omics_type]] = 1

    patient_name = omics_data.index.tolist()
    label_data = label_data.loc[patient_name]
    y = LabelEncoder().fit_transform(label_data['label'].values.tolist())
    label_tensor = torch.tensor(y, dtype=torch.long)

    graph_data_list = []
    for i, patient_name_i in enumerate(patient_name):
        raw_features = torch.tensor(omics_data.iloc[i, :].values, dtype=torch.float)
        node_features = torch.cat([
            raw_features.unsqueeze(1),  
            omics_onehot                
        ], dim=1)
        graph_data = Data(
            x=node_features,#patient_features.view(-1, 1), 
            edge_index=edge_index, 
            intra_edge_index=intra_edge_index,
            cross_edge_index=cross_edge_index,
            y=label_tensor[i], 
            patient_name=str(patient_name_i), 
            edge_attr=edge_attr, 
            intra_mask=intra_indices,
            cross_mask=cross_indices,
            edge_type=edge_type) #
        graph_data_list.append(graph_data)
    # loader = DataLoader(graph_data_list, batch_size=10, shuffle=True)
    return graph_data_list, label_tensor.cpu().numpy()

def load_data(workshop_path, omics_data_path, omics_graph_path, omics_label_path, omics_data_test_path, omics_label_test_path, omics):
    train_omics = pd.read_csv(omics_data_path, index_col=0)
    train_label_data = pd.read_csv(omics_label_path, index_col=0)
    test_omics = pd.read_csv(omics_data_test_path, index_col=0)
    test_label_data = pd.read_csv(omics_label_test_path, index_col=0)
    graph_data = torch.load(omics_graph_path)

    omics_mapping = cal_mapping(workshop_path, omics)
    train_graph_data_list, train_labels_list = dataset(graph_data, train_omics, train_label_data, omics_mapping)
    test_graph_data_list, test_labels_list = dataset(graph_data, test_omics, test_label_data, omics_mapping)

    return train_omics, train_graph_data_list, train_labels_list, test_omics, test_graph_data_list, test_labels_list, omics_mapping

def create_data_loaders(train_graph_list, test_graph_list, batch_size, random_seed):
    _utils_.seed_everything(random_seed)  # 保证每个批次的数据是一致的
    x, val = train_test_split(train_graph_list, test_size=0.1, random_state=random_seed)
    train_loader = DataLoader(x, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_graph_list, batch_size=len(test_graph_list), shuffle=False)
    return train_loader, val_loader, test_loader

def batch_data_matrix(x, batch, num_graphs):
    feature_matrix = []
    for i in range(num_graphs):
        node_indices = (batch == i).nonzero(as_tuple=True)[0]
        flattened_features = x[node_indices][:, 0].tolist()
        feature_matrix.append(flattened_features)
    return feature_matrix

def cal_mapping(workshop_path, omics):
    
    if omics=='mRNA':
        mRNA_df = pd.read_csv(os.path.join(workshop_path, 'mRNA_df.csv'), index_col=0)
        num_mRNA = mRNA_df.shape[1]
        omics_mapping = {
            'mRNA':list(range(num_mRNA))}
    elif omics=='protein':
        protein_df = pd.read_csv(os.path.join(workshop_path, 'protein_df.csv'), index_col=0)
        num_protein = protein_df.shape[1]
        omics_mapping = {
            'protein':list(range(num_protein))}
    elif omics=='miRNA':
        miRNA_df = pd.read_csv(os.path.join(workshop_path, 'miRNA_df.csv'), index_col=0)
        num_miRNA = miRNA_df.shape[1]
        omics_mapping = {
            'miRNA':list(range(num_miRNA))}
    elif omics=='protein_miRNA':
        protein_df = pd.read_csv(os.path.join(workshop_path, 'protein_df.csv'), index_col=0)
        miRNA_df = pd.read_csv(os.path.join(workshop_path, 'miRNA_df.csv'), index_col=0)
        num_miRNA, num_protein = miRNA_df.shape[1], protein_df.shape[1]
        omics_mapping = {
            'protein':list(range(num_protein)),
            'miRNA':list(range(num_protein, num_miRNA+num_protein)),
        }
    elif omics=='mRNA_miRNA':
        mRNA_df = pd.read_csv(os.path.join(workshop_path, 'mRNA_df.csv'), index_col=0)
        num_mRNA = mRNA_df.shape[1]
        miRNA_df = pd.read_csv(os.path.join(workshop_path, 'miRNA_df.csv'), index_col=0)
        num_miRNA = miRNA_df.shape[1]
        omics_mapping = {
            'mRNA':list(range(num_mRNA)),
            'miRNA':list(range(num_mRNA, num_mRNA+num_miRNA)),
        }
    elif omics=='mRNA_protein_miRNA':
        mRNA_df = pd.read_csv(os.path.join(workshop_path, 'mRNA_df.csv'), index_col=0)
        protein_df = pd.read_csv(os.path.join(workshop_path, 'protein_df.csv'), index_col=0)
        miRNA_df = pd.read_csv(os.path.join(workshop_path, 'miRNA_df.csv'), index_col=0)
        num_miRNA, num_protein, num_mRNA = miRNA_df.shape[1], protein_df.shape[1], mRNA_df.shape[1]
        omics_mapping = {
            'mRNA':list(range(num_mRNA)),
            'protein':list(range(num_mRNA, num_mRNA+num_protein)),
            'miRNA':list(range(num_mRNA+num_protein, num_mRNA+num_protein+num_miRNA)),
        }
    else:
        mRNA_df = pd.read_csv(os.path.join(workshop_path, 'mRNA_df.csv'), index_col=0)
        protein_df = pd.read_csv(os.path.join(workshop_path, 'protein_df.csv'), index_col=0)
        num_mRNA = mRNA_df.shape[1]
        num_protein = protein_df.shape[1]
        omics_mapping = {
            'mRNA':list(range(num_mRNA)),
            'protein':list(range(num_mRNA, num_mRNA+num_protein)),
        }
    return omics_mapping

def add_laplacian_feature(data_list, lap_k):
    from torch_geometric.transforms import AddLaplacianEigenvectorPE, AddRandomWalkPE

    np.random.seed(42)
    new_data_list = []
    v0_init = np.random.rand(data_list[0].x.shape[0])
    ncv = 30 * lap_k + 1
    transform = AddLaplacianEigenvectorPE(k=int(lap_k), attr_name='PE_features', is_undirected=True, v0=v0_init, ncv=ncv)
    for data_i in data_list:
        if data_i.edge_index.dtype != torch.long:
            data_i.edge_index = data_i.edge_index.long()
        data = transform(data_i)
        data_i.PE_features = torch.cat([data_i.x, data.PE_features], dim=1)
        new_data_list.append(data_i)
    return new_data_list