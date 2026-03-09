import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import LayerNorm
import os.path as osp
import time
import torch_geometric.transforms as T
from torch_geometric.nn import GAE, VGAE, GCNConv, GATConv, GINConv, GATv2Conv, GPSConv,DeepGraphInfomax, SAGEConv, BatchNorm, TransformerConv
from ._utils_ import seed_everything, min_max_normalization


class GNN(nn.Module):
    def __init__(self, args_seed, gnn_type, in_channels, out_channels, head_num, dropout):
        super().__init__()
        seed_everything(seed=args_seed)
        self.gnn_type = gnn_type
        
        if gnn_type == 'GCN':
            dim = out_channels//head_num*head_num
            self.residual = nn.Linear(in_channels, dim)
            self.dropout = dropout
            self.norm = nn.LayerNorm(out_channels)
            self.conv = GCNConv(in_channels, out_channels)
        elif gnn_type == 'GAT':
            dim = out_channels//head_num*head_num
            self.residual = nn.Linear(in_channels, dim)
            self.dropout = dropout
            self.norm = nn.LayerNorm(out_channels)
            self.conv = GATConv(in_channels, out_channels//head_num, head_num)
        elif gnn_type == 'SAGEConv':
            dim = out_channels//head_num*head_num
            self.residual = nn.Linear(in_channels, dim)
            self.dropout = dropout
            self.norm = nn.LayerNorm(out_channels)
            self.conv = SAGEConv(in_channels, out_channels)
        elif gnn_type == 'GIN':
            dim = out_channels//head_num*head_num
            self.residual = nn.Linear(in_channels, dim)
            self.dropout = dropout
            self.norm = nn.LayerNorm(out_channels)
            mlp = nn.Sequential(
                nn.Linear(in_channels, out_channels),
                nn.LeakyReLU(),
                nn.Linear(out_channels, out_channels))
            self.conv = GINConv(mlp)
        elif gnn_type == 'GATv2':
            dim = out_channels//head_num*head_num
            self.residual = nn.Linear(in_channels, dim)
            self.dropout = dropout
            self.norm = nn.LayerNorm(out_channels)
            self.conv = GATv2Conv(in_channels, out_channels//head_num, heads=head_num)
            self.mlp = nn.Linear(out_channels, out_channels)
        elif gnn_type == 'GAE':
            dim = out_channels//head_num*head_num
            self.residual = nn.Linear(in_channels, dim)
            self.dropout = dropout
            self.norm = nn.LayerNorm(out_channels)
            encoder = GCNConv(in_channels, out_channels)
            self.conv = GAE(encoder)
        elif gnn_type == 'GraphTransformer':
            dim = out_channels//head_num*head_num
            self.residual = nn.Linear(in_channels, dim)
            self.dropout = dropout
            self.norm = nn.LayerNorm(out_channels)
            self.conv = TransformerConv(in_channels, out_channels // head_num, heads=head_num)  
            self.mlp = nn.Linear(out_channels, out_channels)
        elif gnn_type == 'DGI_GCN'or gnn_type == 'DGI_GAT' or gnn_type=='DGI_SAGEConv':
            encoder = Encoder(in_channels, out_channels, gnn_type, dropout, head_num)
            summary_ = summary
            corruption_ = corruption
            self.dgi = DeepGraphInfomax(hidden_channels=out_channels,
                                        encoder=encoder,summary=summary_, 
                                        corruption=corruption_)
        else:
            raise ValueError(f'Unsupported model type:{gnn_type}')
        
    def forward(self, x, edge_index, edge_weight=None):
        x = x.to(torch.float32).to(next(self.parameters()).device)
        edge_index = edge_index.to(x.device)
        if edge_weight is not None:
            edge_weight = edge_weight.to(x.device)
        
        if self.gnn_type == 'GCN':
            x1 = self.conv(x, edge_index, edge_weight)
            x1 = self.norm(x1)
            x1 = F.leaky_relu(x1)
            x1 = F.dropout(x1, p=self.dropout, training=self.training)
            x = x+x1
            loss = torch.tensor(0.0)
        elif self.gnn_type == 'GATv2':
            x1 = self.conv(x, edge_index, edge_weight)
            x1 = self.norm(x1)
            x1 = F.leaky_relu(x1)
            x1 = F.dropout(x1, p=self.dropout, training=self.training)
            x = x+x1
            loss = torch.tensor(0.0)
        elif self.gnn_type == 'GAE':
            residual = self.residual(x) 
            x = self.conv.encoder(x, edge_index)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x += residual
            x = F.leaky_relu(x)
            x = self.norm(x)
            loss = self.conv.recon_loss(x, edge_index)
        elif self.gnn_type == 'GraphTransformer':
            residual = self.residual(x) 
            x = F.leaky_relu(self.conv(x, edge_index))
            x = self.mlp(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x += residual
            x = F.leaky_relu(x)
            x = self.norm(x)
            loss = torch.tensor(0.0)
        elif self.gnn_type.startswith('DGI'):
            x, neg_z, summary = self.dgi(x, edge_index, edge_weight)

            dgi_node_loss = self.dgi.loss(x, neg_z, summary)
            loss = dgi_node_loss
        else: 
            x1 = self.conv(x, edge_index, edge_weight)
            x1 = self.norm(x1)
            x1 = F.leaky_relu(x1)
            x1 = F.dropout(x1, p=self.dropout, training=self.training)
            x = x+x1
            loss = torch.tensor(0.0)
        return x, loss

def summary(x, *args, **kwargs):
    return torch.mean(x, dim=0)

def corruption(x, edge_index, edge_weight=None):
    noise = 0.3 * torch.randn_like(x)

    if edge_weight is not None:
        return x + noise, edge_index, edge_weight
    else:
        return x + noise, edge_index

class Encoder(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, gnn_type, dropout, head_num):
        super(Encoder, self).__init__()
        if gnn_type == 'DGI_GAT':
            self.conv = GATConv(num_features, hidden_channels//head_num, head_num)
        elif gnn_type == 'DGI_SAGEConv':
            self.conv = SAGEConv(num_features, hidden_channels)
        else:
            self.conv = GCNConv(num_features, hidden_channels)
        dim = hidden_channels//head_num*head_num
        self.residual = nn.Linear(num_features, hidden_channels)
        self.dropout = dropout
        self.norm = nn.LayerNorm(hidden_channels)

    def forward(self, x, edge_index, edge_weight=None):
        x = x.to(torch.float32).to(next(self.parameters()).device)
        edge_index = edge_index.to(x.device)
        if edge_weight is not None:
            edge_weight = edge_weight.to(x.device)
        x1 = self.conv(x, edge_index, edge_weight)
        x1 = self.norm(x1)
        x1 = F.leaky_relu(x1)
        x1 = F.dropout(x1, p=self.dropout, training=self.training)
        x = x+x1
        return x
    
