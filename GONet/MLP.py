#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/1/30 17:23
# @Author  : Eayon
# @Site    : 
# @File    : MLP.py
# @Software: PyCharm
# @logit:

import torch
import torch.nn.init as init
import torch.nn as nn
from ._utils_ import seed_everything


class MLP(nn.Module):
    def __init__(self, args_seed, input_dim, num_layers):
        super().__init__()
        seed_everything(args_seed)
        hidden_dims = [input_dim//(4**(i+1)) for i in range(num_layers)]

        mlp_layer = []
        mlp_layer.append(nn.Linear(input_dim, hidden_dims[0]))
        mlp_layer.append(nn.LayerNorm(hidden_dims[0]))
        mlp_layer.append(nn.LeakyReLU())
        mlp_layer.append(nn.Dropout(0.2))
        for i in range(len(hidden_dims) - 1):
            mlp_layer.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            mlp_layer.append(nn.LayerNorm(hidden_dims[i+1]))
            mlp_layer.append(nn.LeakyReLU())
            mlp_layer.append(nn.Dropout(0.2))
        self.mlp = nn.Sequential(*mlp_layer)

    def forward(self, x):
        if not torch.is_tensor(x):
            x = torch.tensor(x.values, dtype=torch.float32)
        x = x.to(next(self.mlp.parameters()).device)
        x = self.mlp(x)
        return x
