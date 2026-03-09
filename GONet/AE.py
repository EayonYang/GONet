#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/1/30 10:18
# @Author  : Eayon
# @Site    : 
# @File    : AE.py
# @Software: PyCharm
# @logit:

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm
from ._utils_ import seed_everything, min_max_normalization

class AE_net(nn.Module):
    def __init__(self, args_seed, input_dim: int, num_layers: int):
        """
        :param input_dim:feature dimension of the input patient data
        :param hidden_dims: a list of hidden dimensions
        """
        super().__init__()
        seed_everything(args_seed)

        hidden_dims = [input_dim // (4**(i+1)) for i in range(num_layers)]

        # Encoder
        encoder_layers = []
        encoder_layers.append(nn.Linear(input_dim, hidden_dims[0]))
        encoder_layers.append(nn.LayerNorm(hidden_dims[0]))
        encoder_layers.append(nn.LeakyReLU())
        encoder_layers.append(nn.Dropout(0.2))
        for i in range(len(hidden_dims) - 1):
            encoder_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            encoder_layers.append(nn.LayerNorm(hidden_dims[i+1]))
            encoder_layers.append(nn.LeakyReLU())
            encoder_layers.append(nn.Dropout(0.2))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = []
        for i in range(len(hidden_dims) - 1, 0, -1):   # reserve order
            decoder_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i - 1]))
            decoder_layers.append(nn.LayerNorm(hidden_dims[i-1]))
            decoder_layers.append(nn.LeakyReLU())
            decoder_layers.append(nn.Dropout(0.2))
        decoder_layers.append(nn.Linear(hidden_dims[0], input_dim))
        decoder_layers.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        if not torch.is_tensor(x):
            x = torch.tensor(x.values, dtype=torch.float32)  # converts df to array--tensor
        x = x.to(next(self.encoder.parameters()).device).to(torch.float32)
        x = torch.reshape(x, (x.shape[0], -1))
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        AE_recon_loss = F.mse_loss(decoded, x, reduction='mean')
        # kl_loss = -0.5 * torch.sum(1 + torch.log(1e-8 + encoded.var()) - encoded.mean().pow(2) - encoded.var())
        
        AE_loss = AE_recon_loss #+ kl_loss
        x = encoded
        return x, AE_loss, decoded
