import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool, Set2Set, SetTransformerAggregation, TopKPooling
from .GNN import GNN
from pytorch_metric_learning import losses
from .AE import AE_net
from .MLP import MLP


class GONet(nn.Module):
    def __init__(self, args, ae_in_dim, AElayer_numer, gnn_out_dim, gnn_type, AE_module, head_num, GNN_module, omics_module, dropout, GNNlayer_numer, temperature, omics_head):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor([0.75]))
        self.gamma = nn.Parameter(torch.tensor([2.0]))
        self.args = args
        self.device = self.args.device
        self.num_omics = len(str(self.args.omics).split('_'))
        self.fusion = args.fusion
        
        self.ae_in_dim = ae_in_dim
        self.AElayer_numer = AElayer_numer
        self.gnn_out_dim = gnn_out_dim
        self.fusion_dim = gnn_out_dim 
        self.gnn_type = gnn_type
        self.head_num = head_num
        self.dropout = dropout
        self.temperature = temperature
        self.omics_head = omics_head
        self.GNNlayer_numer = GNNlayer_numer
               
        self.AE_module = AE_module
        self.GNN_module = GNN_module
        self.omics_module = omics_module
        self._initialize_models()

    def _initialize_models(self):
        if self.AE_module=='ae':
            self.AE = AE_net(self.args.seed, self.ae_in_dim, self.AElayer_numer)
        else:# self.AE_module == 'mlp':
            self.mlp_AE = MLP(self.args.seed, self.ae_in_dim, self.AElayer_numer)

        gnn_in_dim = self.gnn_out_dim
        self.embedding = nn.Sequential(
            nn.Linear(1+self.num_omics, gnn_in_dim),
            nn.LeakyReLU()
        )

        self.res_linear = nn.Linear(gnn_in_dim, self.gnn_out_dim)
        
        # ! a.组学内聚合层+跨组学交互层
        self.intra_layers = nn.ModuleList([
            GNN(self.args.seed, self.gnn_type, gnn_in_dim, self.gnn_out_dim, self.head_num, self.dropout)
            for _ in range(self.GNNlayer_numer)])
        self.cross_layers = nn.ModuleList([
            GNN(self.args.seed, self.gnn_type, gnn_in_dim, self.gnn_out_dim, self.head_num, self.dropout)
            for _ in range(self.GNNlayer_numer)])
        
        # ! b.不同组学之间的重要性
        self.omics_attn = nn.MultiheadAttention(self.gnn_out_dim, num_heads=self.omics_head, batch_first=True)
        
        # ! c.模块间融合
        ae_dim = self.ae_in_dim // (4**self.AElayer_numer)
        gnn_dim = self.gnn_out_dim*2
        omics_dim = self.gnn_out_dim
        self.global_proj = nn.Linear(ae_dim, self.fusion_dim)
        self.gnn_proj = nn.Linear(gnn_dim, self.fusion_dim)
        if self.omics_module == 'cat':
            self.omics_proj = nn.Linear(omics_dim, self.fusion_dim)
        else:
            self.omics_proj = nn.Linear(omics_dim, self.fusion_dim)
        self.layer_norm = nn.LayerNorm(self.fusion_dim)

        if self.omics_module=='cat' and self.AE_module:
            self.fusion_gate = nn.Sequential(
                nn.Linear((2+self.num_omics)*self.fusion_dim, self.fusion_dim//4),
                nn.LeakyReLU(),
                nn.Linear(self.fusion_dim//4, (2+self.num_omics)),
                nn.Softmax(dim=-1))
        elif self.omics_module=='att' and self.AE_module:
            self.fusion_gate = nn.Sequential(
                nn.Linear(3*self.fusion_dim, self.fusion_dim//4),
                nn.LeakyReLU(),
                nn.Linear(self.fusion_dim//4, 3),
                nn.Softmax(dim=-1))
        elif self.omics_module=='cat' and (not self.AE_module):
            self.fusion_gate = nn.Sequential(
                nn.Linear((1+self.num_omics)*self.fusion_dim, self.fusion_dim//4),
                nn.LeakyReLU(),
                nn.Linear(self.fusion_dim//4, (1+self.num_omics)),
                nn.Softmax(dim=-1))
        elif self.omics_module=='att' and (not self.AE_module):
            self.fusion_gate = nn.Sequential(
                nn.Linear(2*self.fusion_dim, self.fusion_dim//4),
                nn.LeakyReLU(),
                nn.Linear(self.fusion_dim//4, 2),
                nn.Softmax(dim=-1))
        else:
            self.fusion_gate = nn.Sequential(
                nn.Linear(2*self.fusion_dim, self.fusion_dim//4),
                nn.LeakyReLU(),
                nn.Linear(self.fusion_dim//4, 2),
                nn.Softmax(dim=-1))
        if self.fusion=='cat':
            self.classifier = nn.Sequential(
                nn.Linear(self.fusion_dim+ae_dim+gnn_dim, self.fusion_dim//4),
                nn.LeakyReLU(),
                nn.Linear(self.fusion_dim//4, self.args.num_classes))
        else:
            self.classifier = nn.Sequential(
                nn.Linear(self.fusion_dim, self.fusion_dim//4),
                nn.LeakyReLU(),
                nn.Linear(self.fusion_dim//4, self.args.num_classes))
        self.to(self.device)

    def forward(self, x, edge_index, intra_edge_index, cross_edge_index, y, ae_x, batch, omics_mapping, status, intra_edge_weight=None, cross_edge_weight=None):
        y = y.to(torch.int64)
        contra_loss = torch.tensor(0.0)
        # ! 1.global embedding learning in AE
        global_x, AE_recon_loss = self._process_ae(ae_x)
        # ! 2.初始编码：升维mlp【4-->16】
        x = self.embedding(x.float())
        # ! 3.GNN迭代消息传递
        gnn_x, dgi_loss = self._process_gnn(x, edge_index, intra_edge_index, cross_edge_index,intra_edge_weight, cross_edge_weight)
        mean_pool = global_mean_pool(gnn_x, batch)
        max_pool = global_max_pool(gnn_x, batch)
        gnn_pool = torch.cat([mean_pool, max_pool], dim=-1)

        # ! 4.组学特征学习
        omics_features, contra_loss, omics_stack, omics_weights = self._process_omics(gnn_x, batch, omics_mapping, y)
        
        omics_x = omics_features if self.omics_module == 'cat' else omics_stack
        if self.fusion=='cat':
            final_x = torch.cat([global_x, gnn_pool, omics_x], dim=-1)
            gate_weights = omics_weights
        else:
            final_x, gate_weights = self.fusion_cat(global_x, gnn_pool, omics_x)
            
        final_x = F.dropout(final_x, p=self.dropout, training=self.training)

        logits = self.classifier(final_x)
        class_scores = torch.softmax(logits, dim=1)
        cls_loss = self.Focal_loss(logits, y)

        return class_scores if status == 'explain' else (global_x, final_x, gnn_pool, omics_x, omics_weights, gate_weights, cls_loss, class_scores, AE_recon_loss, contra_loss, dgi_loss)
    
    def _process_omics(self, x, batch, omics_mapping, y):
        if self.omics_module == 'cat':
            omics_features = []
            batch_size = len(torch.unique(batch))
            trans_x = x.reshape(batch_size, -1, x.size(-1))
            for indices in omics_mapping.values():
                omics_x = trans_x[:, indices, :]
                pooled = global_mean_pool(omics_x, batch[indices])
                pooled = pooled.squeeze(dim=1)

                omics_features.append(pooled)
            omics_x = torch.stack(omics_features, dim=1)
            contra_loss = self.contrastive_loss(omics_x, y, self.temperature)
        elif self.omics_module == 'att':
            omics_features = []
            batch_size = len(torch.unique(batch))
            trans_x = x.reshape(batch_size, -1, x.size(-1))
            for indices in omics_mapping.values():
                omics_x = trans_x[:, indices, :]
                pooled = global_mean_pool(omics_x, batch[indices])
                omics_features.append(pooled.squeeze(dim=1))
            
            omics_stack = torch.stack(omics_features, dim=1)
            omics_x, omics_weights = self.omics_attn(
                omics_stack, omics_stack, omics_stack, average_attn_weights=True)
            
            contra_loss = self.contrastive_loss(omics_x, y, self.temperature)
            omics_x = omics_x.mean(dim=1)
            omics_weights = omics_weights.mean(dim=1)

        else:
            omics_features, contra_loss, omics_x, omics_weights = None, torch.tensor(0.0), None, None
        return omics_features, contra_loss, omics_x, omics_weights

    def fusion_cat(self, global_x, gnn_x, omics_features):
        if self.omics_module == 'cat' and self.AE_module:
            global_proj_x = self.global_proj(global_x)
            gnn_proj_x = self.gnn_proj(gnn_x)
            omics_proj_list = [self.omics_proj(omics_feature) for omics_feature in omics_features]
            final_x = torch.cat([global_proj_x, gnn_proj_x] + omics_proj_list, dim=-1)
            gate_weights = self.fusion_gate(final_x)

            final_x = gate_weights[:,0:1] * global_proj_x + gate_weights[:,1:2] * gnn_proj_x + sum(gate_weights[:, 2 + i:3 + i] * omics_proj_list[i] for i in range(self.num_omics))
        elif self.omics_module == 'att' and self.AE_module:
            global_proj_x = self.global_proj(global_x)
            gnn_proj_x = self.gnn_proj(gnn_x)
            omics_proj_x = self.omics_proj(omics_features)
            final_x = torch.cat([global_proj_x, gnn_proj_x, omics_proj_x], dim=-1)
            gate_weights = self.fusion_gate(final_x)
            final_x = gate_weights[:,0:1] * global_proj_x + gate_weights[:,1:2] * gnn_proj_x +gate_weights[:, 2:3] * omics_proj_x
        elif self.omics_module=='cat' and (not self.AE_module):
            gnn_proj_x = self.gnn_proj(gnn_x)
            omics_proj_list = [self.omics_proj(omics_feature) for omics_feature in omics_features]
            final_x = torch.cat([gnn_proj_x] + omics_proj_list, dim=-1)
            gate_weights = self.fusion_gate(final_x)
            final_x = gate_weights[:,0:1] * gnn_proj_x + sum(gate_weights[:, 1 + i:2 + i] * omics_proj_list[i] for i in range(self.num_omics))
        elif self.omics_module=='att' and (not self.AE_module):
            gnn_proj_x = self.gnn_proj(gnn_x)
            omics_proj_x = self.omics_proj(omics_features)
            final_x = torch.cat([gnn_proj_x, omics_proj_x], dim=-1)
            gate_weights = self.fusion_gate(final_x)
            final_x = gate_weights[:,0:1] * gnn_proj_x + gate_weights[:,1:2] * omics_proj_x
        else:
            global_proj_x = self.global_proj(global_x)
            gnn_proj_x = self.gnn_proj(gnn_x)
            final_x = torch.cat([global_proj_x, gnn_proj_x], dim=-1)
            gate_weights = self.fusion_gate(final_x)
            final_x = gate_weights[:,0:1] * global_proj_x + gate_weights[:,1:2] * gnn_proj_x
        return final_x, gate_weights

    def _process_gnn(self, x, edge_index, intra_edge_index, cross_edge_index, intra_edge_weight, cross_edge_weight):
        if self.GNN_module == 'intra':
            for intra in self.intra_layers:
                x_intra, dgi_loss = intra(x, intra_edge_index)
                x = F.relu(x_intra)
        elif self.GNN_module == 'cross':
            for cross in self.cross_layers:
                x_cross, dgi_loss = cross(x, cross_edge_index) 
                x = F.relu(x_cross)
        else:
            for gnn_layers in self.cross_layers:
                x_gnn, dgi_loss = gnn_layers(x, edge_index)
                x = F.relu(x_gnn)
        return x, dgi_loss

    def _process_ae(self, x):
        """Process data through AE if AE_module is enabled."""
        if self.AE_module == 'ae':
            global_feature, AE_recon_loss, _ = self.AE(x)
        elif self.AE_module == 'mlp':
            x = torch.tensor(x.values, dtype=torch.float32)
            global_feature = self.mlp_AE(x)
            AE_recon_loss = torch.tensor(0.0)
        else:
            global_feature, AE_recon_loss = None, torch.tensor(0.0)
        return global_feature, AE_recon_loss
    
    def Focal_loss(self, predictions, labels):

        alpha = torch.sigmoid(self.alpha)  # 约束到 (0,1) 范围
        gamma = F.softplus(self.gamma)

        CE_loss = F.cross_entropy(predictions, labels, reduction='none',  label_smoothing=0.1) 
        pt = torch.exp(-CE_loss)
        focal_loss = alpha * ((1-pt)**gamma) * CE_loss
        return focal_loss.mean()

    def contrastive_loss(self, omics_stack, y, temperature):
        if len(omics_stack.size()) == 3:
            batch_size, num_omics, feat_dim = omics_stack.shape
            omics_stack = omics_stack.reshape(batch_size*num_omics, feat_dim)
            labels = y.repeat_interleave(num_omics)
        else:
            batch_size, feat_dim = omics_stack.shape
            labels = y
        omics_flat = torch.nn.functional.normalize(omics_stack, p=2, dim=-1)
        criterion = losses.SupConLoss(temperature=self.temperature)
        loss = criterion(omics_flat, labels)

        return loss
