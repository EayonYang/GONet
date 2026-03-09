import os
import torch

cancer_type='HTML_THCA'
omics='mRNA_miRNA'
workshop_path=f'./data/{cancer_type}'
result_path='./results'
omics_data_path = os.path.join(workshop_path, f'{omics}_df.csv')
omics_data_test_path = os.path.join(workshop_path, f'{omics}_df_test.csv')
num_classes=2

AElayer_numer = 2
GNNlayer_numer = 2
gnn_out_dim =  64
gnn_type = 'DGI_GCN'
head_num =  4
temperature = 0.7
omics_head = 4
dropout = 0.2
weight_decay = 0.0005
lr = 0.001
num_epoches = 1#200
cls_w = 4.0
ae_w = 1.5
gnn_w = 2.0
cont_w = 0.20
AE_module = 'ae'
GNN_module='all'
omics_module='att'
fusion='all'
seed=42
batch_size=10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')