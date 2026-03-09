import sys
import os
import gc
import torch
current_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(os.path.dirname(os.path.abspath(current_dir)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import args
from GONet.dataset import load_data
from GONet._utils_ import seed_everything, set_environment
from GONet.trainer import train_on_full_training_set, Trainer_early

if __name__ == '__main__':
    gc.collect()
    torch.cuda.empty_cache()
    seed_everything(42)
    set_environment(42)

    ## Load paths
    workshop_path, omics_data_path, omics_data_test_path = args.workshop_path, args.omics_data_path, args.omics_data_test_path
    omics_graph_path = os.path.join(workshop_path, f'{args.omics}_rela_graph.pt')
    omics_label_path = os.path.join(workshop_path, f'{args.omics}_clinic.csv')
    omics_label_test_path = os.path.join(workshop_path, f'{args.omics}_clinic_test.csv')

    # Load and preprocess data
    train_data, train_graph_list, train_labels, test_data, test_graph_list, test_labels, omics_mapping = load_data(workshop_path, omics_data_path, omics_graph_path, omics_label_path, omics_data_test_path, omics_label_test_path, args.omics)

    # model training
    precision, recall, f1, accuracy, roc_auc, auprc, model_done, _, _, _, _, _, _, _ = train_on_full_training_set(args, omics_mapping, train_graph_list, test_graph_list)

    print(f'Test Set Evaluation: F1: {f1:.4f}, ROC AUC: {roc_auc:.4f}, auprc:{auprc:.4f}, Accuracy: {accuracy:.4f}')


