# GONet

GONet establishes a robust and generalizable graph representation learning for integrating multi-omics data (mRNA, miRNA, and proteomics) and interaction networks.

# Requirements
Python >= 3.8

PyTorch >= 2.0

PyTorch Geometric >= 2.4.0

Scikit-learn, Pandas, Numpy, Matplotlib


# Installation
git clone https://github.com/yourusername/GONet.git
cd GONet
conda env create -f environment.yml
conda activate gonet

# Data Input
GONet expects input data in CSV format. Each omic layer should be a matrix of (samples × features).
We provide a processed data in the data/ directory, and place your raw data in same path if you want to test your data.

# Usage
GONet provides a streamlined pipeline from hyperparameter optimization to final model training.
### 1. Configuration
All model parameters and experiment settings are managed in args.py. Before running, ensure you have configured the paths and basic settings (e.g., learning rate, weight decay, or GNN layers) in this file.


### 2. Hyperparameter Optimization (Optional)
To find the optimal hyperparameters for a specific cancer dataset, we utilize Optuna. This step is recommended for achieving peak performance on new data.


```python
python3 main_optuna_model.py
```

The search space and number of trials can be adjusted within main_optuna_model.py.

### 3. Model usage
You can train GONet using either the Python script or the provided shell script for batch processing.

Option A: Direct usage
Specify the target cancer type using the --cancer_type argument:


```python
python3 main_train.py --cancer_type HTML_THCA
```

Option B: Batch/Shell usage
Use the shell script to execute training with predefined environment settings (recommended for server-side execution):

```Bash 
run_train_new.sh HTML_THCA
```

The model outputs performance metrics including AUC-ROC, F1-score, and Accuracy.
