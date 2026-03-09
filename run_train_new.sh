

#!/bin/bash
whoami
date
env | grep CUDA
nvidia-smi

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
# LD_LIBRARY_PATH=/share/appspace_data/shared_groups/bgi_zhangzh_jszx/.conda/envs/py38/lib:$LD_LIBRARY_PATH
LD_LIBRARY_PATH=/share/org/BGI/bgi_zhangzh/.conda/envs/py38_pro/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH

conda activate /share/org/BGI/bgi_zhangzh/.conda/envs/py38_pro

cancer_type=$1

python3 main_train.py --cancer_type "$cancer_type"