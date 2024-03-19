#!/bin/bash
#SBATCH --time=04:00:00
#SBATCH --qos=batch

source /nfshomes/sathvik/.bashrc
module add cuda
conda activate grnn-env
nvidia-smi
python3 cuda_check.py
python3 /fs/clip-psych/sathvik/lm-syntactic-generalization/retrain_gpt2.py --vocab /fs/clip-psych/sathvik/lm-syntactic-generalization/grnn_data/vocab.txt --train /fs/clip-psych/sathvik/lm-syntactic-generalization/grnn_data/train.txt
