#!/bin/bash
#SBATCH --time=04:00:00
#SBATCH --qos=batch

source /nfshomes/sathvik/.bashrc
module add cuda
conda activate grnn-env
nvidia-smi
python3 cuda_check.py
python3 colorlessgreenRNNs/src/language_models/main.py --data /fs/clip-psych/sathvik/lm-syntactic-generalization/grnn_data --model LSTM --emsize 650 --nhid 650 --nlayers 2 --dropout 0.2 --lr 10 --batch_size 64 --cuda
