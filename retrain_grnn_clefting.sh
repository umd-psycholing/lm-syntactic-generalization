#!/bin/bash
source /nfshomes/sathvik/.bashrc
module add cuda
conda activate grnn-env
nvidia-smi
python3 cuda_check.py
DATA_PATH=/fs/clip-psych/sathvik/lm-syntactic-generalization/grnn_data/clefting
MODEL_NAME=grnn_data/clefting/ouputs/model_clefting.pt
LOG_NAME=grnn_data/clefting/outputs/clefting_train_log.txt
python3 colorlessgreenRNNs/src/language_models/main.py --data $DATA_PATH --model LSTM --emsize 650 --nhid 650 --nlayers 2 --dropout 0.2 --lr 20 --batch_size 128 --cuda --save $MODEL_NAME --log $LOG_NAME
