#!/bin/bash
source /nfshomes/sathvik/.bashrc
module add cuda
conda activate grnn-env
nvidia-smi
python3 cuda_check.py
DATA_PATH=/fs/clip-psych/sathvik/lm-syntactic-generalization/grnn_data/intro_topic
MODEL_NAME=grnn_data/intro_topic/outputs/model_intro_topic.pt
LOG_NAME=grnn_data/intro_topic/outputs/intro_topic_train_log.txt
python3 colorlessgreenRNNs/src/language_models/main.py --data $DATA_PATH --model LSTM --emsize 650 --nhid 650 --nlayers 2 --dropout 0.2 --lr 20 --batch_size 128 --cuda --save $MODEL_NAME --log $LOG_NAME
