#!/bin/bash
#SBATCH --time=04:00:00
#SBATCH --qos=batch

source /nfshomes/sathvik/.bashrc
module add cuda
conda activate grnn-env
python /fs/clip-psych/sathvik/lm-syntactic-generalization/perplexity.py --cuda
