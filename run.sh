#!/bin/bash
#SBATCH --job-name=w2v_cl
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2
#SBATCH --mem=0
#SBATCH --partition=gpu_v100
source activate espnet

export NCCL_DEBUG=WARN
export PYTHONFAULTHANDLER=1
. ./path.sh 

train_config=conf/train.yaml
data_name=MUST-C/en-de
dict=../lang_1spm/train_sp.en-de.de_bpe8000_units_tc.txt
train_json=../data/${data_name}/wav2vec/data/train/data_bpe8000.lc.rm_tc.json
valid_json=../data/${data_name}/wav2vec/data/dev/data_bpe8000.lc.rm_tc.json

srun python local/st_train.py --config ${train_config} \
                              --num-nodes 1 \
                              --gpus 1 \
                              --dict ${dict} \
                              --seed 42 \
                              --verbose 1 \
                              --train-json ${train_json} \
                              --valid-json ${valid_json} \
                              --accelerator ddp \
                              --num-workers 4
