# FCGCL: Fine- and Coarse-Granularity Contrastive Learning for Speech Translation

This is the pytorch implementation for paper "FCCL: Fine- and Coarse-Granularity Contrastive Learning for Speech Translation".

# Enviroment Configuration

Our code is based on [Espnet](https://github.com/espnet/espnet) and use [PyTorch-Lightning](https://github.com/Lightning-AI/lightning) to organize our code. Please install Espnet and PyTorch-Lightning following the official guidance. 

# Data Preparation
1. Download the [wav2vec 2.0](https://huggingface.co/facebook/wav2vec2-large-960h/tree/main) model published in Huggingface.
2. We extract feature bases on wav2vec 2.0 before training. The scripts are saved on ./scripts/.
3. Save to json file. This is consistent with Espnet. We upload the dev_json and the corresponding feature for reference to quickly debug the code. 

# Model Training

. ./run.sh

The training process in defined on ./src/bins/plModule.py. The contrastive module is defined on ./src/bins/cl_loss.py. 

