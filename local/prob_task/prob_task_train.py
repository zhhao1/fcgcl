#!/usr/bin/env python3
import argparse
import json
import time
import math
import numpy as np
import os,sys,io
import os.path as op
import configargparse
import logging 

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from bins.prob_task.transformer_st import E2E
from bins.prob_task.plDataModule_prob import data_prep
from bins.prob_task.plModule_prob import MyModule
from bins.prob_task.parameters import get_parser

from tools.dataload.functions import data_prep_func
from tools.function import log_and_dict_prep

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.base import Callback

from espnet.utils.io_utils import LoadInputsAndTargets
from espnet.nets.pytorch_backend.e2e_asr import pad_list

class BaseCallback(Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        trainer.datamodule.train_sampler.shuffle(trainer.current_epoch)

def train():
    #args prepare  
    parser = get_parser()
    E2E.add_arguments(parser)
    #pasrse = Trainer.add_argparse_args(parser) #new version can be used in this manner
    args = parser.parse_args()

    #seet seed  
    seed_everything(args.seed)

    #dict prep
    with open(args.dict, encoding = "utf-8") as label_file:
        labels = [line.strip() for line in label_file]
    odim = len(labels)
    label2id = dict([(labels[i].split()[0], i) for i in range(len(labels))])
    id2label = dict([(i, labels[i].split()[0]) for i in range(len(labels))])
    #data prep
    data = data_prep(args, label2id)   
    model = MyModule(83, 7981, odim, args)
    
    #callback and logger
    criterion = 'val_' + args.criterion + '_epoch'
    checkpoint_callback_top_k = ModelCheckpoint(monitor='val_acc_epoch', save_top_k=1, filename='{epoch:02d}-{val_acc_epoch:.4f}', mode='max')
    early_stop_callback = EarlyStopping(monitor='val_acc_epoch', min_delta=args.threshold, patience=args.patience, mode='max')   
    logger = TensorBoardLogger("logs", name=args.name)

    trainer = Trainer.from_argparse_args(args,
                                         logger=logger,                                         
                                         callbacks=[checkpoint_callback_top_k, BaseCallback(), early_stop_callback])
                         
    #logging.info(model)                
    trainer.fit(model,data)
    
def test(args):
    with open(args.dict, encoding = "utf-8") as label_file:
        labels = [line.strip() for line in label_file]
    odim = len(labels)
    label2id = dict([(labels[i].split()[0], i) for i in range(len(labels))])
    id2label = dict([(i, labels[i].split()[0]) for i in range(len(labels))])
    
    data = data_prep(args, label2id)
    
    test_path = args.test_model_path
    trainer = Trainer()
    model = MyModule.load_from_checkpoint(test_path, 83, 7981, odim)
    trainer.test(model, data)
                             
if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    if args.stage == 'train':
        train()
    else:
        test(args)
        
    
          
