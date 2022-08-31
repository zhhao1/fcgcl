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

from bins.transformer_st import E2E
from bins.plDataModule import data_prep
from bins.plModule import MyModule
from bins.parameters import get_parser

from tools.dataload.functions import data_prep_func
from tools.function import log_and_dict_prep

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.base import Callback

from espnet.utils.io_utils import LoadInputsAndTargets
from espnet.nets.pytorch_backend.e2e_asr import pad_list
            
def main(cmd_args):
    #args prepare  
    parser = get_parser()
    E2E.add_arguments(parser)
    #pasrse = Trainer.add_argparse_args(parser) #new version can be used in this manner
    args = parser.parse_args()
    
    #seet seed  
    seed_everything(args.seed)

    ##logging and dict prepare   
    idim, odim = log_and_dict_prep(args)
        
    #dataloader
    train_json, valid_json, transform_tr, transform_cv = data_prep_func(args)  
    data = data_prep(args, train_json, valid_json, transform_tr, transform_cv)
    #callback and logger
    logger = TensorBoardLogger("logs", name=args.name)
    if not args.pretrain:
        criterion = 'val_acc_epoch'
        checkpoint_callback_top_k = ModelCheckpoint(monitor=criterion, save_top_k=5, filename='{epoch:02d}-{val_acc_epoch:.4f}', mode='max')
        early_stop_callback = EarlyStopping(monitor=criterion, min_delta=args.threshold, patience=args.patience, mode='max')
        trainer = Trainer.from_argparse_args(args,
                                         logger=logger,
                                         callbacks=[checkpoint_callback_top_k, early_stop_callback])  
    elif args.moco:  # without valid
        criterion = 'loss_st'
        checkpoint_callback_top_k = ModelCheckpoint(monitor=criterion, save_top_k=5, filename='{epoch:02d}-{loss_st_epoch:.4f}', mode='min')
        trainer = Trainer.from_argparse_args(args,
                                         logger=logger,
                                         callbacks=[checkpoint_callback_top_k]) 
    else:
        criterion = 'val_loss_epoch'                               
        checkpoint_callback_top_k = ModelCheckpoint(monitor=criterion, save_top_k=5, filename='{epoch:02d}-{val_loss_epoch:.4f}', mode='min')
        early_stop_callback = EarlyStopping(monitor=criterion, min_delta=args.threshold, patience=args.patience, mode='min')
    #checkpoint_callback_best = ModelCheckpoint(monitor=criterion, filename='best-{epoch:02d}-{val_acc_epoch:.2f}', mode='max')  

        trainer = Trainer.from_argparse_args(args,
                                         logger=logger,
                                         callbacks=[checkpoint_callback_top_k, early_stop_callback])                    
    model = MyModule(idim, odim, args)
    #logging.info(model)                
    trainer.fit(model,data)
                             
if __name__ == "__main__":
    main(sys.argv[1:])
          
