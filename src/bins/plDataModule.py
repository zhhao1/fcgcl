import pytorch_lightning as pl
import os.path as op
from tools.dataload.kaldi_loader import KaldiDataLoader, KaldiDataset, BucketingSampler, BucketingDistributedSampler

class data_prep(pl.LightningDataModule):
    def __init__(self, args, train_json, valid_json, transform_tr, transform_cv):
        super().__init__()        
        self.args = args
            
        self.train_data = KaldiDataset(args, train_json)
        self.valid_data = KaldiDataset(args, valid_json)
        self.transform_tr = transform_tr
        self.transform_cv = transform_cv

        if args.accelerator == 'ddp':
            self.sampler = BucketingDistributedSampler
        else :
            self.sampler = BucketingSampler
                                                              
    def train_dataloader(self):

        self.train_sampler = self.sampler(self.train_data)
                                          
        train_loader = KaldiDataLoader(self.transform_tr,
                                       self.train_data, 
                                       batch_sampler = self.train_sampler,
                                       num_workers = self.args.num_workers,
                                       pin_memory = self.args.pin_memory)
        return train_loader
     
    def val_dataloader(self):
    
        self.valid_sampler = self.sampler(self.valid_data)
                                                
        valid_loader = KaldiDataLoader(self.transform_cv,
                                       self.valid_data,
                                       batch_sampler = self.valid_sampler,
                                       num_workers = self.args.num_workers,
                                       pin_memory = self.args.pin_memory)     
        return valid_loader
