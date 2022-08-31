import pytorch_lightning as pl
import os.path as op
from bins.prob_task.kaldi_loader import KaldiDataLoader, KaldiDataset, BucketingSampler, BucketingDistributedSampler

class data_prep(pl.LightningDataModule):
    def __init__(self, args, label2id):
        super().__init__()        
        self.args = args
          
        self.train_data = KaldiDataset(feats_scp = op.join(self.args.train_data, "feats.scp"), 
                                  info = op.join(self.args.train_data, "train"), 
                                  dur_file = op.join(self.args.train_data, "utt2num_frames"),
                                  label2id = label2id,
                                  tasks = args.tasks)

        self.valid_data = KaldiDataset(feats_scp = op.join(self.args.valid_data, "feats.scp"), 
                                  info = op.join(self.args.valid_data, "valid"), 
                                  dur_file = op.join(self.args.valid_data, "utt2num_frames"),
                                  label2id = label2id,
                                  tasks = args.tasks)

        self.test_data = KaldiDataset(feats_scp = op.join(self.args.test_data, "feats.scp"), 
                                  info = op.join(self.args.test_data, "test"), 
                                  dur_file = op.join(self.args.test_data, "utt2num_frames"),
                                  label2id = label2id,
                                  tasks = args.tasks)

        if self.args.accelerator == 'ddp':
            self.sampler = BucketingDistributedSampler
        else:
            self.sampler = BucketingSampler
                                                              
    def train_dataloader(self):

        self.train_sampler = self.sampler(self.train_data, 
                                          batch_size = self.args.batch_size,
                                          batch_frames = self.args.batch_frames)
        train_loader = KaldiDataLoader(self.train_data, 
                                       num_workers = self.args.num_workers, 
                                       batch_sampler = self.train_sampler,
                                       pin_memory = self.args.pin_memory)
        return train_loader
     
    def val_dataloader(self):
    
        self.valid_sampler = self.sampler(self.valid_data, 
                                          batch_size = self.args.batch_size,
                                          batch_frames = self.args.batch_frames)                                         
        valid_loader = KaldiDataLoader(self.valid_data, 
                                       num_workers = self.args.num_workers, 
                                       batch_sampler = self.valid_sampler,
                                       pin_memory = self.args.pin_memory)     
        return valid_loader
        
    def test_dataloader(self):
    
        self.test_sampler = self.sampler(self.test_data, 
                                          batch_size = self.args.batch_size,
                                          batch_frames = self.args.batch_frames)                                         
        test_loader = KaldiDataLoader(self.test_data, 
                                       num_workers = self.args.num_workers, 
                                       batch_sampler = self.test_sampler,
                                       pin_memory = self.args.pin_memory)     
        return test_loader
