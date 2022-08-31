import numpy as np
import os
import torch
import random
import math

import torch.distributed as dist
from torch.distributed import get_rank
from torch.distributed import get_world_size
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler

from espnet.utils.training.batchfy import make_batchset
from espnet.utils.io_utils import LoadInputsAndTargets
    
class KaldiDataset(Dataset):
    def __init__(self, args, data_json):
        super(KaldiDataset, self).__init__()
        
        self.use_sortagrad = args.sortagrad == -1 or args.sortagrad > 0

        self.data = make_batchset(
                                  data_json,
                                  args.batch_size,
                                  args.maxlen_in,
                                  args.maxlen_out,
                                  args.minibatches, #debug
                                  min_batch_size=args.gpus if args.gpus > 1 else 1,
                                  shortest_first=self.use_sortagrad, #-1 or 0, namely true or false ,this is for all epoch, data in batch will be sort in transform
                                  count=args.batch_count,
                                  batch_bins=args.batch_bins,
                                  batch_frames_in=args.batch_frames_in,
                                  batch_frames_out=args.batch_frames_out,
                                  batch_frames_inout=args.batch_frames_inout,
                                  iaxis=0,
                                  oaxis=0
                                           )
    
    def __getitem__(self, index):
        return index

    def __len__(self):
        return len(self.data)
   
class KaldiDataLoader(DataLoader):
    def __init__(self, transform, *args, **kwargs):
        super(KaldiDataLoader, self).__init__(*args, **kwargs)
        self.transform = transform #will sort data in minibatch according the input lengths
        self.collate_fn = self._collate_fn
        
    def _collate_fn(self, batch):
        return self.transform(batch)


class BucketingSampler(Sampler):
    def __init__(self, dataset):
        """
        Samples batches assuming they are in order of size to batch similarly sized samples together.
        """
        super(BucketingSampler, self).__init__(dataset)
        self.data = dataset.data
        self.use_sortagrad = dataset.use_sortagrad
        
        if not self.use_sortagrad:
            self.sampler = RandomSampler(self.data)
        else:
            self.sampler = SequentialSampler(self.data)
            
    def __iter__(self):
        for batch_index in self.sampler:
            batch = self.data[batch_index]
            yield batch

    def __len__(self):
        return len(self.data)

        
class BucketingDistributedSampler(Sampler):

    def __init__(self, dataset, num_replicas=None, rank=None):
        """
        Samples batches assuming they are in order of size to batch similarly sized samples together.
        """
        super(BucketingDistributedSampler, self).__init__(dataset)
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()                    
        self.data = dataset.data
        self.use_sortagrad = dataset.use_sortagrad
        self.num_replicas = num_replicas
        self.rank = rank

        if not self.use_sortagrad:
            self.sampler = RandomSampler(self.data)
        else:
            self.sampler = SequentialSampler(self.data)
                            
    def __iter__(self):
        for batch_index in self.sampler:
            batch = self.data[batch_index]
            num_samples = int(math.ceil(len(batch) * 1.0 / self.num_replicas))
            total_size = num_samples * self.num_replicas
            batch.extend(batch[:(total_size - len(batch))])
            batch_one_gpu = batch[self.rank:total_size:self.num_replicas]
            yield batch_one_gpu                   

    def __len__(self):
        return len(self.data)
