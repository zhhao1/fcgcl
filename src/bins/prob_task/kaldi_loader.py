import numpy as np
import os
import torch
import random
import math
from torch.distributed import get_rank
from torch.distributed import get_world_size
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from torch.utils.data.distributed import DistributedSampler

from bins.prob_task.kaldi_io import *
import torch.distributed as dist

def build_LFR_features(feats_path, m, n):
    """
    Actually, this implements stacking frames and skipping frames.
    if m = 1 and n = 1, just return the origin features.
    if m = 1 and n > 1, it works like skipping.
    if m > 1 and n = 1, it works like stacking but only support right frames.
    if m > 1 and n > 1, it works like LFR.

    Args:
        feats_path: path to read features in kaldi format
        m: number of frames to stack
        n: number of frames to skip
    """
    inputs = read_mat(feats_path)
    LFR_inputs = []
    T = inputs.shape[0]
    T_lfr = int(np.ceil(T / n))
    for i in range(T_lfr):
        if m <= T - i * n:
            LFR_inputs.append(np.hstack(inputs[i*n:i*n+m]))
        else: # process last LFR frame
            num_padding = m - (T - i * n)
            frame = np.hstack(inputs[i*n:])
            for _ in range(num_padding):
                frame = np.hstack((frame, inputs[-1]))
            LFR_inputs.append(frame)
    return np.vstack(LFR_inputs)

    
class KaldiDataset(Dataset):
    def __init__(self, feats_scp, info, dur_file, label2id, LFR_m=1, LFR_n=1, tasks='speaker'):
        super(KaldiDataset, self).__init__()

        self.LFR_m = LFR_m
        self.LFR_n = LFR_n
  
        utt2feats = {}
        with open(feats_scp) as f:
            for line in f:
                utt, feat = line.strip().split()
                utt2feats[utt] = feat

        if tasks == 'speaker':      
            utt2label = {}
            with open(info, encoding = "utf-8") as t:
                for line in t:
                    utt = line.strip().split(',')[0]
                    utt2label[utt] = self.parse_label(line.strip().split(',')[1], label2id)

        elif tasks == 'intention':
            utt2label = {}
            with open(info, encoding = "utf-8") as t:
                for line in t:
                    utt = line.strip().split(',')[0]
                    utt2label[utt] = self.parse_label(line.strip().split(',')[2], label2id)

        else:
            print("not defined prob task, check it")

        utt2dur = {}
        with open(dur_file) as d:
            for line in d:
                utt, dur = line.strip().split()
                utt2dur[utt] = int(dur)
        
        ids = []
        dur = []
        utts = []
        for utt in utt2feats:
            utts.append(utt)
            ids.append([utt2feats[utt], utt2label[utt], utt])
            dur.append(utt2dur[utt])
        
        self.ids = ids
        self.dur = dur
        self.utts = utts
        self.size = len(ids)
        
    @property
    def duration_lists(self) : return self.dur
    
    def __getitem__(self, index):
        sample = self.ids[index]
        feats_path, label, utt = sample[0], sample[1], sample[2]
        feats = self.parse_feats(feats_path)
        return feats, label, utt

    def __len__(self):
        return self.size
        
    def parse_feats(self, feats_path):
        spect = build_LFR_features(feats_path, self.LFR_m, self.LFR_n).T
        spect = torch.FloatTensor(spect)
        return spect

    def parse_label(self, label, label2id):
        ids = label2id.get(label)           
        return ids

def _collate_fn(batch):
    def func(p):
        return p[0].size(1)

    # descending sorted
    batch = sorted(batch, key=lambda sample: sample[0].size(1), reverse=True)
    max_seq_len = max(batch, key=func)[0].size(1)
    freq_size = max(batch, key=func)[0].size(0)

    input_sizes = torch.IntTensor(len(batch))
    inputs = torch.add(torch.zeros(len(batch), 1, freq_size, max_seq_len), -1)
   
    utts = []
    label = []
    for x in range(len(batch)):
        sample = batch[x]
        input_data = sample[0]
        label.append(sample[1])
        seq_length = input_data.size(1)
        input_sizes[x] = seq_length
        inputs[x][0].narrow(1, 0, seq_length).copy_(input_data)
        utts.append(sample[2])
            
    return inputs, input_sizes, label, utts

class KaldiDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super(KaldiDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn


class BucketingSampler(Sampler):
    def __init__(self, data_source, batch_size = 1, batch_frames = 0):
        """
        Samples batches assuming they are in order of size to batch similarly sized samples together.
        """
        super(BucketingSampler, self).__init__(data_source)
        self.data_source = data_source
        dur = data_source.duration_lists
        
        ids = np.argsort([d for d in dur])
        
        self.bins = []
        if batch_frames == 0 :
            start = 0
            while True :
                '''ilen, olen = dur[ids[start]]
                factor = max(int(ilen / src_max_len), int(olen / tgt_max_len))'''
                factor = 0
                b = max(1, int(batch_size / (1 + factor)))
                end = min(len(ids), start + b)
                self.bins.append(ids[start : end])
                
                if end == len(ids) :
                    break
                start = end
        else :
            start = 0
            while True:
                total_frames = 0
                end = start
                while total_frames < batch_frames and end < len(ids):
                    ilen = int(dur[ids[end]][0])
                    total_frames += ilen
                    end += 1

                self.bins.append(ids[start : end])
                if end == len(ids):
                    break
                start = end
        
        # ids = list(range(0, len(data_source)))
        #self.bins = [ids[i:i + batch_size] for i in range(0, len(ids), batch_size)]
        if constant.args.shuffle :
            seed = constant.args.seed
            random.seed(seed)
            np.random.seed(seed)
            np.random.shuffle(self.bins)

    def __iter__(self):
        for ids in self.bins:
            np.random.shuffle(ids)
            yield ids

    def __len__(self):
        return len(self.bins)

    def shuffle(self, epoch):
        np.random.shuffle(self.bins)

class BucketingDistributedSampler(Sampler):

    def __init__(self, dataset, batch_size = 1, batch_frames = 0, num_replicas=None,
                       rank=None, shuffle_batch=True, shuffle_epoch=True, seed=0):
        """
        Samples batches assuming they are in order of size to batch similarly sized samples together.
        """
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()                    
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0        
        self.shuffle_batch = shuffle_batch #shuffle data in each batch
        self.shuffle_epoch = shuffle_epoch #shuffle all batchs in epoch
        self.seed = seed        
        dur = dataset.duration_lists
        ids = np.argsort([d for d in dur])                    
        self.bins = []
        if batch_frames == 0 :
            start = 0
            while True :
                '''ilen, olen = dur[ids[start]]
                factor = max(int(ilen / src_max_len), int(olen / tgt_max_len))'''
                factor = 0
                b = max(1, int(batch_size / (1 + factor)))
                end = min(len(ids), start + b)
                self.bins.append(ids[start : end])                
                if end == len(ids) :
                    break
                start = end
        else :
            start = 0
            while True:
                total_frames = 0
                end = start
                while total_frames < batch_frames and end < len(ids):
                    ilen = int(dur[ids[end]][0])
                    total_frames += ilen
                    end += 1
                self.bins.append(ids[start : end])
                if end == len(ids):
                    break
                start = end

        seed = 1
        random.seed(seed)
        np.random.seed(seed)
        np.random.shuffle(self.bins)
                                              
    def __iter__(self):   
        for ids in self.bins:
            ids = ids.tolist()
            num_samples = int(math.ceil(len(ids) * 1.0 / self.num_replicas))
            total_size = num_samples * self.num_replicas
            ids.extend(ids[:(total_size - len(ids))])
            indices = ids[self.rank:total_size:self.num_replicas]
            #indices = ids[self.rank*num_samples:(self.rank+1)*num_samples]
            if self.shuffle_batch:
            # deterministically shuffle based on epoch and seed
                np.random.shuffle(indices)
            yield indices    

    def __len__(self):
        return len(self.bins)

    def shuffle(self, epoch):
        np.random.shuffle(self.bins)
