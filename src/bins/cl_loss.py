import numpy as np
import logging
import math
import torch
import torch.nn as nn
import torch.distributed as dist
from torch import Tensor
from torch.nn import functional as F
from tools.function import remove_pad_and_mean, compute_kernel_bias, transform_and_normalize, normalize 

class contrastive(nn.Module):
    def __init__(self, args):
        super().__init__()
        input_dim = args.adim
        output_dim = args.output_dim
        hidden_dim = args.hidden_dim
        mlp_layers = args.mlp_layers
        self.args = args
        self.d_k = args.adim
        self.fine_linear_q = nn.Linear(args.adim, args.adim)
        self.fine_linear_k = nn.Linear(args.adim, args.adim)
        all_text_repre = torch.load('1')
        to_compute = np.array(all_text_repre.cpu())
        self.kernel, self.bias = compute_kernel_bias(to_compute)
        #self.STM = nn.Linear(args.adim*2, 2)
        if args.projection_type == 'linear':
            if args.share:
                self.projection = nn.Linear(input_dim, output_dim)
            else:
                self.projection_1 = nn.Linear(input_dim, output_dim)
                self.projection_2 = nn.Linear(input_dim, output_dim)
                self.projection_3 = nn.Linear(input_dim, output_dim)
        elif args.projection_type == 'non_linear':
            if args.share:
                layers = list()
                layers.append(nn.Linear(input_dim, hidden_dim))
                if args.norm:
                    layers.append(nn.LayerNorm(hidden_dim))
                layers.append(nn.ReLU())
                for i in range(mlp_layers - 2):
                    layers.append(nn.Linear(hidden_dim, hidden_dim))
                    if args.norm:
                        layers.append(nn.LayerNorm(hidden_dim))
                    layers.append(nn.ReLU())
                    layers.append(nn.Dropout(args.mlp_droprate))
                layers.append(nn.Linear(hidden_dim, output_dim, bias=False))
                self.projection = nn.Sequential(*layers)
            else:
                layers = list()              
                layers.append(nn.Linear(input_dim, hidden_dim))
                if args.norm:
                    layers.append(nn.LayerNorm(hidden_dim))
                layers.append(nn.ReLU())
                for i in range(mlp_layers - 2):
                    layers.append(nn.Linear(hidden_dim, hidden_dim))
                    if args.norm:
                        layers.append(nn.LayerNorm(hidden_dim))
                    layers.append(nn.ReLU())
                    layers.append(nn.Dropout(args.mlp_droprate))
                layers.append(nn.Linear(hidden_dim, output_dim, bias=False))
                self.projection_1 = nn.Sequential(*layers)
                
                layers = list()            
                layers.append(nn.Linear(input_dim, hidden_dim))
                if args.norm:
                    layers.append(nn.LayerNorm(hidden_dim))
                layers.append(nn.ReLU())
                for i in range(mlp_layers - 2):
                    layers.append(nn.Linear(hidden_dim, hidden_dim))
                    if args.norm:
                        layers.append(nn.LayerNorm(hidden_dim))
                    layers.append(nn.ReLU())
                    layers.append(nn.Dropout(args.mlp_droprate))
                layers.append(nn.Linear(hidden_dim, output_dim, bias=False))
                self.projection_2 = nn.Sequential(*layers)
                                 
                layers = list()            
                layers.append(nn.Linear(input_dim, hidden_dim))
                if args.norm:
                    layers.append(nn.LayerNorm(hidden_dim))
                layers.append(nn.ReLU())
                for i in range(mlp_layers - 2):
                    layers.append(nn.Linear(hidden_dim, hidden_dim))
                    if args.norm:
                        layers.append(nn.LayerNorm(hidden_dim))
                    layers.append(nn.ReLU())
                    layers.append(nn.Dropout(args.mlp_droprate))
                layers.append(nn.Linear(hidden_dim, output_dim, bias=False))
                self.projection_3 = nn.Sequential(*layers)
        else:
            raise Exception('Error define projection') 
                
            self.barlow = nn.BatchNorm1d(512, affine=False) #for barlow cl loss
            
        self.queue = []
        self.batch_store_length = self.args.num_negatives
        self.batch_store = torch.randn(1, output_dim)
        self.batch_store = F.normalize(self.batch_store, dim=1)
                                        
    def forward(self, speech_hs_pad, text_src_hs_pad, text_tgt_hs_pad, speech_hs_mask, text_src_mask, text_tgt_mask, args, text_src_hs_first, speech_src_hs_first, fine=True, whiteing=True, whiteing_speech=False, average=False):                                                   
        if fine:
            speech_hs_pad_fine = self.pad_t(speech_hs_pad) # B*T_max*dmodel
            text_src_hs_pad_fine = self.pad_t(text_src_hs_pad) # B*T_max*dmodel
            speech_hs_mask_fine = self.pad_t(speech_hs_mask, is_mask=True) # B*T_max
            text_src_mask_fine = self.pad_t(text_src_mask, is_mask=True)  # B*T_max
        if args.remove_pad:
            speech_hs_pad = remove_pad_and_mean(speech_hs_pad, speech_hs_mask)
            text_src_hs_pad = remove_pad_and_mean(text_src_hs_pad, text_src_mask)
        else:
            speech_hs_pad = torch.mean(speech_hs_pad, dim=1)
            text_src_hs_pad = torch.mean(text_src_hs_pad, dim=1)
        if average:
            text_src_hs_pad = (text_src_hs_pad + text_src_hs_first) * 0.5 #first and last mean
            #speech_hs_pad = (speech_hs_pad + speech_src_hs_first) * 0.5 #first and last mean
        #self._enqueue_and_dequeue(text_src_hs_pad)
        #whiteing
        if whiteing:
            device = speech_hs_pad.device
            text_src_hs_pad, batch_store = self.whiteing(text_src_hs_pad, device, "single_gpu")        
            if whiteing_speech:
                to_compute = speech_hs_pad.cpu().detach().numpy()
                kernel, bias = compute_kernel_bias(to_compute)
                kernel = torch.from_numpy(kernel).to(device).to(torch.float32)
                bias = torch.from_numpy(bias).to(device).to(torch.float32)
                speech_hs_pad = torch.mm(speech_hs_pad + bias, kernel)                 
        else:
            batch_store = self.batch_store        
        if fine: 
            loss_fine = self.nt_xent_loss_token_fine_max_mean(speech_hs_pad, text_src_hs_pad,
                                                              speech_hs_pad_fine, speech_hs_mask_fine, text_src_hs_pad_fine, text_src_mask_fine, 
                                                              batch_store, len(self.queue)) 
                                                              
        loss_speech_src = self.nt_xent_loss(speech_hs_pad, text_src_hs_pad, batch_store, len(self.queue))
        self._enqueue_and_dequeue(text_src_hs_pad)                
        if fine:
            return loss_speech_src, loss_fine
        else:
            return loss_speech_src, 0.0   
    def whiteing(self, to_white, device, choice='single_gpu'):
        #to_white = np.array(to_white.cpu())
        #to_white = transform_and_normalize(to_white, self.kernel, self.bias)
        #to_white = torch.from_numpy(to_white).to(device).to(torch.float32)
        #batch_store = self.batch_store
        #return to_white, batch_store 
        
        if choice == 'store_batch':
            to_compute = SyncFunction.apply(to_white)
            to_compute = torch.cat([to_compute, self.batch_store], dim=0)
            to_compute = np.array(to_compute.cpu())
            kernel, bias = compute_kernel_bias(to_compute)
            kernel = torch.from_numpy(kernel).to(device).to(torch.float32)
            bias = torch.from_numpy(bias).to(device).to(torch.float32)
            to_white = torch.mm(to_white + bias, kernel)
            batch_store = torch.mm(self.batch_store + bias, kernel) 
            return to_white, batch_store             
        else:                   
            if choice == 'single_gpu':
                to_compute = np.array(to_white.cpu())
                kernel, bias = compute_kernel_bias(to_compute)
            elif choice == 'all_gpu':             
                to_compute = np.array(SyncFunction.apply(to_white).cpu())                   
                kernel, bias = compute_kernel_bias(to_compute)               
            elif choice == 'all_batch': 
                kernel, bias = self.kernel, self.bias #pre calculate kernel and bias on all data
            to_white = np.array(to_white.cpu())
            to_white = transform_and_normalize(to_white, kernel, bias) #not normalize            
            to_white = torch.from_numpy(to_white).to(device).to(torch.float32)
            batch_store = self.batch_store
            return to_white, batch_store       
        
    @torch.no_grad()
    def _enqueue_and_dequeue(self, batch_last):
        batch_last = SyncFunction.apply(batch_last)
        #batch_last = F.normalize(batch_last, dim=1)
        if len(self.queue) == self.batch_store_length:
            self.queue = self.queue[1:]
            self.queue.append(batch_last) #[n*B1*d, n*B2*d, ...]
        else:
            self.queue.append(batch_last) #[n*B1*d, n*B2*d, ...]
        self.batch_store = torch.cat(self.queue, 0)

    def pad_t(self, src_input, is_mask=False):
        if is_mask:
            src_input = src_input.squeeze(1)
        device = src_input.device
        batch_size = src_input.size(0)
        length = src_input.size(1)
        length_tensor = torch.tensor(length).to(device)
        if not is_mask:
            dmodel = src_input.size(2)               
        output = [torch.zeros_like(length_tensor) for i in range(dist.get_world_size())]
        dist.all_gather(output, length_tensor)
        max_length = torch.stack(output).max()
        to_pad = max_length - length
        if not is_mask:
            pad_t = torch.zeros((batch_size, to_pad, dmodel)).to(device)
        else:
            pad_t = torch.zeros((batch_size, to_pad)).to(device)
        padded_input = torch.cat((src_input, pad_t), dim=1)        
        return padded_input  
             
    def nt_xent_loss(self, out_1, out_2, batch_store, batch_store_number, eps=1e-6):
        # out_1: speech  out_1: text
        out_1 = F.normalize(out_1, dim=-1)
        out_2 = F.normalize(out_2, dim=-1)
        batch_store = F.normalize(batch_store, dim=-1)
        temperature = self.args.temperature
        out_2_dist = SyncFunction.apply(out_2)       
        if batch_store_number == self.batch_store_length:
            out_2_dist = torch.cat([out_2_dist, batch_store], 0)             
        cov = torch.mm(out_1, out_2_dist.t().contiguous())
        sim = torch.exp(cov / temperature)
        neg = sim.sum(dim=-1)        
        pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
        loss = -torch.log(pos / (neg + eps)).mean()
               
        return loss
    
    def nt_xent_loss_token_fine_max_mean(self, out_1, out_2, out_1_fine, out_1_mask, out_2_fine, out_2_mask, batch_store, batch_store_number, choice='before', eps=1e-6):
        torch.set_deterministic(False)
        # out: B*T*d      out_mask: B*1*T
        # pos: max sim, 
        # neg: other sentence mean value
        temperature = self.args.temperature
        out_1_mask = out_1_mask.squeeze(1) # B*T1
        out_2_mask = out_2_mask.squeeze(1) # B*T2
        #batch_store = F.normalize(batch_store, dim=-1)
        
        out_1 = F.normalize(out_1, dim=-1) # B*d
        out_2 = F.normalize(out_2, dim=-1) # B*d
        out_2_dist = SyncFunction.apply(out_2) # (n*B)*d
        if batch_store_number == self.batch_store_length:
            out_2_dist = torch.cat([out_2_dist, batch_store, 0) #optional,may out of memory       
        out_1_fine = F.normalize(out_1_fine, dim=-1) # B*T1*d 
        out_2_fine = F.normalize(out_2_fine, dim=-1) # B*T2*d

        matrix = torch.matmul(out_1_fine, out_2_fine.transpose(-1, -2)) #B*T1*T2
        matrix = matrix.masked_fill((out_2_mask==0).unsqueeze(1), -1e9) # mask pad in T2 with -1e9
        pos, _ = matrix.max(2) # B*T1
        pos = pos[out_1_mask==1] # new_length, pad in T1 is removed
        pos = torch.exp(pos / temperature) 
        
        neg_pad_pos_mean = torch.matmul(out_1_fine, out_2_dist.transpose(-1, -2)) # B*T1*(n*B)
        neg_pad_pos_mean = torch.exp(neg_pad_pos_mean / temperature) # B*T1*(n*B)
        neg_pad_pos_mean = neg_pad_pos_mean.sum(dim=-1) # B*T1
        pos_mean = torch.matmul(out_1_fine, out_2.unsqueeze(-1)).squeeze(-1) # B*T1,
        pos_mean = torch.exp(pos_mean / temperature) # B*T1
        #logging.info(neg_pad_pos_mean.size())
        #logging.info(pos_mean.size())
        neg_mean = neg_pad_pos_mean - pos_mean # B*T1 
        neg_mean = neg_mean[out_1_mask==1] #new_length
        
        loss = -torch.log(pos / (neg_mean + pos + eps)).mean()       
        return loss
                        
    def contrastive_loss(self, hs_1, hs_2):
        tau_plus, beta, lambd, estimator = self.args.tau_plus, self.args.beta, self.args.lambd, self.args.estimator
        temperature = self.args.temperature
        hs_1_emd = F.normalize(hs_1, dim=1) #normalize embedding dimension B*d
        hs_2_emd = F.normalize(hs_2, dim=1) #normalize embedding dimension
        batch_size = hs_1_emd.size(0)
        out = torch.cat([hs_1_emd, hs_2_emd], dim=0)
        neg = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
        mask = self.get_negativae_mask(batch_size).to(neg.device)
        neg = neg.masked_select(mask).view(2*batch_size, -1)
        
        pos = torch.exp(torch.sum(hs_1_emd * hs_2_emd, dim=-1) / temperature)
        pos = torch.cat([pos, pos], dim=0)
        
        if estimator == 'hard':
            N = batch_size * 2 - 2
            imp = (beta * neg.log()).exp()
            reweight_neg = (imp*neg).sum(dim=-1) / imp.mean(dim=-1)
            Ng = (-tau_plus * N * pos + reweight_neg) / (1-tau_plus)
            #Ng = torch.clamp(Ng, min = N * np.e**(-1 / temperature))
            loss = (-torch.log(pos / (pos+Ng))).mean()
            
        elif estimator == 'flatclr':
            pos = pos.view(pos.size(0), -1)
            logits = neg - pos
            v = torch.logsumexp(logits, dim=1, keepdim=True)
            loss_vec = torch.exp(v - v.detach())
            loss = loss_vec.mean()-1
            
        elif estimator == 'barlow':
            c = self.barlow(hs_1).T @ self.barlow(hs_2) #normalize batch dimension
            c.div_(batch_size)
            on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
            off_diag = self.off_diagonal(c).pow_(2).sum()
            loss = on_diag + lambd * off_diag
            
        elif estimator == 'simclr':    
            Ng = neg.sum(dim=-1)
            loss = (-torch.log(pos / (pos+Ng))).mean()
        else:
            raise Exception('Invalid estimator selected, choice["hard", "simclr", "flatclr", "barlow"]')
            
        return loss 
              
    def off_diagonal(self, x):
        n, m = x.size()
        assert n == m
        return x.flatten()[:-1].view(n-1, n+1)[:, 1:].flatten()
   
class reconstruct(nn.Module):
    def __init__(self):
        super().__init__()
        self.dlinear = nn.Sequential(torch.nn.Linear(256, 5120),
                        nn.ReLU()) #see more in subsampling.py

        self.dconv = nn.Sequential(
                             torch.nn.ConvTranspose2d(256, 256, 3, 2),
                             torch.nn.ReLU(),
                             torch.nn.ConvTranspose2d(256, 1, 3, 2))
        
    def forward(self, speech_hs_pad, args, speech_source):
        speech_hs_pad_re = self.dlinear(speech_hs_pad)
        a, b, c = speech_hs_pad_re.size()
        speech_hs_pad_re = speech_hs_pad_re.view(a, b, input_dim, -1).transpose(1, 2).contiguous()
        speech_hs_pad_re = self.dconv(speech_hs_pad_re).squeeze(1)
        loss_rescontruct = nn.MSELoss(reduction="mean")(speech_source[:, :speech_hs_pad_re.size(1), :], speech_hs_pad_re)
        return loss_rescontruct

class SyncFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor):
        ctx.batch_size = tensor.size(0)      
        output = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())] 
        dist.all_gather(output, tensor)
        gathered_tensor = torch.cat(output, 0)
        return gathered_tensor
        
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        dist.all_reduce(grad_input, op=dist.ReduceOp.SUM, async_op=False)
        idx_from = dist.get_rank() * ctx.batch_size
        idx_to = (dist.get_rank() + 1) * ctx.batch_size
        return grad_input[idx_from:idx_to]     
