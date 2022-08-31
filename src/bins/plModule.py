import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from bins.transformer_st import E2E
from tools.function import load_trained_modules, load_text_src
from thop import profile
from ptflops import get_model_complexity_info
from deepspeed.profiling.flops_profiler import FlopsProfiler

class MyModule(pl.LightningModule):
    def __init__(self, idim, odim, args):
        super().__init__()
        self.args = args
        self.save_hyperparameters(args)
        self.model = E2E(idim, odim, args)
        ##init st model by trained asr or mt model
        if args.enc_init is not None or args.dec_init is not None:
            load_trained_modules(args, self.model)
        if args.text_src_init is not None:
            load_text_src(args, self.model, 'encoder')
        if args.online_KD:
            load_text_src(args, self.model, 'decoder')
        for p in self.model.text_src_encoder.parameters():
            p.requires_grad = False
        for p in self.model.text_src_decoder.parameters():
            p.requires_grad = False
        self.automatic_optimization = args.automatic_optimization
        self.accumulate_batches = args.accumulate_batches
        self.optimizer_steps = 1
        self.grad_clip_threshold = args.gradient_clip_threshold
        self.manual_optimizer = NoamOpt(args.adim, args.transformer_scale, args.transformer_warmup_steps, torch.optim.Adam(self.parameters(), betas=(0.9, 0.98), eps=1e-9))
        self.prof = FlopsProfiler(self.model)
        
    def training_step(self, batch, batch_idx):
        if self.automatic_optimization:
            speech_pad, speech_ilens, text_src_pad, text_s_ilens, text_tgt_pad, text_t_ilens, ys_pad = batch
            #if self.global_step == 5:
                #self.prof.start_profile()
            loss, loss_st, loss_mt, loss_ae, st_acc = self.model(speech_pad, speech_ilens, text_src_pad, text_s_ilens, text_tgt_pad, text_t_ilens, ys_pad, current_epoch=self.current_epoch, 
            steps=self.global_step, flag='Train')
            #if self.global_step == 5:
                #self.prof.stop_profile()
                #flops = self.prof.get_total_flops()
                #self.prof.print_model_profile(profile_step=5)
                #self.prof.end_profile()
                #print(flops/(1000**3))
        
            metric = {'total_loss': loss, 'loss_mt': loss_mt, 'loss_ae': loss_ae,}
            metric_bar = {'tr_acc': st_acc, 'loss_st': loss_st}
            self.log_dict(metric, on_step=True, on_epoch=True)
            self.log_dict(metric_bar, on_step=True, on_epoch=True, prog_bar=True)
            return loss
        else:
            opt = self.manual_optimizer
            speech_pad, speech_ilens, text_src_pad, text_s_ilens, text_tgt_pad, text_t_ilens, ys_pad = batch
            loss, loss_st, loss_mt, loss_ae, st_acc = self.model(speech_pad, speech_ilens, text_src_pad, text_s_ilens, text_tgt_pad, text_t_ilens, ys_pad, current_epoch=self.current_epoch, 
            steps=self.global_step, flag='Train')
            loss = loss / self.accumulate_batches
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip_threshold)
            self.manual_backward(loss)
            metric = {'total_loss': loss, 'loss_mt': loss_mt, 'loss_ae': loss_ae,}
            metric_bar = {'tr_acc': st_acc, 'loss_st': loss_st}
            self.log_dict(metric, on_step=True, on_epoch=True)
            self.log_dict(metric_bar, on_step=True, on_epoch=True, prog_bar=True)
            if (batch_idx + 1) % self.accumulate_batches == 0:
                opt.step()
                opt.optimizer.zero_grad()            
        
    def validation_step(self, batch, batch_idx):
        if not self.args.moco:
            speech_pad, speech_ilens, text_src_pad, text_s_ilens, text_tgt_pad, text_t_ilens, ys_pad = batch  
            loss, loss_st, loss_mt, loss_ae, st_acc = self.model(speech_pad, speech_ilens, text_src_pad, text_s_ilens, text_tgt_pad, text_t_ilens, ys_pad, flag='test')
        
            metric_bar = {'val_acc': st_acc, 'val_loss': loss_st, 'val_loss_mt': loss_mt}
            self.log_dict(metric_bar, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
                
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
        return optimizer
         
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure, 
                             second_order_closure=None, on_tpu=False, using_native_amp=False, using_lbfgs=False):
        step = self.trainer.global_step + 1
        factor = self.args.transformer_scale
        model_size = self.args.adim
        warmup = self.args.transformer_warmup_steps
        rate = factor * (model_size ** (-0.5)) * min(step ** (-0.5), step * warmup ** (-1.5))
        for pg in optimizer.param_groups:
            pg['lr'] = rate            
        optimizer.step(closure=optimizer_closure)        

    def optimizer_zero_grad(self, current_epoch, batch_idx, optimizer, opt_idx):
        optimizer.zero_grad()
        
class NoamOpt:
    "Optim wrapper that implements rate."

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5)) * min(step ** (-0.5), step * self.warmup ** (-1.5))
