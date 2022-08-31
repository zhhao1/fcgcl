import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from bins.transformer_st import E2E
from tools.function import load_trained_modules

class MyModule(pl.LightningModule):
    def __init__(self, idim, odim, args):
        super().__init__()
        self.args = args
        self.save_hyperparameters(args)
        self.model = E2E(idim, odim, args)
        ##init st model by trained asr or mt model
        if args.enc_init is not None or args.dec_init is not None:
            load_trained_modules(args, self.model)
        
    def training_step(self, batch, batch_idx):
    
        xs_pad, ilens, ys_pad, ys_pad_src = batch 
        loss, acc, loss_ctc, bleu = self.model(xs_pad, ilens, ys_pad, ys_pad_src, flag='Train')
        
        metric = {'tr_acc': acc, 'ctc': loss_ctc}
        self.log_dict(metric, on_step=True, on_epoch=True)
        self.log('tr_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
        
    def validation_step(self, batch, batch_idx):
    
        xs_pad, ilens, ys_pad, ys_pad_src = batch  
        loss, acc, loss_ctc, bleu = self.model(xs_pad, ilens, ys_pad, ys_pad_src, flag='Valid')
        
        metric = {'val_loss': loss, 'val_acc': acc, 'val_bleu': bleu}       
        self.log_dict(metric, on_step=True, on_epoch=True, prog_bar=True)
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.finetune_lr)
        return optimizer
         
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure, 
                             second_order_closure=None, on_tpu=False, using_native_amp=False, using_lbfgs=False):          
        optimizer.step(closure=optimizer_closure)        

    def optimizer_zero_grad(self, current_epoch, batch_idx, optimizer, opt_idx):
        optimizer.zero_grad()
