import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from bins.prob_task.transformer_st import E2E
from tools.function import load_trained_modules

class MyModule(pl.LightningModule):
    def __init__(self, idim, odim_trans, odim, args):
        super().__init__()
        self.args = args
        self.save_hyperparameters(args)
        self.model = E2E(idim, odim_trans, args)
        ##init st model by trained asr or mt model
        if args.enc_init is not None or args.dec_init is not None:
            load_trained_modules(args, self.model)
              
        #layers = list()
        #layers.append(nn.Linear(self.args.adim, 2048)) #speaker=2048
        #layers.append(nn.ReLU())
        #layers.append(nn.Dropout(0.1))
        #layers.append(nn.Linear(2048, 2048))
        #layers.append(nn.ReLU())
        #layers.append(nn.Dropout(0.1))
        #layers.append(nn.Linear(2048, odim))
               
        #self.layers = nn.ModuleList(layers)
        self.layers = nn.Linear(self.args.adim, odim)
        self.out_layer = args.out_layer
        
    def training_step(self, batch, batch_idx):
    
        inputs, input_sizes, label, utts = batch 
        hs_pad, hs_mask = self.model(inputs, input_sizes, out_layer=self.out_layer)

        lengths = hs_mask.sum(-1)
        hs_pad = hs_pad * hs_mask.unsqueeze(-1)
        hs_pad = torch.sum(hs_pad, dim=1) / lengths.unsqueeze(-1)
        #hs_pad = torch.mean(hs_pad, dim=1)
        
        #out = torch.zeros([hs_pad.size()[0], self.args.adim]).to(hs_pad.device)

        #for i in range(len(self.layers)):
            #hs_pad = self.layers[i](hs_pad)
        hs_pad = self.layers(hs_pad)        
        loss = nn.CrossEntropyLoss()(hs_pad, torch.tensor(label).to(hs_pad.device))
        index = hs_pad.argmax(1)
        acc = index.eq(torch.tensor(label).to(hs_pad.device)).sum() / len(label)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss
                
    def validation_step(self, batch, batch_idx):

        inputs, input_sizes, label, utts = batch 
        hs_pad, hs_mask = self.model(inputs, input_sizes, out_layer=self.out_layer)

        lengths = hs_mask.sum(-1)
        #print(lengths.size())  #batch size
        #print(hs_pad.size())   #batch*T*d
        #print(hs_mask.size())  #batch*T
        hs_pad = hs_pad * hs_mask.unsqueeze(-1)
        #print(hs_pad.size())   #batch*T*d
        #print(torch.sum(hs_pad, dim=1).size()) #batch*d
        hs_pad = torch.sum(hs_pad, dim=1) / lengths.unsqueeze(-1)
        #hs_pad = torch.mean(hs_pad, dim=1)

        #for i in range(len(self.layers)):
            #hs_pad = self.layers[i](hs_pad)
        hs_pad = self.layers(hs_pad)          
        loss = nn.CrossEntropyLoss()(hs_pad, torch.tensor(label).to(hs_pad.device))
        index = hs_pad.argmax(1)
        acc = index.eq(torch.tensor(label).to(hs_pad.device)).sum() / len(label)
        
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_step=True, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):

        inputs, input_sizes, label, utts = batch 
        hs_pad, hs_mask = self.model(inputs, input_sizes, out_layer=self.out_layer)
        hs_pad = torch.mean(hs_pad, dim=1)

        for i in range(len(self.layers)):
            hs_pad = self.layers[i](hs_pad)        
        index = hs_pad.argmax(1)
        acc = index.eq(torch.tensor(label).to(hs_pad.device)).sum() / len(label)
        
        self.log('test_acc', acc)
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.layers.parameters(), lr=self.args.prob_lr)
        return optimizer
         
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure, 
                             second_order_closure=None, on_tpu=False, using_native_amp=False, using_lbfgs=False):      
        optimizer.step(closure=optimizer_closure)        

    def optimizer_zero_grad(self, current_epoch, batch_idx, optimizer, opt_idx):
        optimizer.zero_grad()
