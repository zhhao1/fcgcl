import argparse, yaml
import json
import os,sys,io
import logging
import torch
import torch.nn as nn
import numpy as np

from scipy import linalg
from torch.nn import functional as F
from espnet.asr.pytorch_backend.asr_init import filter_modules
from espnet.asr.pytorch_backend.asr_init import get_partial_state_dict
from espnet.asr.pytorch_backend.asr_init import transfer_verification
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
def log_and_dict_prep(args):
    # logging info
    if args.verbose > 0:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    else:
        logging.basicConfig(
            level=logging.WARN,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
        logging.warning("Skip DEBUG/INFO messages")   
             
    if args.dict is not None:
        with open(args.dict, "rb") as f:
            dictionary = f.readlines()
        char_list = [entry.decode("utf-8").split(" ")[0] for entry in dictionary]
        char_list.insert(0, "<blank>")
        char_list.append("<eos>")
        args.char_list = char_list
    else:
        args.char_list = None 
        
    # load dictionary for debug log
    with open(args.valid_json, "rb") as f:
        valid_json = json.load(f)["utts"]
    utts = list(valid_json.keys())
    
    idim = int(valid_json[utts[0]]["input"][0]["shape"][-1]) #feat dimension
    odim = int(valid_json[utts[0]]["output"][0]["shape"][-1])#dict dimen
    
    args.idim = idim
    args.odim = odim
    assert odim == len(args.char_list)
    logging.info("#idim dims : " + str(idim))
    logging.info("#output dims: " + str(odim))
        
    # write model config
    #for key in sorted(vars(args).keys()):
        #logging.info("ARGS: " + key + ": " + str(vars(args)[key]))
    
    return idim, odim


def test_init(args):
    with open(args.trans_json, "rb") as f:
        valid_json = json.load(f)["utts"]
    utts = list(valid_json.keys())
    idim = int(valid_json[utts[0]]["input"][0]["shape"][-1]) #feat dimension
    odim = int(valid_json[utts[0]]["output"][0]["shape"][-1])#dict dimen

    file_name = open(args.hparam, 'r', encoding='utf-8')
    file_data = file_name.read()
    file_name.close()
    trained_args = argparse.Namespace(**yaml.load(file_data), Loader=yaml.FullLoader)
    return idim, odim, trained_args

def load_trained(trained_model_path, prefix='model', model=None, initilize_module=None):
    """
    model is None: return trained_model_path.state_dict()  : for init model
    model is not None: return model.load_state_dcit(trained_model_path.state_dict())   : for test
    """
    temp = torch.load(trained_model_path, map_location='cpu')
    if len(temp.keys()) < 15:   
        trained_model_state_dict = temp['state_dict']
    else:
        trained_model_state_dict = temp
    if model is not None:     
        main_state_dict = model.state_dict()   
    temp_state_dict = {}
    for module in trained_model_state_dict.keys():
        if initilize_module is None:
            if module.startswith(prefix):
                temp_state_dict[module.replace(prefix+".", "")] = trained_model_state_dict[module]
                if model is not None:
                    main_state_dict.update(temp_state_dict)
        else:
            for to_initilize in initilize_module:
                if module.startswith(prefix + '.' + to_initilize):
                    temp_state_dict[module.replace(prefix+".", "")] = trained_model_state_dict[module]
                    if model is not None:
                        main_state_dict.update(temp_state_dict)
    if model is None:
        return temp_state_dict
    else:
        model.load_state_dict(main_state_dict)

def print_new_keys(state_dict, modules, model_path):
        logging.warning("loading %s from model: %s", modules, model_path)

        for k in state_dict.keys():
            logging.warning("override %s" % k)

def load_trained_modules(args, model):
    enc_model_prefix = args.enc_model_prefix
    dec_model_prefix = args.dec_model_prefix
    enc_model_path = args.enc_init
    dec_model_path = args.dec_init
    enc_modules = args.enc_init_mods
    dec_modules = args.dec_init_mods
    main_state_dict = model.state_dict()
    for model_path, modules, prefix in [
            (enc_model_path, enc_modules, enc_model_prefix),
            (dec_model_path, dec_modules, dec_model_prefix),
        ]:
        if model_path is not None:
            model_state_dict = load_trained(model_path, prefix)
            modules = filter_modules(model_state_dict, modules)
            partial_state_dict = get_partial_state_dict(model_state_dict, modules)
            if transfer_verification(main_state_dict, partial_state_dict, modules):
                print_new_keys(partial_state_dict, modules, model_path)
                main_state_dict.update(partial_state_dict)
            else:
                logging.warning(f"modules {modules} in model {model_path} "
                                f"don't match your training config",)
    model.load_state_dict(main_state_dict)

def load_text_src(args, model, to_init):
    model_prefix = args.text_src_model_prefix
    model_path = args.text_src_init
    model_state_dict = load_trained(model_path, model_prefix)
    partial_state_dict = {}
    main_state_dict = model.state_dict()
    for key, value in model_state_dict.items():
        if key.startswith(to_init):
            partial_state_dict[key.replace(to_init+'.', 'text_src_' + to_init +'.')] = value
    print_new_keys(partial_state_dict, 'text_' + to_init, model_path)
    main_state_dict.update(partial_state_dict)
    model.load_state_dict(main_state_dict)

def ctc_shrink(tokens, hs_mask, encoder_out, pad, blk):
    """only count the first one for the repeat freams
    """
    torch.set_deterministic(False)
    device = encoder_out.device
    B, T, dmodel= encoder_out.size()
    # intermediate vars along time
    list_fires = []
    token_prev = torch.ones(B).to(device) * -1
    blk_batch = torch.ones(B).to(device) * blk
    pad_batch = torch.ones(B).to(device) * pad

    for t in range(T):
        token = tokens[:, t] #B
        fire_place = torch.logical_and(token != blk_batch, token != token_prev)#B True or False
        fire_place = torch.logical_and(fire_place, token != pad_batch)#B True or False
        list_fires.append(fire_place)#list_fires=[list, list], len(list_fires)=T, len(list)=B, list_fires: 1*T
        token_prev = token

    fires = torch.stack(list_fires, 1)
    if hs_mask is not None:
        fires = fires * hs_mask.squeeze(-2)
    len_decode = fires.sum(-1)
    max_decode_len = len_decode.max()
    min_decoder_len = len_decode.min()
    list_ls = []
    len_ls = []

    for b in range(B):
        l = encoder_out[b, :, :].index_select(0, torch.where(fires[b])[0])
        pad_l = torch.zeros([max_decode_len - l.size(0), dmodel]).to(device)
        list_ls.append(torch.cat([l, pad_l], 0))
        len_ls.append(l.size(0))
    encoder_shrunk = torch.stack(list_ls, 0)
    new_masks = (~make_pad_mask(len_decode.tolist())).to(encoder_out.device).unsqueeze(-2) 
    if min_decoder_len == 0:
        return encoder_out, hs_mask
    else:
        return encoder_shrunk, new_masks

def rrqr_reduce(encoder_out, hs_mask, tao=0.05):  #not fully test, maybe not correct
    device = encoder_out.device
    lengths = hs_mask.squeeze(-2).sum(-1) #B
    dmodel = encoder_out.size(2)
    lens_ls = []
    reduced_hs = []

    for i in range(len(lengths)):
        batch_i = encoder_out[i][:lengths[i]]
        q, r, p = linalg.qr(batch_i.detach().cpu().numpy().transpose(0, 1), pivoting=True)
        diag = np.abs(np.diag(r))
        if np.all(diag[1:] <= diag[:-1]) and lengths[i]<= dmodel:
            diag_norm = diag / np.max(diag)
            place_threshold = (diag_norm <= tao).sum()
            len_i = lengths[i] - place_threshold
            lens_ls.append(len_i)
            reduced_hs.append(batch_i[p[:len_i]])
        else:
            lens_ls.append(lengths[i])
            reduced_hs.append(batch_i)
            
    max_reduced_len = max(lens_ls)
    reduced_hs_pad = []
    for i in range(len(lens_ls)):
        pad_batch = torch.zeros([max_reduced_len-lens_ls[i], dmodel]).to(device)
        print(pad_batch.size())
        print(reduced_hs[i].size())
        reduced_hs_pad.append(torch.cat([reduced_hs[i], pad_batch], 0)) 
        
    encoder_out_reduced = torch.stack(reduced_hs_pad, 0)
    new_masks = (~make_pad_mask(lens_ls.tolist())).to(encoder_out.device)

    return encoder_out_reduced, new_masks


def load_train_temp(path, model):
    temp = torch.load(path, map_location='cpu')
    if len(temp.keys()) < 15:   
        trained_model_state_dict = temp['state_dict']
    else:
        trained_model_state_dict = temp
    main_state_dict = model.state_dict()
    print(main_state_dict.keys())
    dict_1 = {}
    for module in trained_model_state_dict.keys():
        if module.startswith('model_st.adaptive_layer.encoders'):
            layer_number = int(module.replace('model_st.adaptive_layer.encoders.', '')[0]) + 6
            dict_1['encoder.encoders.' + str(layer_number) + module.replace('model_st.adaptive_layer.encoders.', '')[1:]] = trained_model_state_dict[module]
        elif module.startswith('model_st') and not module.startswith('model_st.adaptive_layer.embed') and not module.startswith('model_st.adaptive_layer.after_norm'):
            dict_1[module.replace("model_st.", "")] = trained_model_state_dict[module]
    model.load_state_dict(dict_1) 

class cif_shrink(nn.Module):
    
    def __init__(self, dmodel):
        super().__init__()
        self.project = nn.Linear(dmodel, 1)
        #self.dropout = nn.Dropout(0.1)
    
    def forward(self, hs, hs_mask, threshold=0.95):
        x = self.project(hs).squeeze(-1)
        #x = self.dropout(x)
        alphas_s = torch.sigmoid(x)
        alphas = alphas_s * hs_mask.squeeze(1)
        
        hs, hs_mask = self.cif(hs, alphas, threshold, hs_mask, alphas_s)
        
        return hs, hs_mask
        
    def cif(self, hs, alphas, threshold, hs_mask, alphas_s):
        device = hs.device
        batch_size, len_time, dmodel = hs.size()
        
        integrate = torch.zeros([batch_size]).to(device)
        frame = torch.zeros([batch_size, dmodel]).to(device)
        
        list_fires = []
        list_frames = []
        
        for t in range(len_time):
            alpha = alphas[:, t]
            distribution_completion = torch.ones([batch_size]).to(device) - integrate
            integrate += alpha
            list_fires.append(integrate)
            
            fireplace = integrate > threshold
            integrate = torch.where(fireplace,
                                    integrate - torch.ones([batch_size]).to(device),
                                    integrate)
                                    
            cur = torch.where(fireplace,
                              distribution_completion,
                              alpha)
            
            remainds = alpha - cur
            
            frame += cur[:, None] * hs[:, t, :]
            list_frames.append(frame)
            frame = torch.where(fireplace[:, None].repeat(1, dmodel),
                                remainds[:, None]* hs[:, t, :],
                                frame)
        try:
            fires = torch.stack(list_fires, 1)
        except:
            torch.save(hs, "./hs.pt")
            torch.save(alphas, "./alphas.pt")
            torch.save(hs_mask, "./hs_mask.pt")
            torch.save(alphas_s, "./alphas_s.pt")
        frames = torch.stack(list_frames, 1)
        list_ls = []
        len_labels = torch.round(alphas.sum(-1)).int()
        max_label_len = len_labels.max()
        for b in range(batch_size):
            fire = fires[b, :]
            l = torch.index_select(frames[b, :, :], 0, torch.where(fire>threshold)[0])
            pad_l = torch.zeros([max_label_len - l.size(0), dmodel]).to(device)
            list_ls.append(torch.cat([l, pad_l], 0))
        
        new_masks = (~make_pad_mask(len_labels.tolist())).to(device).unsqueeze(-2) 
        
        return torch.stack(list_ls, 0), new_masks                                

def loss_metrics(src, tgt, ys_out_pad, ignore_id, loss_type='CE', normalize_length=False):
    """
    src B*T*d
    tgt B*T*d
    """
    normalize_length = normalize_length
    batch_size = src.size(0)
    src = src.view(-1, tgt.size(-1)) # (B*T)*d
    src = F.log_softmax(src, dim=-1)
    tgt = tgt.view(-1, tgt.size(-1)) # (B*T)*d
    tgt = F.softmax(tgt, dim=-1)
    ys_out_pad = ys_out_pad.view(-1) # (B*T)
 
    ignore = ys_out_pad == ignore_id
    total = len(ys_out_pad) - ignore.sum().item() # (B*T)-pad
    denom = total if normalize_length else batch_size

    #kl loss
    if loss_type == 'KL':
        kl = nn.KLDivLoss(reduction="none")(src, tgt)
        loss = kl.masked_fill(ignore.unsqueeze(1), 0).sum() / denom    
    #ce loss
    elif loss_type == 'CE':
        ce = -torch.mul(src, tgt) # (B*T)*d
        loss = ce.masked_fill(ignore.unsqueeze(1), 0).sum() / total
    else:
        print("error command")            
    return loss

def remove_pad_and_mean(hs, hs_mask):
    torch.set_deterministic(False)
    hs_mask = hs_mask.squeeze(1)
    list_mean = []
    for i in range(hs.size(0)):
        temp = torch.index_select(hs[i], 0, torch.where(hs_mask[i]>0)[0])
        temp = torch.mean(temp, dim=0)
        list_mean.append(temp)
    mean = torch.stack(list_mean, 0)
    return mean

def mask_out(encoder_out, prob, value, position, src_lengths=None):
    """
    masked position is set to 1(false)
    """
    if position == 'row':
        if src_lengths is not None:
            total_length = sum(src_lengths)
        else:
            total_length = encoder_out.size(0) * encoder_out.size(1)
        total_sample = np.random.binomial(n=1, p=prob, size=total_length)
        batch_size = encoder_out.size(0)
        start = 0
        for i in range(batch_size):
            if src_lengths is not None:
                length = src_lengths[i]
            else:
                length = encoder_out.size(1)
            sample = torch.from_numpy(total_sample[start:start+length]).unsqueeze(1)
            start += length
            sample = sample.to(encoder_out.device)
            if value == "0":
                encoder_out[i, :length] = torch.mul(encoder_out[i, :length], sample.lt(1))
            elif value == "mean":
                value_matrix = torch.mean(encoder_out[i, :length], dim=0).unsqueeze(0) 
                value_matrix = torch.mul(value_matrix, sample)
                encoder_out[i, :length] = torch.mul(encoder_out[i, :length], sample.lt(1)) + value_matrix
            else:
                print("error value define")
                
    elif position == 'col':
        sample = np.random.binomial(n=1, p=prob, size=(encoder_out.size(0), encoder_out.size(2)))
        sample = torch.from_numpy(sample).unsqueeze(1)
        sample = sample.to(encoder_out.device)
        if value == '0':
            encoder_out_masked = torch.mul(encoder_out, sample.lt(1))    
        elif value == 'mean':
            value_matrix = torch.mean(encoder_out, dim=2).unsqueeze(2)
            value_matrix = torch.mul(value_matrix, sample)
            encoder_out = torch.mul(encoder_out, sample.lt(1)) + value_matrix
        else:
            print("error value define")
    else:
        print("error position define") 
    return encoder_out

def mask_block(encoder_out, block_number, K):
    """
    masked position is set to 1(false)
    """
    dmodel = encoder_out.size(2)
    batch_size = encoder_out.size(0)
    
    if block_number == 1:
        k = torch.Tensor(1, int(batch_size)).random_(0, K)#number dimension in a block
        value_matrix = torch.mean(encoder_out, dim=2).unsqueeze(2)
        for i in range(batch_size):
            position = int(torch.Tensor(1).random_(0, dmodel-int(k[0][i])))    
            encoder_out[i, :, position:position+int(k[0][i])] = value_matrix[i]
                
    elif block_number == 2:
        k = torch.Tensor(block_number, int(batch_size)).random_(1, K)#number dimension in a block
        value_matrix = torch.mean(encoder_out, dim=2).unsqueeze(2)
        max_number, _ = k.max(dim=0)
        for i in range(batch_size):
            position1 = int(torch.Tensor(1).random_(0, dmodel-2*int(max_number[i])))
            position2 = int(torch.Tensor(1).random_(position1+int(max_number[i]), dmodel-int(max_number[i])))     
            encoder_out[i, :, position1:position1+int(k[0][i])] = value_matrix[i]
            encoder_out[i, :, position2:position2+int(k[1][i])] = value_matrix[i]
                
    elif block_number == 3:
        k = torch.Tensor(block_number, int(batch_size)).random_(1, K)#number dimension in a block
        value_matrix = torch.mean(encoder_out, dim=2).unsqueeze(2)
        max_number, _ = k.max(dim=0)
        for i in range(batch_size):
            position1 = int(torch.Tensor(1).random_(0, dmodel-3*int(max_number[i])))
            position2 = int(torch.Tensor(1).random_(position1+int(max_number[i]), dmodel-2*int(max_number[i])))  
            position3 = int(torch.Tensor(1).random_(position2+int(max_number[i]), dmodel-int(max_number[i])))    
            encoder_out[i, :, position1:position1+int(k[0][i])] = value_matrix[i]
            encoder_out[i, :, position2:position2+int(k[1][i])] = value_matrix[i]
            encoder_out[i, :, position3:position3+int(k[2][i])] = value_matrix[i]
            
    return encoder_out

def compute_kernel_bias(vecs):
    mu = vecs.mean(axis=0, keepdims=True)
    cov = np.cov(vecs.T)
    u, s, vh = np.linalg.svd(cov)
    W = np.dot(u, np.diag(1 / np.sqrt(s)))
    return W, -mu

def transform_and_normalize(vecs, kernel, bias):
    vecs = (vecs + bias).dot(kernel)
    #return normalize(vecs)
    return vecs

def normalize(vecs):
    return vecs / (vecs**2).sum(axis=1, keepdims=True)**0.5    
