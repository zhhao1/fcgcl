"""Transformer speech recognition model (pytorch)."""
from argparse import Namespace
from distutils.util import strtobool
import logging
import math
import numpy as np

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch import Tensor
from torch.nn import functional as F
from bins.cl_loss import contrastive
from tools.function import ctc_shrink, cif_shrink, loss_metrics, remove_pad_and_mean

from espnet.nets.e2e_asr_common import end_detect
from espnet.nets.e2e_asr_common import ErrorCalculator as ASRErrorCalculator
from espnet.nets.e2e_mt_common import ErrorCalculator as MTErrorCalculator
from espnet.nets.pytorch_backend.ctc import CTC
from espnet.nets.pytorch_backend.nets_utils import get_subsample
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.nets_utils import to_device
from espnet.nets.pytorch_backend.nets_utils import pad_list
from espnet.nets.pytorch_backend.nets_utils import th_accuracy
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
from espnet.nets.pytorch_backend.transformer.decoder import Decoder
from espnet.nets.pytorch_backend.transformer.encoder import Encoder
from espnet.nets.pytorch_backend.transformer.encoder_mt import Encoder as Encoder_text
from espnet.nets.pytorch_backend.transformer.embedding import embed
from espnet.nets.pytorch_backend.transformer.initializer import initialize
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import (
    LabelSmoothingLoss,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.mask import subsequent_mask
from espnet.nets.pytorch_backend.transformer.mask import target_mask
from espnet.utils.fill_missing_args import fill_missing_args
from thop import profile
CTC_LOSS_THRESHOLD = 10000

class E2E(torch.nn.Module):

    @staticmethod
    def add_arguments(parser):
        """Add arguments."""
        group = parser.add_argument_group("transformer model setting")

        group.add_argument(
            "--transformer-init",
            type=str,
            default="pytorch",
            choices=[
                "pytorch",
                "xavier_uniform",
                "xavier_normal",
                "kaiming_uniform",
                "kaiming_normal",
            ],
            help="how to initialize transformer parameters",
        )
        group.add_argument(
            "--transformer-input-layer",
            type=str,
            default="conv2d",
            choices=["conv1d", "conv2d", "linear", "embed"],
            help="transformer input layer type",
        )
        group.add_argument(
            "--transformer-attn-dropout-rate",
            default=None,
            type=float,
            help="dropout in transformer attention. use --dropout-rate if None is set",
        )
        group.add_argument(
            "--transformer-scale",
            default=10.0,
            type=float,
            help="scale value of learning rate",
        )
        group.add_argument(
            "--transformer-warmup-steps",
            default=25000,
            type=int,
            help="optimizer warmup steps",
        )
        group.add_argument(
            "--transformer-length-normalized-loss",
            default=False,
            type=strtobool,
            help="normalize loss by length",
        )

        group.add_argument(
            "--dropout-rate",
            default=0.1,
            type=float,
            help="Dropout rate for the encoder",
        )
        # Encoder
        group.add_argument(
            "--elayers-speech", default=4, type=int, help="Number of encoder layers for source language",
        )
        group.add_argument(
            "--elayers-text-src", default=4, type=int, help="Number of encoder layers for text in source language",
        )
        group.add_argument(
            "--elayers-text-tgt", default=4, type=int, help="Number of encoder layers for text in tgt language",
        )
        group.add_argument(
            "--elayers-shared", default=4, type=int, help="Number of encoder layers for shared about speech-text_src-text_tgt",
        )
                      
        group.add_argument(
            "--eunits",
            "-u",
            default=2048,
            type=int,
            help="Number of encoder hidden units",
        )
        # Attention
        group.add_argument(
            "--adim",
            default=256,
            type=int,
            help="Number of attention transformation dimensions",
        )
        group.add_argument(
            "--aheads",
            default=4,
            type=int,
            help="Number of heads for multi head attention",
        )
        # Decoder
        group.add_argument(
            "--dlayers", default=6, type=int, help="Number of decoder layers"
        )
        group.add_argument(
            "--dunits", default=2048, type=int, help="Number of decoder hidden units"
        )
        return parser

    def __init__(self, idim, odim, args, ignore_id=-1):
        """Construct an E2E object.

        :param int idim: dimension of inputs
        :param int odim: dimension of outputs
        :param Namespace args: argument Namespace containing options
        """
        torch.nn.Module.__init__(self)
        
        args = fill_missing_args(args, self.add_arguments)
        
        if args.transformer_attn_dropout_rate is None:
            args.transformer_attn_dropout_rate = args.dropout_rate

        self.encoder = Encoder(
            idim=idim,
            attention_dim=args.adim,
            attention_heads=args.aheads,
            linear_units=args.eunits,
            num_blocks=args.elayers_speech,
            input_layer=args.transformer_input_layer,
            dropout_rate=args.dropout_rate,
            positional_dropout_rate=args.dropout_rate,
            attention_dropout_rate=args.transformer_attn_dropout_rate,
            dropdim=args.dropdim_speech_encoder,
            shrink=args.ctc_shrink,
            ctc=args.ctc,
        )                       
        self.text_src_encoder = Encoder_text(
            idim=odim,
            attention_dim=args.adim,
            attention_heads=args.aheads,
            linear_units=args.eunits,
            num_blocks=args.elayers_text_src,
            input_layer="embed",
            dropout_rate=args.dropout_rate,
            positional_dropout_rate=args.dropout_rate,
            attention_dropout_rate=args.transformer_attn_dropout_rate,
        )        
        self.speech_embed = embed(idim, args.adim, args.dropout_rate, "linear")
        #self.text_embed = embed(odim, args.adim, args.dropout_rate, "embed")        
                        
        self.decoder = Decoder(
            odim=odim,
            attention_dim=args.adim,
            attention_heads=args.aheads,
            linear_units=args.dunits,
            num_blocks=args.dlayers,
            dropout_rate=args.dropout_rate,
            positional_dropout_rate=args.dropout_rate,
            self_attention_dropout_rate=args.transformer_attn_dropout_rate,
            src_attention_dropout_rate=args.transformer_attn_dropout_rate,
            dropdim=args.dropdim_decoder,
        )
        self.text_src_decoder = Decoder(
            odim=odim,
            attention_dim=args.adim,
            attention_heads=args.aheads,
            linear_units=args.dunits,
            num_blocks=args.dlayers,
            dropout_rate=args.dropout_rate,
            positional_dropout_rate=args.dropout_rate,
            self_attention_dropout_rate=args.transformer_attn_dropout_rate,
            src_attention_dropout_rate=args.transformer_attn_dropout_rate,
            dropdim=args.dropdim_decoder,
        )
        self.pad = 0  # use <blank> for padding
        self.sos = odim - 1
        self.eos = odim - 1
        self.odim = odim
        self.ignore_id = ignore_id
        self.subsample = get_subsample(args, mode="st", arch="transformer")
        self.cross_modal = args.cross_modal
        self.temperature = args.temperature
        self.online_KD = args.online_KD
        self.online_KD_type = args.online_KD_type
        self.criterion = LabelSmoothingLoss(
            self.odim,
            self.ignore_id,
            args.lsm_weight,
            args.transformer_length_normalized_loss,
        )
        self.adim = args.adim
        
        # translation error calculator
        self.error_calculator = MTErrorCalculator(
            args.char_list, args.sym_space, args.sym_blank
        )
        if args.joint_training or args.pretrain:
            self.cl_loss = contrastive(args)
        else:
            self.cl_loss = None
            
        if args.ctc:
            self.ctc = CTC(odim, args.adim, args.dropout_rate, ctc_type=args.ctc_type, reduce=True, ignore_id=0)
        else:
            self.ctc = None
            
        if args.reconstruct:
            self.reconstruct_linear = nn.Linear(args.adim, idim)
            self.mask_frame = nn.Parameter(torch.zeros(1, 1, idim))
                      
        self.reset_parameters(args)  # place after the submodule initialization
        self.args = args
        
    def reset_parameters(self, args):
        """Initialize parameters."""
        # initialize parameters
        initialize(self, args.transformer_init)

    def forward(self, speech_pad, speech_ilens, text_src_pad, text_src_ilens, text_tgt_pad, text_tgt_ilens, ys_pad, flag='Train', current_epoch=0, steps=0):
        """E2E forward.

        :param torch.Tensor xs_pad: batch of padded source sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of source sequences (B)
        :param torch.Tensor ys_pad: batch of padded target sequences (B, Lmax)
        :param torch.Tensor ys_pad_src: batch of padded target sequences (B, Lmax)
        :return: ctc loass value
        :rtype: torch.Tensor
        :return: attention loss value
        :rtype: torch.Tensor
        :return: accuracy in attention decoder
        :rtype: float
        """
        #init metrics value
        loss, loss_st, loss_mt, loss_ae, st_acc, loss_kd = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        loss_cl = 0.0
        pure_st, pretrain, joint_training, reconstruct = self.args.pure_st, self.args.pretrain, self.args.joint_training, self.args.reconstruct
        kd = self.args.online_KD

        #for decoder input
        ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id)
        ys_mask = target_mask(ys_in_pad, self.ignore_id)
        
        speech_pad = speech_pad[:, : max(speech_ilens)]
        speech_src_mask = (~make_pad_mask(speech_ilens.tolist())).to(speech_pad.device).unsqueeze(-2)
        speech_pad_, speech_src_mask = self.speech_embed(speech_pad, speech_src_mask)
        speech_hs_pad, speech_hs_mask, speech_src_hs_first_pad = self.encoder(speech_pad_, speech_src_mask, self.args, flag=flag)
        
        if pretrain:
            text_src_pad = text_src_pad[:, : max(text_src_ilens)]
            text_src_mask = (~make_pad_mask(text_src_ilens.tolist())).to(text_src_pad.device).unsqueeze(-2)
            text_src_pad, text_src_mask = self.text_src_emded(text_src_pad, text_src_mask)
            text_src_hs_pad, text_src_mask = self.encoder(text_src_pad, text_src_mask)
            
            loss_cl = self.args.cl_loss_scale * self.cl_loss(speech_hs_pad, text_src_hs_pad, speech_hs_mask, text_src_mask, self.args)
            loss, loss_st =  loss_cl, loss_cl
            
        if pure_st:
            st_pred_pad, pred_mask = self.decoder(ys_in_pad, ys_mask, speech_hs_pad, speech_hs_mask, args=self.args, flag=flag)
            loss_st = self.criterion(st_pred_pad, ys_out_pad)
            st_acc = th_accuracy(st_pred_pad.view(-1, self.odim), ys_out_pad, ignore_label=self.ignore_id)
            loss = loss_st
            
        if reconstruct:
            # mask
            torch.set_deterministic(False)
            B, T, D = speech_pad.size()
            len_keep = int(T * (1-0.3))  # mask_ration = 0.3
            noise = torch.rand(B, T, device=speech_pad.device)
            ids_shuffle = torch.argsort(noise, dim=1)
            ids_restore = torch.argsort(ids_shuffle, dim=1)
            ids_keep = ids_shuffle[:, :len_keep]
            
            x_masked = torch.gather(speech_pad, dim=1, index=ids_keep.unsqueeze(-1).repeat(1,1,D))
            mask_token = self.mask_frame.repeat(speech_pad.size(0), ids_restore.size(1)-x_masked.size(1), 1)
            
            xs = torch.cat([x_masked, mask_token], dim=1)
            speech_pad_masked = torch.gather(xs, dim=1, index=ids_restore.unsqueeze(-1).repeat(1,1,xs.size(2)))
            mask = torch.ones([B,T], device=xs.device)
            mask[:, :len_keep] = 0
            mask = torch.gather(mask, dim=1, index=ids_restore)  # 1:removed, 0:keeped, speech_src_mask:pad mask, mask:random mask
            speech_pad_masked, speech_src_mask = self.speech_embed(speech_pad_masked, speech_src_mask)
            speech_hs_pad_masked, speech_hs_mask = self.encoder(speech_pad_masked, speech_src_mask)                        
            reconstruct_enc = self.reconstruct_linear(speech_hs_pad_masked)
            
            loss_rec = (reconstruct_enc - speech_pad) ** 2
            loss_rec = loss_rec.mean(dim=-1)
            mask = speech_src_mask.squeeze(1)
            loss_rec = (loss_rec * mask).sum() / mask.sum()

            st_pred_pad, pred_mask = self.decoder(ys_in_pad, ys_mask, speech_hs_pad, speech_hs_mask, args=self.args)
            loss_st = self.criterion(st_pred_pad, ys_out_pad)
            st_acc = th_accuracy(st_pred_pad.view(-1, self.odim), ys_out_pad, ignore_label=self.ignore_id)
            loss = loss_rec + loss_st        
            
        if joint_training:
            text_src_pad = text_src_pad[:, : max(text_src_ilens)]
            text_src_mask = (~make_pad_mask(text_src_ilens.tolist())).to(text_src_pad.device).unsqueeze(-2)
            text_src_hs_pad, text_src_mask, text_src_hs_first_pad, first = self.text_src_encoder(text_src_pad, text_src_mask)             
                
            st_pred_pad, pred_mask = self.decoder(ys_in_pad, ys_mask, speech_hs_pad, speech_hs_mask, args=self.args, flag=flag)
            st_acc = th_accuracy(st_pred_pad.view(-1, self.odim), ys_out_pad, ignore_label=self.ignore_id)
            loss_st = self.criterion(st_pred_pad, ys_out_pad)
            if kd:
                coarse, fine = self.cl_loss(speech_hs_pad, text_src_hs_pad, None, speech_hs_mask, text_src_mask, None, self.args, text_src_hs_first_pad, speech_src_hs_first_pad)  
                mt_pred_pad, pred_mask = self.text_src_decoder(ys_in_pad, ys_mask, text_src_hs_pad, text_src_mask, args=self.args)
                loss_kd = loss_metrics(st_pred_pad, mt_pred_pad.detach(), ys_out_pad, self.ignore_id)
                loss = (1-self.args.a)*loss_kd + self.args.a*loss_st + self.args.b*coarse + self.args.c*fine  
                #loss = loss_kd + loss_st + coarse + fine                
                return loss, loss_st, loss_mt, loss_ae, st_acc     
                
            coarse, fine = self.cl_loss(speech_hs_pad, text_src_hs_pad, None, speech_hs_mask, text_src_mask, None, self.args, text_src_hs_first_pad, speech_src_hs_first_pad)                   
            loss = loss_st + self.args.b*coarse + self.args.c*fine
                                                                 
        return loss, loss_st, loss_mt, loss_ae, st_acc
                  
                  
    def scorers(self):
        """Scorers."""
        return dict(decoder=self.decoder)

    def encode(self, x, task):
        """Encode source acoustic features.

        :param ndarray x: source acoustic feature (T, D)
        :return: encoder outputs
        :rtype: torch.Tensor
        """        
        if task == 'ST':
            x = torch.as_tensor(x).unsqueeze(0)
            x, _ = self.speech_embed(x, None)
            enc_output, _ = self.encoder.infer(x, None)
        else:
            x = to_device(self, torch.from_numpy(np.fromiter(map(int, x[0]), dtype=np.int64)))
            x = x.unsqueeze(0)
            x, _ = self.text_src_emded(x, None)
            enc_output, _ = self.encoder.infer(x, None)
        return enc_output #1*T*dmodel

    def translate(
        self, x, trans_args, char_list=None, task='ST', rnnlm=None, use_jit=False
    ):
        """Translate input speech.

        :param ndnarray x: input acoustic feature (B, T, D) or (T, D)
        :param Namespace trans_args: argment Namespace contraining options
        :param list char_list: list of characters
        :param torch.nn.Module rnnlm: language model module
        :return: N-best decoding results
        :rtype: list
        """
        self.eval()
        enc_output = self.encode(x, task)
        source_length = enc_output.size(1)
        logging.info("input lengths: " + str(source_length))
        
        # preprate sos
        if getattr(trans_args, "tgt_lang", False):
            if self.replace_sos:
                y = torch.tensor(char_list.index(trans_args.tgt_lang)).type_as(enc_output).long()
        else:
            y = torch.tensor(self.sos).type_as(enc_output).long()  

        
        # search parms
        beam = trans_args.beam_size
        penalty = trans_args.penalty

        vy = enc_output.new_zeros(1).long()

        if trans_args.maxlenratio == 0:
            maxlen = enc_output.shape[0]
        else:
            # maxlen >= 1
            maxlen = max(1, int(trans_args.maxlenratio * source_length))
        minlen = int(trans_args.minlenratio * source_length)
        logging.info("max output length: " + str(maxlen))
        logging.info("min output length: " + str(minlen))

        # initialize hypothesis
        if rnnlm:
            hyp = {"score": 0.0, "yseq": [y], "rnnlm_prev": None}
        else:
            hyp = {"score": 0.0, "yseq": [y]}
        hyps = [hyp]
        ended_hyps = []

        import six

        traced_decoder = None
        for i in six.moves.range(maxlen):
            logging.debug("position " + str(i))

            hyps_best_kept = []
            for hyp in hyps:
                vy[0] = hyp["yseq"][i]

                # get nbest local scores and their ids
                ys_mask = subsequent_mask(i + 1).unsqueeze(0).type_as(enc_output).long()
                ys = torch.tensor(hyp["yseq"]).unsqueeze(0).type_as(enc_output).long()
                # FIXME: jit does not match non-jit result
                if use_jit:
                    if traced_decoder is None:
                        traced_decoder = torch.jit.trace(
                            self.decoder.forward_one_step, (ys, ys_mask, enc_output)
                        )
                    local_att_scores = traced_decoder(ys, ys_mask, enc_output)[0]
                else:
                    local_att_scores = self.decoder.forward_one_step(
                        ys, ys_mask, enc_output
                    )[0]

                if rnnlm:
                    rnnlm_state, local_lm_scores = rnnlm.predict(hyp["rnnlm_prev"], vy)
                    local_scores = (
                        local_att_scores + trans_args.lm_weight * local_lm_scores
                    )
                else:
                    local_scores = local_att_scores

                local_best_scores, local_best_ids = torch.topk(
                    local_scores, beam, dim=1
                )

                for j in six.moves.range(beam):
                    new_hyp = {}
                    new_hyp["score"] = hyp["score"] + float(local_best_scores[0, j])
                    new_hyp["yseq"] = [0] * (1 + len(hyp["yseq"]))
                    new_hyp["yseq"][: len(hyp["yseq"])] = hyp["yseq"]
                    new_hyp["yseq"][len(hyp["yseq"])] = int(local_best_ids[0, j])
                    if rnnlm:
                        new_hyp["rnnlm_prev"] = rnnlm_state
                    # will be (2 x beam) hyps at most
                    hyps_best_kept.append(new_hyp)

                hyps_best_kept = sorted(
                    hyps_best_kept, key=lambda x: x["score"], reverse=True
                )[:beam]

            # sort and get nbest
            hyps = hyps_best_kept
            logging.debug("number of pruned hypothes: " + str(len(hyps)))
            if char_list is not None:
                logging.debug(
                    "best hypo: "
                    + "".join([char_list[int(x)] for x in hyps[0]["yseq"][1:]])
                )

            # add eos in the final loop to avoid that there are no ended hyps
            if i == maxlen - 1:
                logging.info("adding <eos> in the last postion in the loop")
                for hyp in hyps:
                    hyp["yseq"].append(self.eos)

            # add ended hypothes to a final list, and removed them from current hypothes
            # (this will be a probmlem, number of hyps < beam)
            remained_hyps = []
            for hyp in hyps:
                if hyp["yseq"][-1] == self.eos:
                    # only store the sequence that has more than minlen outputs
                    # also add penalty
                    if len(hyp["yseq"]) > minlen:
                        hyp["score"] += (i + 1) * penalty
                        if rnnlm:  # Word LM needs to add final <eos> score
                            hyp["score"] += trans_args.lm_weight * rnnlm.final(
                                hyp["rnnlm_prev"]
                            )
                        ended_hyps.append(hyp)
                else:
                    remained_hyps.append(hyp)

            # end detection
            if end_detect(ended_hyps, i) and trans_args.maxlenratio == 0.0:
                logging.info("end detected at %d", i)
                break

            hyps = remained_hyps
            if len(hyps) > 0:
                logging.debug("remeined hypothes: " + str(len(hyps)))
            else:
                logging.info("no hypothesis. Finish decoding.")
                break

            if char_list is not None:
                for hyp in hyps:
                    logging.debug(
                        "hypo: " + "".join([char_list[int(x)] for x in hyp["yseq"][1:]])
                    )

            logging.debug("number of ended hypothes: " + str(len(ended_hyps)))

        nbest_hyps = sorted(ended_hyps, key=lambda x: x["score"], reverse=True)[
            : min(len(ended_hyps), trans_args.nbest)
        ]

        # check number of hypotheis
        if len(nbest_hyps) == 0:
            logging.warning(
                "there is no N-best results, perform translation "
                "again with smaller minlenratio."
            )
            # should copy becasuse Namespace will be overwritten globally
            trans_args = Namespace(**vars(trans_args))
            trans_args.minlenratio = max(0.0, trans_args.minlenratio - 0.1)
            return self.translate(x, trans_args, char_list, rnnlm)

        logging.info("total log probability: " + str(nbest_hyps[0]["score"]))
        logging.info(
            "normalized log probability: "
            + str(nbest_hyps[0]["score"] / len(nbest_hyps[0]["yseq"]))
        )
        return nbest_hyps

    def calculate_all_attentions(self, xs_pad, ilens, ys_pad, ys_pad_src):
        """E2E attention calculation.

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor ys_pad: batch of padded token id sequence tensor (B, Lmax)
        :param torch.Tensor ys_pad_src:
            batch of padded token id sequence tensor (B, Lmax)
        :return: attention weights with the following shape,
            1) multi-head case => attention weights (B, H, Lmax, Tmax),
            2) other case => attention weights (B, Lmax, Tmax).
        :rtype: float ndarray
        """
        with torch.no_grad():
            self.forward(xs_pad, ilens, ys_pad, ys_pad_src)
        ret = dict()
        for name, m in self.named_modules():
            if (
                isinstance(m, MultiHeadedAttention) and m.attn is not None
            ):  # skip MHA for submodules
                ret[name] = m.attn.cpu().numpy()
        return ret
