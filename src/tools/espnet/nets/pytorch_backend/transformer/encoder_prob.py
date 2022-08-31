#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Encoder definition."""

import logging
import torch
import torch.nn as nn
from tools.function import ctc_shrink
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.nets_utils import rename_state_dict
from espnet.nets.pytorch_backend.transducer.vgg import VGG2L
from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
from espnet.nets.pytorch_backend.transformer.dynamic_conv import DynamicConvolution
from espnet.nets.pytorch_backend.transformer.dynamic_conv2d import DynamicConvolution2D
from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding
from espnet.nets.pytorch_backend.transformer.encoder_layer import EncoderLayer
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet.nets.pytorch_backend.transformer.lightconv import LightweightConvolution
from espnet.nets.pytorch_backend.transformer.lightconv2d import LightweightConvolution2D
from espnet.nets.pytorch_backend.transformer.multi_layer_conv import Conv1dLinear
from espnet.nets.pytorch_backend.transformer.multi_layer_conv import MultiLayeredConv1d
from espnet.nets.pytorch_backend.transformer.positionwise_feed_forward import (
    PositionwiseFeedForward,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.repeat import repeat
from espnet.nets.pytorch_backend.transformer.subsampling import Conv2dSubsampling
from espnet.nets.pytorch_backend.transformer.subsampling import Conv2dSubsampling6
from espnet.nets.pytorch_backend.transformer.subsampling import Conv2dSubsampling8


def _pre_hook(
    state_dict,
    prefix,
    local_metadata,
    strict,
    missing_keys,
    unexpected_keys,
    error_msgs,
):
    # https://github.com/espnet/espnet/commit/21d70286c354c66c0350e65dc098d2ee236faccc#diff-bffb1396f038b317b2b64dd96e6d3563
    rename_state_dict(prefix + "input_layer.", prefix + "embed.", state_dict)
    # https://github.com/espnet/espnet/commit/3d422f6de8d4f03673b89e1caef698745ec749ea#diff-bffb1396f038b317b2b64dd96e6d3563
    rename_state_dict(prefix + "norm.", prefix + "after_norm.", state_dict)


class Encoder(torch.nn.Module):
    """Transformer encoder module.

    :param int idim: input dim
    :param int attention_dim: dimention of attention
    :param int attention_heads: the number of heads of multi head attention
    :param int linear_units: the number of units of position-wise feed forward
    :param int num_blocks: the number of decoder blocks
    :param float dropout_rate: dropout rate
    :param float attention_dropout_rate: dropout rate in attention
    :param float positional_dropout_rate: dropout rate after adding positional encoding
    :param str or torch.nn.Module input_layer: input layer type
    :param class pos_enc_class: PositionalEncoding or ScaledPositionalEncoding
    :param bool normalize_before: whether to use layer_norm before the first block
    :param bool concat_after: whether to concat attention layer's input and output
        if True, additional linear will be applied.
        i.e. x -> x + linear(concat(x, att(x)))
        if False, no additional linear will be applied. i.e. x -> x + att(x)
    :param str positionwise_layer_type: linear of conv1d
    :param int positionwise_conv_kernel_size: kernel size of positionwise conv1d layer
    :param int padding_idx: padding_idx for input_layer=embed
    """

    def __init__(
        self,
        idim,
        selfattention_layer_type="selfattn",
        attention_dim=256,
        attention_heads=4,
        conv_wshare=4,
        conv_kernel_length=11,
        conv_usebias=False,
        linear_units=2048,
        num_blocks=6,
        dropout_rate=0.1,
        positional_dropout_rate=0.1,
        attention_dropout_rate=0.0,
        input_layer="conv2d",
        pos_enc_class=PositionalEncoding,
        normalize_before=True,
        concat_after=False,
        positionwise_layer_type="linear",
        positionwise_conv_kernel_size=1,
        padding_idx=-1,
    ):
        """Construct an Encoder object."""
        super(Encoder, self).__init__()
        self._register_load_state_dict_pre_hook(_pre_hook)

        if input_layer == "linear":
            self.embed = torch.nn.Sequential(
                torch.nn.Linear(idim, attention_dim),
                torch.nn.LayerNorm(attention_dim),
                torch.nn.Dropout(dropout_rate),
                torch.nn.ReLU(),
                pos_enc_class(attention_dim, positional_dropout_rate),
            )
        elif input_layer == "conv2d":
            self.embed = Conv2dSubsampling(idim, attention_dim, dropout_rate)
        elif input_layer == "conv2d-scaled-pos-enc":
            self.embed = Conv2dSubsampling(
                idim,
                attention_dim,
                dropout_rate,
                pos_enc_class(attention_dim, positional_dropout_rate),
            )
        elif input_layer == "conv2d6":
            self.embed = Conv2dSubsampling6(idim, attention_dim, dropout_rate)
        elif input_layer == "conv2d8":
            self.embed = Conv2dSubsampling8(idim, attention_dim, dropout_rate)
        elif input_layer == "vgg2l":
            self.embed = VGG2L(idim, attention_dim)
        elif input_layer == "embed":
            self.embed = torch.nn.Sequential(
                torch.nn.Embedding(idim, attention_dim, padding_idx=padding_idx),
                pos_enc_class(attention_dim, positional_dropout_rate),
            )
        elif isinstance(input_layer, torch.nn.Module):
            self.embed = torch.nn.Sequential(
                input_layer,
                pos_enc_class(attention_dim, positional_dropout_rate),
            )
        elif input_layer is None:
            self.embed = torch.nn.Sequential(
                pos_enc_class(attention_dim, positional_dropout_rate)
            )
        else:
            raise ValueError("unknown input_layer: " + input_layer)
        self.normalize_before = normalize_before
        positionwise_layer, positionwise_layer_args = self.get_positionwise_layer(
            positionwise_layer_type,
            attention_dim,
            linear_units,
            dropout_rate,
            positionwise_conv_kernel_size,
        )
        if selfattention_layer_type == "selfattn":
            logging.info("encoder self-attention layer type = self-attention")
            self.encoders = nn.ModuleList([EncoderLayer(
                    attention_dim,
                    MultiHeadedAttention(
                        attention_heads, attention_dim, attention_dropout_rate
                    ),
                    positionwise_layer(*positionwise_layer_args),
                    dropout_rate,
                    normalize_before,
                    concat_after,
                ) for _ in range(num_blocks)])

        elif selfattention_layer_type == "lightconv":
            logging.info("encoder self-attention layer type = lightweight convolution")
            self.encoders = nn.ModuleList([EncoderLayer(
                    attention_dim,
                    LightweightConvolution(
                        conv_wshare,
                        attention_dim,
                        attention_dropout_rate,
                        conv_kernel_length,
                        lnum,
                        use_bias=conv_usebias,
                    ),
                    positionwise_layer(*positionwise_layer_args),
                    dropout_rate,
                    normalize_before,
                    concat_after,
                ) for _ in range(num_blocks)])

        elif selfattention_layer_type == "lightconv2d":
            logging.info(
                "encoder self-attention layer "
                "type = lightweight convolution 2-dimentional"
            )
            self.encoders = nn.ModuleList([EncoderLayer(
                    attention_dim,
                    LightweightConvolution2D(
                        conv_wshare,
                        attention_dim,
                        attention_dropout_rate,
                        conv_kernel_length,
                        lnum,
                        use_bias=conv_usebias,
                    ),
                    positionwise_layer(*positionwise_layer_args),
                    dropout_rate,
                    normalize_before,
                    concat_after,
                ) for _ in range(num_blocks)])

        elif selfattention_layer_type == "dynamicconv":
            logging.info("encoder self-attention layer type = dynamic convolution")
            self.encoders = nn.ModuleList([EncoderLayer(
                    attention_dim,
                    DynamicConvolution(
                        conv_wshare,
                        attention_dim,
                        attention_dropout_rate,
                        conv_kernel_length,
                        lnum,
                        use_bias=conv_usebias,
                    ),
                    positionwise_layer(*positionwise_layer_args),
                    dropout_rate,
                    normalize_before,
                    concat_after,
                ) for _ in range(num_blocks)])

        elif selfattention_layer_type == "dynamicconv2d":
            logging.info(
                "encoder self-attention layer type = dynamic convolution 2-dimentional"
            )
            self.encoders = nn.ModuleList([EncoderLayer(
                    attention_dim,
                    DynamicConvolution2D(
                        conv_wshare,
                        attention_dim,
                        attention_dropout_rate,
                        conv_kernel_length,
                        lnum,
                        use_bias=conv_usebias,
                    ),
                    positionwise_layer(*positionwise_layer_args),
                    dropout_rate,
                    normalize_before,
                    concat_after,
                ) for _ in range(num_blocks)])

        if self.normalize_before:
            self.after_norm = LayerNorm(attention_dim)

    def get_positionwise_layer(
        self,
        positionwise_layer_type="linear",
        attention_dim=256,
        linear_units=2048,
        dropout_rate=0.1,
        positionwise_conv_kernel_size=1,
    ):
        """Define positionwise layer."""
        if positionwise_layer_type == "linear":
            positionwise_layer = PositionwiseFeedForward
            positionwise_layer_args = (attention_dim, linear_units, dropout_rate)
        elif positionwise_layer_type == "conv1d":
            positionwise_layer = MultiLayeredConv1d
            positionwise_layer_args = (
                attention_dim,
                linear_units,
                positionwise_conv_kernel_size,
                dropout_rate,
            )
        elif positionwise_layer_type == "conv1d-linear":
            positionwise_layer = Conv1dLinear
            positionwise_layer_args = (
                attention_dim,
                linear_units,
                positionwise_conv_kernel_size,
                dropout_rate,
            )
        else:
            raise NotImplementedError("Support only linear or conv1d.")
        return positionwise_layer, positionwise_layer_args

    def forward(self, xs, masks, ctc, ctc_layer, ys_pad_src, shrink=False, out_layer=-1):
        """Encode input sequence.

        :param torch.Tensor xs: input tensor
        :param torch.Tensor masks: input mask
        :return: position embedded tensor and mask
        :rtype Tuple[torch.Tensor, torch.Tensor]:
        """
        if isinstance(
            self.embed,
            (Conv2dSubsampling, Conv2dSubsampling6, Conv2dSubsampling8, VGG2L),
        ):
            xs, masks = self.embed(xs, masks)
        else:
            xs = self.embed(xs)
        loss_asr_ctc = 0.0
        out, out_mask = None, None
        for i_layer, encoder_layer in enumerate(self.encoders):
            xs, masks = encoder_layer(xs, masks)
            
            if i_layer+1 == out_layer:
                out, out_mask = xs, masks
            if i_layer+1 == ctc_layer:
                if ctc is not None:
                    loss_asr_ctc, ys_hat_ctc = self.ctc_calculate(ctc, masks, xs, ys_pad_src)
                    if shrink:
                        xs, masks = ctc_shrink(ys_hat_ctc, masks, xs, -1, 0)         
        if self.normalize_before:
            xs = self.after_norm(xs)
            if out_layer == 12:
                out = xs
        if out is None:
            print("error out layer")
        else:
            return out, out_mask, loss_asr_ctc

    def infer(self, xs, masks, ctc, ctc_layer, shrink=False):
        if isinstance(
            self.embed,
            (Conv2dSubsampling, Conv2dSubsampling6, Conv2dSubsampling8, VGG2L),
        ):
            xs, masks = self.embed(xs, masks)
        else:
            xs = self.embed(xs)
        source_length = xs.size(1)
        for i_layer, encoder_layer in enumerate(self.encoders):
            xs, masks = encoder_layer(xs, masks)      
            if i_layer+1 == ctc_layer:
                if ctc is not None:
                    loss_asr_ctc, ys_hat_ctc = self.ctc_calculate(ctc, masks, xs)
                    if shrink:
                        xs, ilens_new = ctc_shrink(ys_hat_ctc, masks, xs, -1, 0)
        if self.normalize_before:
            xs = self.after_norm(xs)
        return xs, source_length, masks
        
    def ctc_calculate(self, ctc, hs_mask, hs_pad, ys_pad_src=None):
        loss_asr_ctc = 0.0
        batch_size = hs_pad.size(0)
        adim = hs_pad.size(2)
        if hs_mask is not None:
            hs_len = hs_mask.view(batch_size, -1).sum(1)
        else:
            hs_len = hs_pad.size(1)
        if ys_pad_src is not None:
            loss_asr_ctc = ctc(hs_pad.view(batch_size, -1, adim), hs_len, ys_pad_src)
        ys_hat_ctc = ctc.argmax(hs_pad.view(batch_size, -1, adim)).data
        return loss_asr_ctc, ys_hat_ctc


    def forward_one_step(self, xs, masks, cache=None):
        """Encode input frame.

        :param torch.Tensor xs: input tensor
        :param torch.Tensor masks: input mask
        :param List[torch.Tensor] cache: cache tensors
        :return: position embedded tensor, mask and new cache
        :rtype Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        if isinstance(self.embed, Conv2dSubsampling):
            xs, masks = self.embed(xs, masks)
        else:
            xs = self.embed(xs)
        if cache is None:
            cache = [None for _ in range(len(self.encoders))]
        new_cache = []
        for c, e in zip(cache, self.encoders):
            xs, masks = e(xs, masks, cache=c)
            new_cache.append(xs)
        if self.normalize_before:
            xs = self.after_norm(xs)
        return xs, masks, new_cache
