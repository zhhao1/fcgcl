from argparse import Namespace
from distutils.util import strtobool
import logging
import math

import torch
import pytorch_lightning as pl
from tools.function import ctc_shrink

from espnet.nets.e2e_asr_common import end_detect
from espnet.nets.e2e_asr_common import ErrorCalculator as ASRErrorCalculator
from espnet.nets.e2e_mt_common import ErrorCalculator as MTErrorCalculator
from espnet.nets.pytorch_backend.ctc import CTC
from espnet.nets.pytorch_backend.nets_utils import get_subsample
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.nets_utils import pad_list
from espnet.nets.pytorch_backend.nets_utils import th_accuracy
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
from espnet.nets.pytorch_backend.transformer.decoder import Decoder
from espnet.nets.pytorch_backend.transformer.encoder import Encoder
from espnet.nets.pytorch_backend.transformer.initializer import initialize
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import (
    LabelSmoothingLoss,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.mask import subsequent_mask
from espnet.nets.pytorch_backend.transformer.mask import target_mask
from espnet.utils.fill_missing_args import fill_missing_args

CTC_LOSS_THRESHOLD = 10000

class E2E(torch.nn.Module):

    @staticmethod
    def add_arguments(parser):
        """Add arguments."""
        group = parser.add_argument_group("transformer model setting")

        group.add_argument(
            "--transformer-input-layer",
            type=str,
            default="conv2d",
            choices=["conv2d", "linear", "embed"],
            help="transformer input layer type",
        )
        # Encoder
        group.add_argument(
            "--elayers", default=12, type=int, help="Number of encoder layers",
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
        return parser

    @property
    def attention_plot_class(self):
        """Return PlotAttentionReport."""
        return PlotAttentionReport

    def __init__(self, idim, args, ignore_id=-1):
        """Construct an E2E object.

        :param int idim: dimension of inputs
        :param int odim: dimension of outputs
        :param Namespace args: argument Namespace containing options
        """
        torch.nn.Module.__init__(self)
        
        args = fill_missing_args(args, self.add_arguments)
                    
        self.encoder = Encoder(
            idim=idim,
            attention_dim=args.adim,
            attention_heads=args.aheads,
            linear_units=args.eunits,
            num_blocks=args.elayers,
            input_layer=args.transformer_input_layer,
            dropout_rate=0.1,
            positional_dropout_rate=0.1,
            attention_dropout_rate=0.1,
        )

    def forward(self, xs_pad, ilens):

        xs_pad = xs_pad[:, : max(ilens)].transpose(-2, -1).squeeze(1)  # for data parallel
        src_mask = (~make_pad_mask(ilens.tolist())).to(xs_pad.device).unsqueeze(-2)
        hs_pad, hs_mask, _ = self.encoder(xs_pad, src_mask, None, -1, None, None) #ctc layer=-1 will disable ctc

        return hs_pad, hs_mask.squeeze(-2)


