#!/usr/bin/env python3
# encoding: utf-8

# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""End-to-end speech translation model decoding script."""

import logging
import os
import random
import sys
import json
import yaml

import argparse
import configargparse
import numpy as np
import torch

from bins.transformer_st import E2E

from tools.function import test_init, load_trained, load_train_temp

from espnet.asr.asr_utils import add_results_to_json
from espnet.utils.cli_utils import strtobool
from espnet.utils.io_utils import LoadInputsAndTargets
from espnet.utils.deterministic_utils import set_deterministic_pytorch

def get_parser():
    """Get default arguments."""
    parser = configargparse.ArgumentParser(
        description="Translate text from speech using a speech translation "
        "model on one CPU or GPU",
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
    )
    # general configuration
    parser.add("--config", is_config_file=True, help="Config file path")

    parser.add_argument("--ngpu", type=int, default=0, help="Number of GPUs")

    parser.add_argument("--debugmode", type=int, default=1, help="Debugmode")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--verbose", "-V", type=int, default=1, help="Verbose option")
    parser.add_argument("--batchsize", type=int, default=1, help="Batch size for beam search (0: means no batch processing)",)
    parser.add_argument("--preprocess-conf", type=str, default=None, help="The configuration file for the pre-processing",)
    #decode ctc shrink 
    parser.add_argument("--decode-shrink",  default=False, type=strtobool, help="haparam file")
    # task related
    parser.add_argument("--initilize-module", default=None, type=str, nargs="+")
    parser.add_argument("--hparam", type=str, required=True, help="haparam file")
    parser.add_argument("--trans-json", type=str, help="Filename of translation data (json)")
    parser.add_argument("--result-label", type=str, required=True, help="Filename of result label data (json)",)

    # model (parameter) related
    parser.add_argument("--prefix", type=str, required=True, help="Model file parameters prefix")
    parser.add_argument("--model", type=str, required=True, help="Model file parameters to read")

    # search related
    parser.add_argument("--nbest", type=int, default=1, help="Output N-best hypotheses")
    parser.add_argument("--beam-size", type=int, default=1, help="Beam size")
    parser.add_argument("--penalty", type=float, default=0.0, help="Incertion penalty")
    parser.add_argument("--maxlenratio", type=float, default=0.0, help="""Input length ratio to obtain max output length.
                                                                       If maxlenratio=0.0 (default), it uses a end-detect function
                                                                       to automatically find maximum hypothesis lengths""",)
    parser.add_argument("--minlenratio", type=float, default=0.0, help="Input length ratio to obtain min output length",)

    # rnnlm related
    parser.add_argument("--rnnlm", type=str, default=None, help="RNNLM model file to read")
    parser.add_argument("--rnnlm-conf", type=str, default=None, help="RNNLM model config file to read")
    parser.add_argument("--word-rnnlm", type=str, default=None, help="Word RNNLM model file to read")
    parser.add_argument("--word-rnnlm-conf", type=str, default=None, help="Word RNNLM model config file to read",)
    parser.add_argument("--word-dict", type=str, default=None, help="Word list to read")
    parser.add_argument("--lm-weight", type=float, default=0.1, help="RNNLM weight")

    # test task
    parser.add_argument("--task", default='ST', type=str, choices=["ST", "MT"],help="task be executed when testing",)

    return parser

def trans(args):
    """Decode with the given args.

    Args:
        args (namespace): The program arguments.

    """
    set_deterministic_pytorch(args)
       
    #load trained model
    idim, odim, trained_args = test_init(args)
    trained_args.ctc_shrink = args.decode_shrink
    trained_args.preprocess_conf = None
    model = E2E(idim, odim, trained_args)
    load_trained(args.model, model=model, prefix=args.prefix, initilize_module=args.initilize_module)
    # gpu
    if args.ngpu == 1:
        gpu_id = list(range(args.ngpu))
        logging.info("gpu id: " + str(gpu_id))
        model = model.cuda()
    # read json data
    with open(args.trans_json, "rb") as f:
        js = json.load(f)["utts"]
    new_js = {}

    load_inputs_and_targets = LoadInputsAndTargets(
        mode="asr",
        load_output=False,
        sort_in_input_length=False,
        preprocess_conf=trained_args.preprocess_conf
        if args.preprocess_conf is None
        else args.preprocess_conf,
        preprocess_args={"train": False},
    )

    with torch.no_grad():
        for idx, name in enumerate(js.keys(), 1):
            logging.info("(%d/%d) decoding " + name, idx, len(js.keys()))
            batch = [(name, js[name])]
            if args.task == 'ST':
                feat = load_inputs_and_targets(batch)[0][0]
                if args.ngpu == 1:
                    feat = torch.tensor(feat).cuda()
            else:
                feat = [js[name]["output"][1]["tokenid"].split()]  
            nbest_hyps = model.translate(feat, args, trained_args.char_list, task=args.task)
            new_js[name] = add_results_to_json(
                js[name], nbest_hyps, trained_args.char_list
                )

    with open(args.result_label, "wb") as f:
        f.write(
            json.dumps(
                {"utts": new_js}, indent=4, ensure_ascii=False, sort_keys=True
            ).encode("utf_8")
        )

def main(args):
    """Run the main decoding function."""
    parser = get_parser()
    args = parser.parse_args(args)

    # logging info
    if args.verbose == 1:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    elif args.verbose == 2:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    else:
        logging.basicConfig(
            level=logging.WARN,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
        logging.warning("Skip DEBUG/INFO messages")

    # check CUDA_VISIBLE_DEVICES

    # seed setting
    random.seed(args.seed)
    np.random.seed(args.seed)
    logging.info("set random seed = %d" % args.seed)

    # trans
    trans(args)


if __name__ == "__main__":
    main(sys.argv[1:])
