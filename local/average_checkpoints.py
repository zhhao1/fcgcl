#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os

import numpy as np

import torch

def main(args):
    avg = None
    # sum
    for path in args.model:
        states = torch.load(path, map_location=torch.device("cpu"))['state_dict']
        if avg is None:
            avg = states
        else:
            for k in avg.keys():
                avg[k] += states[k]

    # average
    for k in avg.keys():
        if not 'queue' in k and avg[k] is not None:
            avg[k] /= args.num

    torch.save(avg, args.out)


def get_parser():
    parser = argparse.ArgumentParser(description="average models from top n model")
    parser.add_argument("--model", required=True, type=str, nargs="+")
    parser.add_argument("--out", required=True, type=str)
    parser.add_argument("--num", default=10, type=int)
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args)

