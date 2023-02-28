import argparse

import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

from tools import load, Dataset

def parse():
    parser = argparse.ArgumentParser(
        description="Train the Ngram LM")
    parser.add_argument('--corpus', default="alice_in_wonderland.txt")
    parser.add_argument('--tokenizer', default='character')

    args = parser.parse_args()
    return args

def train():
    '''train and save the model'''
    args = parse()
    v_size, t2v_map, corpus = load(args.corpus, args.tokenizer)
    dataset = Dataset(corpus)

if __name__ == '__main__':
    train()
