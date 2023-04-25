import argparse
import os
import json

import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

from tools import load, Dataset
from model import LM

def parse():
    parser = argparse.ArgumentParser(
        description="Generate")
    # dir
    parser.add_argument('--path', default="./log")
    parser.add_argument('--model-name', default="model-200.pth")
    parser.add_argument('--len', type=int, default=128)
    args = parser.parse_args()
    return args

def generate():
    '''train and save the model'''
    # prepare
    args = parse()
    with open(os.path.join(args.path, "config.json"), 'r') as f:
        m_args = json.load(f)

    tokenizer = Tokenizer(BPE(unk_token='[UNK]'))
    trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer = tokenizer.from_file(os.path.join(args.path, 'tokenizer.json'))
    lm = LM(N=m_args["n_gram"], 
            v_size=tokenizer.get_vocab_size(), 
            embedding_size=m_args["embedding_size"], 
            t2v_map=tokenizer, 
            device=m_args["device"], 
            simple=m_args["test"])
    lm.load_state_dict(torch.load(os.path.join(args.path, "model-200.pth")))
    lm = lm.eval()
    generated = lm.generate(max_len=args.len)
    print(f" Generated text: {generated}")

if __name__ == '__main__':
    generate()
