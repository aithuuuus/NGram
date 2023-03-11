import argparse

import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

from tools import load, Dataset
from model import LM

def parse():
    parser = argparse.ArgumentParser(
        description="Train the Ngram LM")
    # data
    parser.add_argument('--corpus', default="alice_in_wonderland.txt")
    parser.add_argument('--tokenizer', default='character')
    
    # training
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--verbose', type=int, default=10000)
    parser.add_argument('--device', default='cuda')
    
    # model
    parser.add_argument('--n-gram', type=int, default=2)
    parser.add_argument('--embedding-size', type=int, default=32)

    args = parser.parse_args()
    return args

def train():
    '''train and save the model'''
    # prepare
    args = parse()
    v_size, t2v_map, corpus = load(args.corpus, args.tokenizer)
    dataset = Dataset(corpus, args.batch_size, args.n_gram)
    lm = LM(N=args.n_gram, 
            v_size=v_size, 
            embedding_size=args.embedding_size, 
            device=args.device)
    optim = torch.optim.Adam(lm.parameters(), lr=args.lr)
    # loss_fn = nn.MultiLabelSoftMarginLoss()
    loss_fn = F.cross_entropy

    for epoch in range(args.epoch):
        for cnt, batch in enumerate(iter(dataset)):
            # iterate to use sub-sequence of the origin sequence to 
            # predict the next word
            losses = []
            for j in range(1, batch.shape[-1]):
                # TODO Change to use mask
                X = batch[:, 0:j]
                # Y = batch[:, [j]]
                Y = batch[:, j]
                X = torch.tensor(X).to(args.device)
                Y = torch.tensor(Y).to(args.device)
                pred = lm(X)
                loss = loss_fn(pred, Y)
                optim.zero_grad()
                loss.backward()
                optim.step()
                losses.append(loss.item())
            if cnt % args.verbose == 0:
                print(f"[*] In epoch: {epoch}, loss: {np.mean(losses).item():.3f}")

if __name__ == '__main__':
    train()
