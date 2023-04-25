import argparse
import os

import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter

from tools import load, Dataset
from model import LM

def parse():
    parser = argparse.ArgumentParser(
        description="Train the Ngram LM")
    # data
    parser.add_argument('--corpus', default="alice_in_wonderland.txt")
    parser.add_argument('--tokenizer', default='bpe')
    
    # training
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--verbose', type=int, default=1000)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--test', action='store_true', default=False)

    # model
    parser.add_argument('--n-gram', type=int, default=5)
    parser.add_argument('--embedding-size', type=int, default=64)
    parser.add_argument('--save-path', default="./log")

    args = parser.parse_args()
    return args

def train():
    '''train and save the model'''
    # prepare
    args = parse()
    writer = SummaryWriter(args.save_path)
    v_size, t2v_map, corpus = load(args.corpus, args.tokenizer)
    dataset = Dataset(corpus, args.batch_size, args.n_gram)
    lm = LM(N=args.n_gram, 
            v_size=v_size, 
            embedding_size=args.embedding_size, 
            t2v_map=t2v_map, 
            device=args.device, 
            simple=args.test)
    optim = torch.optim.AdamW(lm.parameters(), lr=args.lr)
    # loss_fn = nn.MultiLabelSoftMarginLoss()
    loss_fn = F.cross_entropy

    for epoch in range(args.epoch):
        for cnt, batch in enumerate(iter(dataset)):
            # iterate to use sub-sequence of the origin sequence to 
            # predict the next word
            losses = []
            batch = torch.tensor(batch).to(args.device)
            mask = torch.ones_like(batch, dtype=torch.bool, device=batch.device)
            for j in range(1, batch.shape[-1]):
                mask[:, j-1] = False
                Y = batch[:, j]
                pred = lm(batch, mask)
                loss = loss_fn(pred, Y)
                optim.zero_grad()
                loss.backward()
                optim.step()
                losses.append(loss.item())
            if cnt % args.verbose == 0:
                print(f"[*] In epoch: {epoch}, cnt: {cnt}, loss: {np.mean(losses).item():.3f}")
                writer.add_scalar("loss", np.mean(losses).item())
                lm = lm.eval()
                generated = lm.generate()
                print(f"\t Generated text: {generated}")
                lm = lm.train()
        torch.save(lm.state_dict(), os.path.join(args.save_path, f'model-{epoch}.pth'))

    torch.save(lm.state_dict(), os.path.join(args.save_path, f'model.pth'))

if __name__ == '__main__':
    train()
