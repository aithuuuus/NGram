from copy import deepcopy
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

class Encoder(nn.Module):
    """MLP That map encoded tensor of natures numbers into word embeddings"""
    def __init__(
        self, 
        N, # Ngram
        v_size, 
        embedding_size, 
        hidden_sizes=[], 
        activation=nn.ReLU,
        device='cuda', 
        simple=False, # set this to true to return the one-hot
    ):
        super().__init__()

        self.N = N
        self.simple = simple

        self.v_size = v_size
        self.embedding_size = embedding_size
        self.hidden_sizes = hidden_sizes
        self.layers = [self.embedding_size] + hidden_sizes + [self.embedding_size]
        self.device = device
        self.embedding = nn.Embedding(v_size, embedding_size).to(self.device)
        # The brute-force position embedding
        self.position_embedding = nn.Embedding(N, embedding_size).to(self.device)
        if self.simple:
            return

        m = [
            [nn.Linear(i, j), activation()] for i, j in
            zip(self.layers[:-1], self.layers[1:])
        ]
        # from https://stackoverflow.com/questions/952914/how-do-i-make-a-flat-list-out-of-a-list-of-lists
        m = [i for l in m for i in l]

        self.model = nn.Sequential(*m[:-1])
        self.model = self.model.to(self.device)


    def forward(self, x, mask):
        '''Map: tensor(B, N) -> tensor(B, N, E)'''
        # one hot encoding
        x = x.masked_fill(mask, self.v_size-1)
        x = x.to(self.device)
        x = self.embedding(x)
        po = self.position_embedding(torch.arange(x.shape[1]).to(self.device))[None,  :]
        x = x + po

        if not self.simple:
            x = self.model(x)
        return x


if __name__ == '__main__':
    x = torch.tensor([
        [1, 0], 
        [0, 0], 
        [0, 0]])
    encoder = Encoder(2, 2, 2)
    print(encoder(x).shape)
