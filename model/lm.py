from copy import deepcopy
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

from .encoder import Encoder
from .decoder import Decoder


class LM(nn.Module):
    """Language model that predict next word based on previous N-1 grams(tokens)"""
    def __init__(
        self, 
        N, # Ngram
        v_size, 
        embedding_size, 
        t2v_map, 
        # hidden layers used to predict the embedding of next 
        # word by previous embedding, also used in encoder and 
        # decoder
        hidden_sizes=[], 
        activation=nn.ReLU,
        device='cuda', 
        simple=False, # set this to true to use simplified lm
    ):
        super().__init__()

        # include the None
        v_size = v_size + 1

        if simple:
            embedding_size = v_size
            hidden_sizes = []

        self.N = N
        self.v_size = v_size
        self.embedding_size = embedding_size
        # TODO: change more refined treatment
        t2v_map[self.v_size-1] = ''
        self.t2v_map = t2v_map
        self.hidden_sizes = hidden_sizes
        self.layers = [self.embedding_size] + hidden_sizes + [self.embedding_size]
        self.device = device
        self.simple = simple

        m = [
            [nn.Linear(i, j), activation()] for i, j in
            zip(self.layers[:-1], self.layers[1:])
        ]
        # from https://stackoverflow.com/questions/952914/how-do-i-make-a-flat-list-out-of-a-list-of-lists
        m = [i for l in m for i in l]

        if simple:
            self.model = lambda x: x
        else:
            self.model = nn.Sequential(*m)
            self.model = self.model.to(self.device)

        self.encoder = Encoder(N, v_size, 
                               embedding_size, 
                               hidden_sizes, 
                               activation, 
                               device=device, 
                               simple=simple)
        self.decoder = Decoder(v_size, 
                               embedding_size, 
                               hidden_sizes, 
                               activation, 
                               device=device, 
                               simple=simple)


    def forward(self, x, mask):
        '''Map: tensor(B, N) -> tensor(B, V)'''
        x = x.to(self.device)

        x = self.encoder(x, mask)
        x = x.mean(1)
        x = self.model(x)
        return x

    def decode_token(self, x):
        assert x.shape[0] == 1, "TODO: add to multi-batch sampling"
        x = x[0]
        x = [self.t2v_map[i.item()] for i in x]
        x = ''.join(x)
        return x

    def generate(self, x=None, max_len=64):
        '''generate text, only one as batch size'''
        if x == None:
            # randomly sample the init
            x = torch.randint(0, self.v_size-1, [1]).to(self.device).long()
            x = x.view(-1, x.shape[0])

        # Truncate the sequence
        if x.shape[-1] >= self.N-1:
            x = x[:, x.shape[1]-self.N+1: x.shape[1]]

        x_len = x.shape[-1]
        assert x_len + max_len > self.N, "TODO"
        generated = torch.zeros(max_len).to(self.device).long().view(-1, max_len)
        generated = torch.cat([x, generated], -1)
        mask = torch.ones([x.shape[0], self.N], dtype=torch.bool).to(self.device)
        mask[:, :x_len] = False

        for i in range(x_len, max_len+x_len):
            if i < self.N:
                mask[:, i-1] = False

            if i < self.N:
                i1, i2 = 0, self.N
            else:
                i1, i2 = i-self.N+1, i+1
            token = self(generated[:, i1: i2], mask)
            token = self.decoder(token)
            dist = torch.distributions.Categorical(token)
            generated[:, i] = dist.sample()
        generated = self.decode_token(generated)
        return generated

if __name__ == '__main__':
    x = torch.tensor([
        [1], 
        [0], 
        [0]])
    lm = LM(2, 2, 2)
    print(lm(x).shape)
    lm.generate()
