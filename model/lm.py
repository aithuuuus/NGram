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
        hidden_sizes=[32], 
        activation=nn.ReLU,
        device='cuda', 
        simple=True, # set this to true to use simplified lm
    ):
        super().__init__()

        assert N == 2, 'only support bigram now!'

        if simple:
            embedding_size = v_size
            hidden_sizes = []

        self.N = N
        self.v_size = v_size
        self.embedding_size = embedding_size
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

        self.model = nn.Sequential(*m[:-1])
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


    def forward(self, x):
        '''Map: tensor(B, N) -> tensor(B, V)'''
        # one hot encoding
        x = x.to(self.device)

        x = self.encoder(x)
        # use mean pooling
        x = x.mean(1)
        x = self.model(x)
        x = self.decoder(x)
        return x

    def decode_token(self, x):
        x = [self.t2v_map[i.item()] for i in x]
        x = ''.join(x)
        return x

    def generate(self, x=None, max_len=64):
        '''generate text, only one as batch size'''
        if x == None:
            # randomly sample the init
            x = torch.randint(0, self.v_size, [1]).to(self.device).long()
        x_len = len(x)
        generated = torch.zeros(max_len).to(self.device).long()
        generated = torch.cat([x, generated], 0)

        # TODO: use bigram for generating, should be
        for i in range(max_len):
            token = self(generated[None, x_len+i-1].view(1, -1)).argmax(-1)
            generated[x_len+i] = token[0]
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
