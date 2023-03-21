from copy import deepcopy
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

class Decoder(nn.Module):
    '''MLP that map embedding to probability of the next word'''
    def __init__(
        self, 
        v_size, 
        embedding_size, 
        hidden_sizes=[32], 
        activation=nn.ReLU, 
        device='cuda', 
        simple=False, 
    ):
        super().__init__()

        self.v_size = v_size
        self.embedding_size = embedding_size
        self.hidden_sizes = hidden_sizes
        self.layers = [self.embedding_size] + hidden_sizes + [self.v_size]
        self.device = device
        self.simple = simple

        m = [
            [nn.Linear(i, j), activation()] for i, j in
            zip(self.layers[:-1], self.layers[1:])
        ]
        # from https://stackoverflow.com/questions/952914/how-do-i-make-a-flat-list-out-of-a-list-of-lists
        m = [i for l in m for i in l]

        if self.simple:
            m = [nn.Softmax(dim=-1)]

        self.model = nn.Sequential(*m[:-1])
        self.model = self.model.to(self.device)

    def forward(self, x):
        '''Map: tensor(B, E) -> tensor(B, V), i.e. from embedding to prob w.r.t. words'''

        x = x.to(self.device)
        x = self.model(x)
        return x



if __name__ == '__main__':
    x = torch.tensor([
        [1, 0], 
        [0, 0], 
        [0, 0]], dtype=torch.float32)

    d = Decoder(2, 2)
    print(d(x))
