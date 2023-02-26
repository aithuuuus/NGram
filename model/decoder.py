from copy import deepcopy
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

class Decoder(nn.Module):
    '''MLP that map embedding to one-hot, and number representations'''
    def __init__(
        self, 
        v_size, 
        embedding_size, 
        hidden_sizes=[32], 
        activation=nn.ReLU, 
        device='cuda'
    ):
        super().__init__()

        self.v_size = v_size
        self.embedding_size = embedding_size
        self.hidden_sizes = hidden_sizes
        self.layers = [self.embedding_size] + hidden_sizes + [self.v_size]
        self.device = device

        m = [
            [nn.Linear(i, j), activation()] for i, j in
            zip(self.layers[:-1], self.layers[1:])
        ]
        # from https://stackoverflow.com/questions/952914/how-do-i-make-a-flat-list-out-of-a-list-of-lists
        m = [i for l in m for i in l]
        m[-1] = nn.Softmax()

        self.model = nn.Sequential(*m)
        self.model = self.model.to(self.device)

    def forward(self, x):
        pass


if __name__ == '__main__':
    d = Decoder(2, 2)
