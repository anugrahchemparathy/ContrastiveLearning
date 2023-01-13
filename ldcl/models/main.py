import os
import math
import time
import copy
import argparse

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

import torchvision
import torchvision.transforms as T

from .resnet import resnet18

class ProjectionMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden_dim, bias=False),
                                 nn.BatchNorm1d(hidden_dim),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(hidden_dim, out_dim, bias=False),
                                 nn.BatchNorm1d(out_dim, affine=False))

    def forward(self, x):
        return self.net(x)


class PredictionMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class Branch(nn.Module):
    def __init__(self, encoder=None):
        super().__init__()
        dim_proj = [2048, 2048]
        if encoder:
            self.encoder = encoder
        else:
            self.encoder = resnet18()
        self.projector = ProjectionMLP(512, dim_proj[0], dim_proj[1])
        self.net = nn.Sequential(
            self.encoder,
            self.projector
        )

        """
        if args.loss == 'simclr':
            self.predictor2 = nn.Sequential(nn.Linear(512, 2048),
                                            nn.LayerNorm(2048),
                                            nn.ReLU(inplace=True),  # first layer
                                            nn.Linear(2048, 2048),
                                            nn.LayerNorm(2048),
                                            nn.ReLU(inplace=True),
                                            nn.Linear(2048, 4))  # output layer
        else:
            self.predictor2 = nn.Sequential(nn.Linear(512, 2048),
                                            nn.LayerNorm(2048),
                                            nn.ReLU(inplace=True),  # first layer
                                            nn.Linear(2048, 2048),
                                            nn.LayerNorm(2048),
                                            nn.Linear(2048, 4))  # output layer
        """

    def forward(self, x):
        return self.net(x)
