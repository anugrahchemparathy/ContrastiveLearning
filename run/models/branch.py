import torch
import torch.nn.functional as F
import torch.nn as nn


class branchEncoder(nn.Module):
    def __init__(self, encoder_in = 4, encoder_out = 4, encoder_hidden = 64):
        super().__init__()

        encoder_layers = [nn.Linear(encoder_in,encoder_hidden),
                            nn.BatchNorm1d(encoder_hidden),
                            nn.ReLU(inplace=True),
                            nn.Linear(encoder_hidden, encoder_hidden),
                            nn.BatchNorm1d(encoder_hidden),
                            nn.ReLU(inplace=True),
                            nn.Linear(encoder_hidden,encoder_hidden),
                            nn.BatchNorm1d(encoder_hidden),
                            nn.ReLU(inplace=True),
                            nn.Linear(encoder_hidden,encoder_out)
                            ]
        self.encoder = nn.Sequential(*encoder_layers)
        

        self.net = self.encoder

    def forward(self, x):
        return self.net(x)