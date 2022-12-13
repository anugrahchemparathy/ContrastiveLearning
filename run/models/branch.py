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

# projectionHead
# implementing the projection head described in simclr paper

class projectionHead(nn.Module):
    def __init__(self, head_size = 4, hidden_size = 64):
        super().__init__()

        layers = [
            nn.Linear(head_size, hidden_size),
            nn.ReLU(inplace = True),
            # nn.Linear(hidden_size, hidden_size),
            # nn.ReLU(inplace = True),
            nn.Linear(hidden_size, head_size)
        ]

        self.head = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.head(x)

class predictor(nn.Module):
    def __init__(self, size, hidden_size = 64):
        super().__init__()

        layers = [
            nn.Linear(size, hidden_size),
            nn.ReLU(inplace = True),
            nn.Linear(hidden_size, size)
        ]

        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)