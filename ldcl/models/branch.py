import torch
import torch.nn.functional as F
import torch.nn as nn


class branchEncoder(nn.Module):
    def __init__(self, encoder_in = 4, encoder_out = 3, encoder_hidden = 64, num_layers = 4, useBatchNorm = False, activation = nn.ReLU(inplace=True)):
        super().__init__()
        self.num_layers = num_layers
        self.bn = useBatchNorm
        self.activation = activation


        encoder_layers = [nn.Linear(encoder_in,encoder_hidden)]

        for i in range(self.num_layers - 2):
            if self.bn: encoder_layers.append(nn.BatchNorm1d(encoder_hidden))
            # encoder_layers.append(nn.ReLU(inplace=True))
            encoder_layers.append(self.activation)
            encoder_layers.append(nn.Linear(encoder_hidden, encoder_hidden))

        if self.bn: encoder_layers.append(nn.BatchNorm1d(encoder_hidden))
        encoder_layers.append(self.activation)
        # encoder_layers.append(nn.ReLU(inplace=True))
        encoder_layers.append(nn.Linear(encoder_hidden, encoder_out))


        self.encoder = nn.Sequential(*encoder_layers)

    def forward(self, x):
        return self.encoder(x)

# projectionHead
# implementing the projection head described in simclr paper

class projectionHead(nn.Module):
    def __init__(self, head_in = 3, head_out = 4, hidden_size = 64):
        super().__init__()

        layers = [
            nn.Linear(head_in, hidden_size),
            nn.ReLU(inplace = True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace = True),
            nn.Linear(hidden_size, head_out)
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