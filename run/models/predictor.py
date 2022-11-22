import torch
import torch.nn.functional as F
import torch.nn as nn

class TopPredictor(nn.Module):
    def __init__(self, encoder, predictor=None, predictor_output = 3, fine_tuning = False, predictor_hidden = 64):
        super().__init__()

        if predictor:
            self.predictor = predictor
        else:
            predictor_layers = [nn.Linear(3,predictor_hidden),
                              nn.BatchNorm1d(predictor_hidden),
                              nn.ReLU(inplace=True),
                              nn.Linear(predictor_hidden, predictor_hidden),
                              nn.BatchNorm1d(predictor_hidden),
                              nn.ReLU(inplace=True),
                              nn.Linear(predictor_hidden,predictor_output),
                             ]
            self.predictor = nn.Sequential(*predictor_layers)
        
        self.encoder = encoder
        self.net = nn.Sequential(self.predictor,self.encoder)
        
        if not fine_tuning:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, x):
        return self.net(x)