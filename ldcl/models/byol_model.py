import copy
import random
from functools import wraps

import torch
from torch import nn
import torch.nn.functional as F

from torchvision import transforms as T

# helper functions

def model_stop_grad(model):
    for p in model.parameters():
        p.requires_grad = False

class moving_average_network(nn.Module):
  def __init__(self, source_network, beta = 0.99):
    super().__init__()
    self.beta = beta
    
    self.source_network = source_network
    self.moving_average_network = copy.deepcopy(self.source_network)
    model_stop_grad(self.moving_average_network)
    
    pass

  def update_weights(self):
    # updates the weights
    for source_params, moving_average_params in zip(self.source_network.parameters(), self.moving_average_network.parameters()):
        update_weight, old_weight = source_params.data, source_params.data
        moving_average_params.data = old_weight * self.beta + (1 - self.beta) * update_weight

  def forward(self, input):
    output = self.moving_average_network(input)
    return output

