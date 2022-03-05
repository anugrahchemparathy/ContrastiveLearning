import torch
import torch.nn as nn
import torch.nn.functional as F

from RealNVP_layer import RealNVP_Layer
import masks

class RealNVP(nn.Module):
    def __init__(self, input_output_size, hidden_size=12, num_layers = 5):
        super(RealNVP, self).__init__()

        self.input_output_size = input_output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        #save masks in a parameterlist so they can be saved with the model for reference
        self.masks = nn.ParameterList([nn.parameter(mask) for mask in masks.mask2(self.input_output_size, self.num_layers)])

        self.layers = nn.ModuleList([RealNVP_Layer(mask,self.input_output_size,self.hidden_size) for mask in self.masks])

        