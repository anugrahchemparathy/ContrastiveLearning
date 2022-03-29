import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal


from RealNVP_layer import RealNVP_Layer
import masks

class RealNVP(nn.Module):
    def __init__(self, input_output_size, hidden_size=12, num_layers = 5):
        super(RealNVP, self).__init__()

        self.input_output_size = input_output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        #save masks in a parameterlist so they can be saved with the model for reference
        self.masks = nn.ParameterList([nn.Parameter(torch.Tensor(mask).float(), requires_grad=False) for mask in masks.mask2(self.input_output_size, self.num_layers)])

        self.layers = nn.ModuleList([RealNVP_Layer(mask,self.input_output_size,self.hidden_size) for mask in self.masks])


        self.normal_distribution = MultivariateNormal(torch.zeros(self.input_output_size), torch.eye(self.input_output_size))

    def forward(self, x):
        """
        Takes in a point from the probability distribution and generates point on the orbit
        """
        output = x
        log_probability = 0
        for layer in self.layers:
            output, log_det_jacobian = layer(output)
            log_probability += log_det_jacobian
        

        return output, log_probability

    def log_probability(self, y):
        """
        param y: (batch_size, input_output_size) array
        """
        batch_size, _ = y.shape
        log_probability = torch.zeros(batch_size)

        for layer in reversed(self.layers):
            y, inverse_log_det_jacobian = layer.inverse(y)
            log_probability += inverse_log_det_jacobian
        
        log_probability += self.normal_distribution.log_prob(y)

        return log_probability