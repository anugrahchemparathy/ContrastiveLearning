import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal


from RealNVP_layer import RealNVP_Layer
import masks

class RealNVP(nn.Module):
    def __init__(self, layer_dim, hidden_size=12, num_layers = 3):
        super(RealNVP, self).__init__()

        self.layer_dim = layer_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        #save masks in a parameterlist so they can be saved with the model for reference
        self.masks = nn.ParameterList([nn.Parameter(torch.Tensor(mask), requires_grad=False) for mask in masks.mask2(self.layer_dim, self.num_layers)])

        self.layers = nn.ModuleList([RealNVP_Layer(mask,self.layer_dim,self.hidden_size) for mask in self.masks])


        self.distribution = MultivariateNormal(torch.zeros(self.layer_dim), torch.eye(self.layer_dim))

    def forward(self, x):
        output = x
        log_probability = 0
        for layer in self.layers:
            output, log_det_jacobian = layer(output)
            log_probability += log_det_jacobian
        

        return output, log_probability

    def forward_sample(self, num_samples):
        inputs = self.distribution.sample((num_samples,))
        log_probability = self.distribution.log_prob(inputs)

        outputs = inputs
        for layer in self.layers:
            outputs, log_det_jacobian = layer(outputs)
            log_probability += log_det_jacobian
        

        return outputs, log_probability

    def log_probability(self, y):
        """
        param y: (batch_size, layer_dim) array
        """
        batch_size, _ = y.shape
        log_probability = torch.zeros(batch_size)

        for layer in reversed(self.layers):
            #print("layer, new y =", y[:5])
            y, inverse_log_det_jacobian = layer.inverse(y)
            log_probability += inverse_log_det_jacobian
        
        log_probability += self.distribution.log_prob(y)

        return log_probability