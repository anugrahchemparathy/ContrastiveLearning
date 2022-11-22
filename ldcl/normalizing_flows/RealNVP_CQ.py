import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal


from RealNVP_layer import RealNVP_Layer
import masks


class RealNVP_Layer_Augmented(nn.Module):
    def __init__(self, mask, layer_dim, hidden_size):
        super(RealNVP_Layer_Augmented, self).__init__()
        """
        layer that takes in x_{1:D} and returns y_{1:D}

        layer_dim: dimension D
        hidden_size: the hidden size of a layer.        
        """

        self.mask = mask
        self.layer_dim = layer_dim
        self.hidden_size = hidden_size


        self.scale_func = nn.Sequential(*self.generate_network(1))
        self.scale_factor = nn.Parameter(torch.Tensor(layer_dim).float())

        self.translate_func = nn.Sequential(*self.generate_network(1))

    def generate_network(self,intermediate_layers):
        """
        Generates a simple fully connected network with intermediate_layers + 2 layers
        Uses LeakyReLU activations
        """
        modules = []
        modules.append(nn.Linear(in_features=self.layer_dim + 3, out_features=self.hidden_size))
        modules.append(nn.LeakyReLU())
        for _ in range(intermediate_layers):
            modules.append(nn.Linear(in_features=self.hidden_size,out_features=self.hidden_size))
            modules.append(nn.LeakyReLU())
        modules.append(nn.Linear(in_features=self.hidden_size,out_features=self.layer_dim))

        return modules

    def forward(self, x):
        """
        layer_dim = D
        param: x : [batch_size, layer_dim] vector of inputs
        returns: y : [batch_size, layer_dim] vector of outputs from coupling layer
        returns: log_det_jacobian: logarithm of the determinant of the jacobian for this coupling layer for the given input
        """
        x_1 = self.mask * x  # x_{1:d}
        x_2 = (1-self.mask) * x # x_{d+1:D}


        #strictly speaking don't need to multiply 1-self.mask for sx_1, since when we multiply by x_2
        #all the 1:d terms get zero'd out anyways
        sx_1 = (1-self.mask) * self.scale_func(x_1)  * self.scale_factor #s(x_{1:d}) -> {d+1:D}
        tx_1 = (1-self.mask) * self.translate_func(x_1) #t(x_{1:d}) -> {d+1:D}

        """
        Following output vectors are still D dimensional, but all 
        components not in the corresponding mask are 0

        in computing y_2, torch.exp(sx_1) produces 1s in the 1:d locations since e^0 = 1
        however, x_2 is already zero'd in these locations so it's fine
        """
        y_1 = x_1 #y_{1:d}
        y_2 = x_2 * (torch.exp(sx_1)) + tx_1 # y_{d+1:D}
    
        y = y_1 + y_2
        log_det_jacobian = torch.sum(sx_1, dim = -1)
        
        return y, log_det_jacobian # [(batch_size, D), (batch_size)]
    
    def inverse(self, y):
        """
        param: y : [batch_size, layer_dim] vector of outputs to be inverted
        returns: x : [batch_size, layer_dim] vector of inputs that would produce the output
        returns: log_det_jacobian: logarithm of the determinant of the jacobian for this coupling layer for the given input
        """
        y_1 = self.mask * y # y_{1:d}
        y_2 = (1-self.mask) * y # y_{d+1:D}

        sy_1 = (1-self.mask) * self.scale_func(y_1) * self.scale_factor # s(y_{1:d})
        ty_1 = (1-self.mask) * self.translate_func(y_1) # t(y_{d+1,D})

        x_1 = y_1 #x_{1:d}
        x_2 = (y_2 - ty_1) * torch.exp(-sy_1) #x_{d+1:D}
        x = x_1 + x_2

        inverse_log_det_jacobian = -torch.sum(sy_1, dim = -1)

        return x, inverse_log_det_jacobian # [(batch_size, D), (batch_size)]


class RealNVP_Augmented(nn.Module):
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

    def forward_sample(self, num_samples, conserved_quantities):
        """
        num_samples: the number of points to generate
        conserved_quantities: torch tensor of [phi0,H,L]
        """
        inputs = self.distribution.sample((num_samples,)) # (num_samples, self.layer_dim)
        log_probability = self.distribution.log_prob(inputs)

        outputs = inputs
        conserved_quantities = conserved_quantities[:,None] #unsqueeze for concatenation broadcasting
        for layer in self.layers:
            outputs, log_det_jacobian = layer(outputs, conserved_quantities)
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