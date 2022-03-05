import torch
import torch.nn as nn
import torch.nn.functional as F


NUM_LAYERS = 3

class RealNVP_Layer(nn.Module):
    def __init__(self, mask, input_output_size, hidden_size,):
        super(RealNVP_Layer).__init__()

        self.mask = torch.randint(2,(input_output_size,))
        self.input_output_size = input_output_size
        self.hidden_size = hidden_size


        self.scale_func = nn.Sequential(self.generate_network(1))
        self.scale_factor = nn.Parameter(torch.Tensor(input_output_size))

        self.translate_func = nn.Sequential(self.generate_network(1))

    def generate_network(self,intermediate_layers):
        """
        Generates a simple fully connected network with intermediate_layers + 2 layers
        Uses LeakyReLU activations
        """
        modules = []
        modules.append(nn.Linear(in_features=self.input_output_size, out_features=self.hidden_size))
        modules.append(nn.LeakyReLU)
        for layer_i in intermediate_layers:
            modules.append(nn.Linear(in_features=self.hidden_size,out_features=self.hidden_size))
            modules.append(nn.LeakyReLU)
        modules.append(nn.Linear(in_features=self.hidden_size,out_features=self.input_output_size))

        return modules

    def forward(self, x):
        """
        param: x : [batch_size, input_output_size] vector of inputs
        returns: y : [batch_size, input_output_size] vector of outputs from coupling layer
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
        log_det_jacobian = torch.sum((1-self.mask) * sx_1,-1) #need to multiply by 1-self.mask to re-zero the 0 terms that got exponentiated to 1
        
        return y, log_det_jacobian
    
    def inverse(self, y):
        """
        param: y : [batch_size, input_output_size] vector of outputs to be inverted
        returns: x : [batch_size, input_output_size] vector of inputs that would produce the output
        returns: log_det_jacobian: logarithm of the determinant of the jacobian for this coupling layer for the given input
        """
        y_1 = self.mask * y # y_{1:d}
        y_2 = (1-self.mask) * y # y_{d+1:D}

        sy_1 = (1-self.mask) * self.scale_func(y_1) * self.scale_factor # s(y_{1:d})
        ty_1 = (1-self.mask) * self.translate_func(y_1) # t(y_{d+1,D})

        x_1 = y_1 #x_{1:d}
        x_2 = (y_2 - ty_1) * torch.exp(-sy_1) #x_{d+1:D}
        x = x_1 + x_2

        log_det_jacobian = torch.sum((1-self.mask) * (-sy_1),-1)

        return x, log_det_jacobian
