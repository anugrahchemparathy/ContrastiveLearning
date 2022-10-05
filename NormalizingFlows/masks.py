import torch
import torch.nn as nn
import torch.nn.functional as F

def mask1(input_output_size,num_layers):
    """
    A randomized tensor of binary masks (num_layers x input_output_size), ex:
    [[0,1,1,1,0,1],
     [1,1,0,1,1,0],
     [0,0,0,1,1,1]
    ]
    """
    return torch.randint(2,(num_layers,input_output_size)).float()

def mask2(input_output_size,num_layers):
    """
    An alternating tensor of binary masks (num_layers x input_output_size), ex:
    [[0,1,0,1,0,1],
     [1,0,1,0,1,0],
     [0,1,0,1,0,1]
    ]
    """
    return torch.tensor([[(i+j)%2 for i in range(input_output_size)] for j in range(num_layers)]).float()