import torch
import torch.nn as nn
import torch.nn.functional as F

def variance(z, gamma = 1, epsilon = 0.0001):
    """
    z: n x d tensor where n is the batch size and d is the embedding dimension
    gamma: constant target value for standard deviation, fixed to 1 in paper
    epsilon: small scalar for preventing numerical instabilities
    """
    var_z = torch.var(z, dim = 0, unbiased = True)
    
    reg_std = torch.sqrt(var_z + epsilon)
    
    loss = torch.mean(nn.ReLU(inplace = True)(gamma - reg_std))
    
    return loss

def covariance(z):
    """
    z: n x d tensor where n is the batch size and d is the embedding dimension
    """
    n,d = z.shape
    cov_z = torch.cov(z.T)
    zero_diag = cov_z.fill_diagonal_(0)
    
    return 1/d * torch.sum(torch.square(zero_diag))

def invariance(z1, z2):
    """
    z1: n x d tensor where n is the batch size and d is the embedding dimension
    z2: n x d tensor where n is the batch size and d is the embedding dimension
    """
    
    diff = z1 - z2
    
    square_norm = torch.sum(torch.square(diff), dim = 1, keepdim = False)
    
    return torch.mean(square_norm)


def vicreg_loss(z1, z2, l, m, n = 1, gamma = 1, epsilon = 0.0001):
    return l * invariance(z1,z2) + m * (variance(z1, gamma, epsilon) + variance(z2, gamma, epsilon)) + n * (covariance(z1) + covariance(z2))