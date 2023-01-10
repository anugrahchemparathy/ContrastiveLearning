import torch
import torch.nn as nn
import torch.nn.functional as F

def byol_loss_alt(x1, x2):
    x1 = F.normalize(x1, dim=-1, p=2) 
    x2 = F.normalize(x2, dim=-1, p=2)
    return torch.mean(2 - 2 * (x2* x1).sum(dim=-1))
  
def byol_loss(x1, x2):
  # expects each to be n x d tensors 
  # n is the number of samples
  # d is the dimension of the embedding
  x1 = F.normalize(x1, dim = 1)
  x2 = F.normalize(x2, dim = 1)

  return torch.mean(torch.norm(x1 - x2, dim = 1)**2)