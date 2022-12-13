import torch
import torch.nn as nn
import torch.nn.functional as F

def simsiam(pa, pb, za, zb):
    """
    expects inputs to be tensors of shape (batch_sz, d)
    zi is the output of the encoder followed by the projector
    pi is the zi passed through the predictor h
    """
    za.detach()
    zb.detach()

    pa = torch.linalg.norm(pa, dim = 1)
    pb = torch.linalg.norm(pb, dim = 1)
    za = torch.linalg.norm(za, dim = 1)
    zb = torch.linalg.norm(zb, dim = 1)

    return -((pa * zb).sum(dim = 1).mean() + (pb * za).sum(dim = 1).mean())/2