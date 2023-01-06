import torch
import torch.nn as nn
import torch.nn.functional as F

def simsiam(za, pb):
    """
    expects inputs to be tensors of shape (batch_sz, d)
    zi is the output of the encoder followed by the projector
    pi is the zi passed through the predictor h
    """
    za_ = za.detach()

    #print(za)
    #print(pb)
    #zb.detach()

    #pa = torch.linalg.norm(pa, dim = 1)
    pb = F.normalize(pb, dim = 1)
    za_ = F.normalize(za_, dim = 1)
    #zb = torch.linalg.norm(zb, dim = 1)

    return -(pb * za_).sum(dim = 1).mean()
