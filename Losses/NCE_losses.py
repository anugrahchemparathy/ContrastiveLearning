import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def infoNCE(nn, p, temperature=0.1):
    nn = torch.nn.functional.normalize(nn, dim=1)
    p = torch.nn.functional.normalize(p, dim=1)

    logits = nn @ p.T
    logits /= temperature
    n = p.shape[0]
    labels = torch.arange(0, n, dtype=torch.long).to(device)
    loss = torch.nn.functional.cross_entropy(logits, labels)
    return loss

def rmseNCE(z1, z2, temperature=0.1):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    z1 = torch.unsqueeze(z1,0) #(nxr -> 1xnxr)
    z2 = torch.unsqueeze(z2,0)
    
    euclidean_dist = -torch.cdist(z1,z2,p=2.0) #p=2.0 for standard euclidean distance
    euclidean_dist = torch.squeeze(euclidean_dist)

    logits = euclidean_dist
    logits /= temperature
    n = z2.shape[1] #since its unsqueezed (1xnxr), the first real dimension is the second one
    labels = torch.arange(0, n, dtype=torch.long).to(device)
    loss = torch.nn.functional.cross_entropy(logits, labels)
    return loss

def normalmseNCE(z1, z2, temperature=0.1):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    z1 = torch.unsqueeze(z1,0) #(nxr -> 1xnxr)
    z2 = torch.unsqueeze(z2,0)
    
    euclidean_dist = torch.cdist(z1,z2,p=2.0) #p=2.0 for squared distance
    euclidean_dist = torch.squeeze(euclidean_dist)
    #print(euclidean_dist)
    squared_dist = -torch.square(euclidean_dist)
    #print(squared_dist)

    logits = squared_dist
    logits /= temperature
    n = z2.shape[1] #since its unsqueezed (1xnxr), the first real dimension is the second one
    labels = torch.arange(0, n, dtype=torch.long).to(device)
    loss = torch.nn.functional.cross_entropy(logits, labels)
    return loss