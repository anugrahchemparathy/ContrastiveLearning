import torch
import torch.nn as nn
import torch.nn.functional as F

def LanguageEmbeddingLoss(encodings, temperature = 1.0, method = "auto"):
    """
    param encodings: (batch_size, output_size) : tensor of outputs from language model
    param temperature: optional numerical parameter
    param method: optional string to play around with manually computing diagonal sum of negative log likelihoods

    loss function for SSL on sentence data
    pushes encodings for different sentences apart
    """

    encodings = F.normalize(encodings)

    batch_size, output_size = encodings.shape
    
    logits = encodings @ encodings.T
    logits = logits / temperature


    if method == "auto":
        labels = torch.arange(0, batch_size, dtype=torch.long).cuda()
        neg_log_likelihood = F.cross_entropy(logits, labels)

    if method == "manual":
        lsm = nn.LogSoftmax(dim = 1)
        lsm_logits = -lsm(logits)
        target_mask = torch.eye(batch_size)

        neg_log_likelihood = torch.sum(lsm_logits * target_mask)

    return neg_log_likelihood