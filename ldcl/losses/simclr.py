import torch
import torch.nn as nn
import torch.nn.functional as F




def NT_Xent_loss(za,zb, temperature = 0.1):
    """
    param za: (batch_size, output_size) : tensor of encoder outputs for inputs with augmentation
    param zb: (batch_size, output_size) : tensor of encoder outputs for corresponding inputs with augmentation
    """
    batch_size, output_size = za.shape

    za_norm = F.normalize(za, dim = 1)
    zb_norm = F.normalize(zb, dim = 1)


    z_norm = torch.cat([za_norm, zb_norm])

    total_logits = z_norm @ z_norm.T # (2 * batch_size, 2 * batch_size)
    total_logits /= temperature

    # remove diagonal elements entirely from the tensor
    total_logits = total_logits[torch.eye(2 * batch_size) == 0].view(2 * batch_size, -1) # (2 * batch_size, 2 * batch_size - 1)
    labels = torch.tensor([batch_size - 1 + i for i in range(batch_size)] + [i for i in range(batch_size)])

    loss = F.cross_entropy(total_logits, labels)
    return loss


def infoNCE(nn, p, temperature=0.1):
    """
    
    """
    nn = torch.nn.functional.normalize(nn, dim=1)
    p = torch.nn.functional.normalize(p, dim=1)

    logits = nn @ p.T
    logits /= temperature
    n = p.shape[0]
    labels = torch.arange(0, n, dtype=torch.long).cuda()
    loss = torch.nn.functional.cross_entropy(logits, labels)
    return loss

def NT_Xent_loss_other(z1, z2, temperature=0.5, idx_bad=None):
    """
    Alternative implementation of NT_Xent loss
    """
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    N, Z = z1.shape
    device = z1.device
    representations = torch.cat([z1, z2], dim=0)
    similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=-1)
    l_pos = torch.diag(similarity_matrix, N)
    r_pos = torch.diag(similarity_matrix, -N)
    positives = torch.cat([l_pos, r_pos]).view(2 * N, 1)
    diag = torch.eye(2 * N, dtype=torch.bool, device=device)
    diag[N:, :N] = diag[:N, N:] = diag[:N, :N]

    negatives = similarity_matrix[~diag].view(2 * N, -1)
    if torch.is_tensor(idx_bad):
        idx_bad_double = torch.cat([idx_bad, idx_bad], dim=0).unsqueeze(-1)
        positives *= idx_bad_double
        negatives *= idx_bad_double

    logits = torch.cat([positives, negatives], dim=1)
    logits /= temperature

    labels = torch.zeros(2 * N, device=device, dtype=torch.int64)

    loss = F.cross_entropy(logits, labels, reduction='sum')
    return loss / (2 * N)