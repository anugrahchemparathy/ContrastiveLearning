from ldcl.plot.plot import VisPlot
from ldcl.plot.embed import embed
from ldcl.plot.color import get_cmap

from ldcl.data.physics import get_dataset
from ldcl.tools.device import get_device

from sklearn.decomposition import PCA
import argparse

import subprocess
import math

import torch
import torchvision
import torchvision.transforms as T
import torch.nn.functional as F
import numpy as np

from tqdm import tqdm

from ldcl.models.cifar_resnet import Branch

import matplotlib.pyplot as plt

import faiss

device = get_device(idx=0)

normalize = T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
single_transform = T.Compose([T.ToTensor(), normalize])


def eval_loop(croplow, epochs, idd, train_loader, test_loader, ind=None):
    branch = Branch()

    state_dict = torch.load(f"../cifar10_ckpts/011923_epochs_{epochs}_croplow_{croplow:.1f}/{idd}.pth", map_location=device)["state_dict"]
    branch.load_state_dict(state_dict)
    branch.to(device)
    branch.eval()
    encoder = branch.encoder

    embeds = []
    lbls = []
    for idx, (images, labels) in enumerate(test_loader):
        if idx > 50:
            print("Dataset cuts off after fifty iterations for efficiency")
            break
        with torch.no_grad():
            b = encoder(images.cuda())
            embeds.append(b)
            lbls.append(labels)
    embeds = torch.cat(embeds)
    lbls = torch.cat(lbls)

    cutoff = math.floor(embeds.shape[0] * 0.1)
    index = faiss.IndexFlatL2(512) 
    fembeds = embeds.detach().cpu().numpy() 
    fembeds = fembeds / np.sqrt(np.sum(np.square(fembeds), axis=1))[:, np.newaxis]
    index.add(fembeds)
    _, ids = index.search(fembeds, cutoff)
    row_idx = np.repeat(np.arange(len(fembeds))[:, np.newaxis], cutoff, axis=1)

    adj_matrix = np.zeros((len(fembeds), len(fembeds)), dtype=bool)
    adj_matrix[row_idx.flatten(), ids.flatten()] = True

    return adj_matrix, lbls.detach().cpu().numpy()

def matrix_similarity(mat1, mat2):
    assert(mat1.shape == mat2.shape)
    return np.sum(np.logical_and(mat1, mat2)) / (np.sum(mat1[0]) * mat1.shape[1])

def main_plot(args):
    # dataset
    train_transform = T.Compose([
        T.RandomResizedCrop(32, interpolation=T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalize
    ])
    test_transform = T.Compose([
        T.Resize(36, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(32),
        T.ToTensor(),
        normalize
    ])

    train_loader = torch.utils.data.DataLoader(
        dataset=torchvision.datasets.CIFAR10('../../data', train=True, transform=test_transform, download=True),
        shuffle=False,
        batch_size=256,
        pin_memory=True,
        num_workers=16,
        drop_last=True
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=torchvision.datasets.CIFAR10('../../data', train=False, transform=test_transform, download=True),
        shuffle=False,
        batch_size=256,
        pin_memory=True,
        num_workers=16
    )

    skip_factor = 25
    num_e = 1600
    results = np.zeros((9,4,int(num_e / skip_factor)))
    for it1, E in enumerate([25,50,75,100,150,200,400,800,1600]):
        adj1, lbl1 = eval_loop(0.2, 1600, E, train_loader, test_loader)
        for it2, n in enumerate([0.2,0.5,0.8,1.0]):
            print(f"Experiment {E} x {n}")
            for it3, e in enumerate(tqdm(range(0,num_e,skip_factor))):
                f = e + skip_factor
                adj2, lbl2 = eval_loop(n, num_e, f, train_loader, test_loader)
                results[it1, it2, it3] = matrix_similarity(adj1, adj2)
                print(E, n, f, results[it1, it2, it3])
                np.save("cifar_results", results)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    args = parser.parse_args()
    main_plot(args)
