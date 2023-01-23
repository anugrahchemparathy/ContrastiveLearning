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

import matplotlib.pyplot as plt

import faiss

device = get_device(idx=0)
res = faiss.StandardGpuResources()

normalize = T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
single_transform = T.Compose([T.ToTensor(), normalize])


def eval_loop(alpha, idd, train_loader, ind=None):
    #state_dict = torch.load(f"../saved_models/simfig_{alpha:.1f}/{idd}.pth", map_location=device)["state_dict"]
    #branch.load_state_dict(state_dict)
    branch = torch.load(f"../saved_models/mass_straj_{alpha}_t0/{idd}_encoder.pt")
    branch.to(device)
    branch.eval()
    encoder = branch.encoder

    embeds = []
    lbls = []
    for idx, (images, labels) in enumerate(train_loader):
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
    index = faiss.index_cpu_to_gpu(res, 0, index)
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
    traind, folder = get_dataset("orbit_config_default.json", "../saved_datasets")
    train_loader = torch.utils.data.DataLoader(
            dataset = traind,
            shuffle = False,
            batch_size = 512,
            drop_last=True
    )

    skip_factor = 20
    num_e = 4000
    results = np.zeros((9,4,int(num_e / skip_factor)))
    for it1, E in enumerate([20,40,80,160,320,640,1280,2560,4000]):
        adj1, lbl1 = eval_loop("100", 4000, E, train_loader)
        for it2, n in enumerate(["03,10,25,100"]):
            print(f"Experiment {E} x {n}")
            for it3, e in enumerate(tqdm(range(0,num_e,skip_factor))):
                f = e + skip_factor
                adj2, lbl2 = eval_loop(n, num_e, f, train_loader)
                results[it1, it2, it3] = matrix_similarity(adj1, adj2)
                print(E, n, f, results[it1, it2, it3])
                np.save("kepler_results", results)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    args = parser.parse_args()
    main_plot(args)
