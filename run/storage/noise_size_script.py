import os
import copy
import subprocess
from ldcl.data import physics
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import normalize
import numpy as np
import torch

test = np.array(list(range(10))) * 0.03
iters = 2

def make_config(size):
    with open("data_configs/noise_adjust.json", "r") as f:
        CONFIG = f.read()

    config = copy.deepcopy(CONFIG)
    config = config.replace("{noise}", str(size))

    with open("data_configs/noise_temp.json", "w") as f:
        f.write(config)

def run_experiment():
    for size in test:
        make_config(size)

        for i in range(1, iters+1):
            if os.path.isdir(f'saved_models/noise_{size}_{i}'): # don't rerun experiments
                continue

            subprocess.run(f"python main.py --fname=noise_{size}_{i} --data_config=noise_temp.json", shell=True)
            print(f"Completed running iteration {i} of experiment {size}...")

def analyze_experiment():
    """mean_reduce = np.zeros((len(test), iters)).tolist()
    max_reduce = np.zeros((len(test), iters)).tolist()
    allvectors = np.zeros((len(test), iters)).tolist()
    allt = np.zeros((len(test), iters)).tolist()"""
    sizes = []

    for idx, size in enumerate(test):
        print(f"Starting {size}...")
        #make_config(size)
        make_config(test[-1])
        data = physics.get_dataset("data_configs/noise_temp.json", "../saved_datasets")[0]
        orbits_loader = torch.utils.data.DataLoader(
            dataset = data,
            shuffle = True,
            batch_size = 1,
        )
        sizes.append([])

        for i in range(1, iters+1):
            branch_encoder = torch.load(f"saved_models/noise_{size}_{i}/final_encoder.pt", map_location=torch.device('cpu'))
            branch_encoder.eval()
            encodings = np.array((len(data), 3))
            conserved_quantities = np.array((len(data), 3))
            encodings = []
            conserved_quantities = []

            for it, (input1, input2, y) in enumerate(orbits_loader):
                encodings.append(branch_encoder(input1.float()).detach().numpy()[0])
                conserved_quantities.append(np.array( (y["H"].item(),y["L"].item(),y["phi0"].item()) ))
            encodings = np.array(encodings)
            conserved_quantities = np.array(conserved_quantities)

            sizes[-1].append(np.sum(np.mean(np.square(encodings - np.mean(encodings, axis=0)), axis=0)))
            
            """vectors = np.array(vectors)
            group0 = np.array(group0)
            group1 = np.array(group1)
            dists = []
            t = [[], []]
            for ii in range(3):
                dists.append(vectors[:, ii, :] @ vectors[:, ii, :].T) # get cosine similarities
                t[0].append(group0[ii, :, :] @ group0[ii, :, :].T)
                t[1].append(group1[ii, :, :] @ group1[ii, :, :].T)
            dists = np.array(dists)
            mean_reduce[idx][i - 1] = np.mean(dists)
            max_reduce[idx][i - 1] = np.max(dists)
            allt[idx][i - 1] = t
            allvectors[idx][i - 1] = vectors"""

    """mean_reduce = np.array(mean_reduce)            
    max_reduce = np.array(max_reduce)            
    allvectors = np.array(allvectors)
    allt = np.array(allt)

    np.save("results/eight_mini_cubes/mean", mean_reduce)
    np.save("results/eight_mini_cubes/max", mean_reduce)
    np.save("results/eight_mini_cubes/test", np.array(test))
    np.save("results/eight_mini_cubes/vecs", np.array(allvectors))
    np.save("results/eight_mini_cubes/t", np.array(allt))"""
    np.save("results/noise", np.array(sizes))

run_experiment()
analyze_experiment()
