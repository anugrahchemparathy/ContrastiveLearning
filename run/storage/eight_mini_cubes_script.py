import os
import copy
import subprocess
from ldcl.data import physics
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import normalize
import numpy as np
import torch

test = [0.33,0.30,0.27,0.24,0.21,0.18,0.15,0.12,0.09,0.06,0.03]
iters = 2

def make_config(size):
    with open("data_configs/eight_mini_cubes_adjust.json", "r") as f:
        CONFIG = f.read()

    config = copy.deepcopy(CONFIG)
    config = config.replace("{H0}", str(-0.5 + 0.25 * size))
    config = config.replace("{H1}", str(-0.25 - 0.25 * size))
    config = config.replace("{L0}", str(size))
    config = config.replace("{L1}", str(1 - size))
    config = config.replace("{p0}", str(3.14 * 2 * size))
    config = config.replace("{p1}", str(3.14 * 2 * size + 3.14))

    with open("data_configs/eight_mini_cubes_temp.json", "w") as f:
        f.write(config)

def run_experiment():
    for size in test:
        make_config(size)

        for i in range(1, iters+1):
            if os.path.isdir(f'saved_models/eight_mini_cubes_{size}_{i}'): # don't rerun experiments
                continue

            subprocess.run(f"python main.py --fname=eight_mini_cubes_{size}_{i} --data_config=eight_mini_cubes_temp.json", shell=True)
            print(f"Completed running iteration {i} of experiment {size}...")

def analyze_experiment():
    """mean_reduce = np.zeros((len(test), iters)).tolist()
    max_reduce = np.zeros((len(test), iters)).tolist()
    allvectors = np.zeros((len(test), iters)).tolist()
    allt = np.zeros((len(test), iters)).tolist()"""
    vecs = np.zeros((len(test), iters))

    for idx, size in enumerate(test):
        print(f"Starting {size}...")
        #make_config(size)
        make_config(test[-1])
        data = physics.get_dataset("data_configs/eight_mini_cubes_temp.json", "../saved_datasets")[0]
        orbits_loader = torch.utils.data.DataLoader(
            dataset = data,
            shuffle = True,
            batch_size = 1,
        )

        for i in range(1, iters+1):
            branch_encoder = torch.load(f"saved_models/eight_mini_cubes_{size}_{i}/final_encoder.pt", map_location=torch.device('cpu'))
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

            vectors = []
            group0 = [[], [], []]
            group1 = [[], [], []]
            for H in [0,1]:
                if H == 0:
                    mask = conserved_quantities[:, 0] > -0.375
                else:
                    mask = conserved_quantities[:, 0] <= -0.375
                s_encodings = encodings[mask]
                s_conserved_quantities = conserved_quantities[mask]

                for L in [0,1]:
                    if L == 0:
                        mask = s_conserved_quantities[:, 1] > 0.5
                    else:
                        mask = s_conserved_quantities[:, 1] <= 0.5
                    r_encodings = s_encodings[mask]
                    r_conserved_quantities = s_conserved_quantities[mask]

                    for phi0 in [0,1]:
                        if phi0 == 0:
                            mask = r_conserved_quantities[:, 2] > 3.14
                        else:
                            mask = r_conserved_quantities[:, 2] <= 3.14
                        q_encodings = r_encodings[mask]
                        q_conserved_quantities = r_conserved_quantities[mask]

                        q_vectors = []
                        for var in range(3):
                            lr = LinearRegression()
                            lr.fit(q_encodings, q_conserved_quantities[:, var])
                            q_vectors.append([H, L, phi0] + lr.coef_ / np.sqrt(np.sum(np.square(lr.coef_))))
                            """if q_conserved_quantities[0, var] <= [-0.375, 0.5, 3.14][var]:
                                group0[var].append(lr.coef_ / np.sqrt(np.sum(np.square(lr.coef_))))
                            else:
                                group1[var].append(lr.coef_ / np.sqrt(np.sum(np.square(lr.coef_))))"""
                        vectors.append(np.array(q_vectors))
            vecs[idx][i - 1] = np.array(vectors)
            
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
    np.save("results/vecs", np.array(vecs))

run_experiment()
analyze_experiment()
