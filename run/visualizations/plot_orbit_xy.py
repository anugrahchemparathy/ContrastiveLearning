import numpy as np
import matplotlib.pyplot as plt
from ldcl.data import physics
import json

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--phi0', default=0, type=float)
parser.add_argument('--H', default=-0.375, type=float)
parser.add_argument('--L', default=0.5, type=float)
args = parser.parse_args()

with open("../data_configs/single_orbit.json", "r") as f:
    s = f.read()

s = s.replace("-0.375,0.5,3.14", f"{args.H},{args.L},{args.phi0}")

with open("../data_configs/single_orbit_temp.json", "w") as f:
    f.write(s)

d, f = physics.get_dataset("../data_configs/single_orbit_temp.json", "../../saved_datasets")
print(f"dataset: {f}")
arr = d.data
arr = np.reshape(arr, (arr.shape[0] * arr.shape[1], arr.shape[2]))

fig, axs = plt.subplots(nrows=1, ncols=3)

axs[0].set_xlabel("x")
axs[0].set_ylabel("y")
axs[0].set_aspect('equal', 'box')
axs[0].scatter(arr[:, 0], arr[:, 1]) # x, y

axs[1].set_xlabel("v.x")
axs[1].set_ylabel("v.y")
axs[1].set_aspect('equal', 'box')
axs[1].scatter(arr[:, 2], arr[:, 3]) # v.x,v.y

axs[2].set_xlabel("x")
axs[2].set_ylabel("v.x")
axs[2].set_aspect('equal', 'box')
axs[2].scatter(arr[:, 0], arr[:, 2]) # x,v.x

plt.show()
