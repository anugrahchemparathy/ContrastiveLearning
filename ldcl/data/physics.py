import numpy as np

from munch import DefaultMunch
import json
from PIL import Image

import time
import glob
import os
from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import torch

from .config import read_config

from .pendulum import pendulum_num_gen, pendulum_img_gen
from .orbit import orbits_num_gen, orbits_img_gen

rng = np.random.default_rng(9)  # manually seed random number generator
verbose = True

class ConservationDataset(torch.utils.data.Dataset):
    def __init__(self, bundle):
        self.size = bundle["data"].shape[0]
        self.data = bundle["data"] # assuming this is where the actual data is
        del bundle["data"]
        self.bundle = bundle

    def __getitem__(self, idx):
        if idx < self.size:
            x_data = self.data[idx]
            random_x_rows = np.random.randint(0,x_data.shape[0],2) # generate two different views

            x_output = x_data[random_x_rows]
            x_view1 = x_output[0]
            x_view2 = x_output[1]

            return [x_view1,x_view2, {k: v[idx] for k, v in self.bundle.items()} | {"idxs_": random_x_rows}]
        else:
            raise ValueError

    def __len__(self):
        return self.size

def get_dataset(config, saved_dir, return_bundle=False):
    """
        General dataset generation.

        Eventually, we'll probably read the config file in the training loop and then just pass it in here as an object...

        :param config: Configuration file to receive parameters in, or already parsed configuration object.
        :return: Dataset object, and then name of folder inside saved_dir where data can be found
    """
    if isinstance(config, str):
        config = read_config(config)
    #print(saved_dir)

    # If cache, check if exists
    bundle = None
    if config.use_cached_data:
        for other_file in glob.glob(saved_dir + "/*/config.json"):
            if json.dumps(read_config(other_file)) == json.dumps(config):
                p = Path(other_file)
                other_file = p.parents[0]
                folder_name = str(os.path.join(other_file, ''))
                
                bundle = {}
                for npfile in glob.glob(os.path.join(other_file, "*.npy")):
                    name = Path(npfile).with_suffix('').name
                    bundle[name] = np.load(npfile)
    else:
        print("Settings specify to not use cached data. Make sure you want this; you're regenerating a dataset every time!")

    # Generate data
    if bundle is None:
        if config.dynamics == "pendulum":
            bundle = pendulum_num_gen(config)
            if config.modality == "image":
                bundle = pendulum_img_gen(config, bundle)
            elif config.modality == "numerical":
                bundle = bundle
            else:
                raise ValueError("Config modality not specified")
        elif config.dynamics == "orbits":
            bundle = orbits_num_gen(config)
            if config.modality == "image":
                bundle = orbits_img_gen(config, bundle)
            elif config.modality == "numerical":
                bundle = bundle
            else:
                raise ValueError("Config modality not specified")
        else:
            raise ValueError("Config dynamics not specified")

        # Save dataset
        folder_name = saved_dir + "/" + time.strftime("%Y%m%d") + "-"
        idx = 0
        while os.path.exists(folder_name + str(idx) + "/"):
            idx += 1
        folder_name = folder_name + str(idx)
        os.mkdir(folder_name)
        for key in bundle.keys():
            np.save(folder_name + "/" + key, bundle[key])

        with open(folder_name + "/config.json", "w") as f:
            json.dump(config, f)
    
    if return_bundle:
        return bundle, folder_name
    else:
        dataset = ConservationDataset(bundle)

        return dataset, folder_name

def combine_datasets(configs, ratio, save_folder):
    arrs = []

    if sum(ratio) < 0.95 or sum(ratio) > 1:
        raise ValueError("ratios must sum to 1")

    for config in configs:
        arrs.append(get_dataset(config, save_folder, return_bundle=True)[0])

    bundle = {}
    for key in arrs[0].keys():
        bundle[key] = np.concatenate([x[key] for x in arrs])

    dataset = ConservationDataset(bundle)

    return dataset
    

#get_dataset("orbit_config_default.json")

# Template to test distribution generation.
"""class TestDistribution:
    def __init__(self):
        self.type = "uniform_with_intervals"
        self.mode = "explicit"
        self.dims = 2
        self.intervals = [[[-1,0],[0.5,1],[2,3]], [[0,1],[2,3]]]
        self.combine = "any"

dist = TestDistribution()

sample = sample_distribution(dist, 100000)
plt.scatter(sample[:, 0], sample[:, 1], s=0.1)
plt.show()"""
