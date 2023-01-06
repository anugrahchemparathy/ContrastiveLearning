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
import torchvision

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

class NaturalDataset(torch.utils.data.Dataset):
    def __init__(self, dtset, sz):
        self.data = []
        self.target = []
        for i in range(len(dtset)):
            self.data.append(torchvision.transforms.ToTensor()(dtset[i][0]))
            self.target.append(dtset[i][1])
        self.data = torch.stack(self.data)
        self.target = torch.IntTensor(self.target)
        self.size = self.target.size()[0]
        self.imgsz = sz

    def __getitem__(self, idx):
        if idx < self.size:
            x_data = self.data[idx]
            ts = torch.nn.Sequential(
                torchvision.transforms.RandomResizedCrop(self.imgsz, scale=[0.2,1.0]),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomApply([torchvision.transforms.ColorJitter(brightness=0.4,contrast=0.4,saturation=0.4,hue=0.1)],p=0.8),
                torchvision.transforms.RandomGrayscale(p=0.2)
            )
            if len(list(x_data.shape)) == 3:
                x_view1 = torch.swapaxes(ts(x_data), 0, 2)
                x_view2 = torch.swapaxes(ts(x_data), 0, 2)
            else:
                x_view1 = torch.swapaxes(ts(x_data), 1, 3)
                x_view2 = torch.swapaxes(ts(x_data), 1, 3)

            return [x_view1,x_view2, self.target[idx]]
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
    if "cifar10" in config or "cifar100" in config:
        return get_natural_dataset(config)
    else:
        return get_conservation_dataset(config, saved_dir, return_bundle)

def get_natural_dataset(config):
    if "train" in config:
        train = True
    elif "test" in config:
        train = False
    else:
        raise ValueError('train or test')

    if "cifar100" in config:
        dataset = torchvision.datasets.CIFAR100("../saved_datasets", download=True, train=train)
        dsname = "cifar100"
    elif "cifar10" in config:
        dataset = torchvision.datasets.CIFAR10("../saved_datasets", download=True, train=train)
        dsname = "cifar10"
    else:
        raise NotImplementedError

    return NaturalDataset(dataset, sz=224), dsname + "_train" if train else "_test"

def get_conservation_dataset(config, saved_dir, return_bundle=False):
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
