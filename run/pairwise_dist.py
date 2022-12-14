import numpy as np
import torch
from pathlib import Path

import argparse

from ldcl.data import physics

def getAllReps(encoder_location, data):
    branch_encoder = torch.load(encoder_location, map_location=torch.device('cpu'))
    branch_encoder.eval()
    
    rep_list = []
    
    for orbit in data:
        predicted = branch_encoder(torch.from_numpy(orbit).float()).detach().numpy()
        rep_list.append(predicted)
    
    rep_outputs = np.stack(rep_list)
    
    return rep_outputs

def dist(args):
    orbits_dataset, folder = physics.get_dataset(args.config, "../saved_datasets")

    reps = getAllReps(Path(args.encoder), orbits_dataset.data)

    diff = reps[:, None, :, :] - reps[:,:,None,:]
    dist = np.linalg.norm(diff, axis = -1)

    num_traj = dist.shape[0]
    num_samples = dist.shape[-1]

    avg_dist = np.sum(dist) / (num_traj * (num_samples * (num_samples-1)))

    return avg_dist, folder

if __name__ == '__main__':
    """
    To use this script, just enter the location of the encoder and the config file you want to use 
    """
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder', type = str, required = True)
    parser.add_argument('--config', type = str, required = True)

    args = parser.parse_args()

    avg_dist, folder = dist(args)
    print(avg_dist)
    print(folder)
    
