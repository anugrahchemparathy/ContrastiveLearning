import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

import os
import shutil
import argparse

from ldcl.models import branch, predictor

from ldcl.tools.seed import set_deterministic
from ldcl.optimizers.lr_scheduler import LR_Scheduler, get_lr_scheduler
from ldcl.data import physics
#from ldcl.losses.nce import infoNCE, rmseNCE, normalmseNCE
#from ldcl.losses.simclr import NT_Xent_loss, infoNCE
#from ldcl.losses.simsiam import simsiam
from ldcl.plot.plot import plot_loss
from ldcl.tools.device import get_device, t2np

import tqdm

device = get_device()

import pathlib
SCRIPT_PATH = pathlib.Path(__file__).parent.resolve().as_posix() + "/" # always get this directory

saved_epochs = list(range(20)) + [20,40,60,80,100,200,300,400,500,1000,1400]

def training_loop(args):
    global saved_epochs

    """
    drop_last: drop the last non_full batch (potentially useful for training weighting etc.)
    pin_memory: speed dataloader transfer to cuda
    num_workers: multiprocess data loading
    """
    save_progress_path = os.path.join(SCRIPT_PATH, "saved_models", args.fname)
    os.mkdir(save_progress_path)

    data_config_file = "data_configs/" + args.data_config

    train_orbits_dataset, folder = physics.get_dataset(data_config_file, "../saved_datasets")
    #print(f"Using dataset {folder}...")
    shutil.copy(data_config_file, os.path.join(save_progress_path, "data_config.json"))
    train_orbits_loader = torch.utils.data.DataLoader(
        dataset = train_orbits_dataset,
        shuffle = True,
        batch_size = args.bsz,
    )

    if args.all_epochs:
        saved_epochs = list(range(args.epochs))

    #encoder = branch.branchEncoder(encoder_out=3, useBatchNorm=True)
    encoder = branch.branchEncoder(encoder_out=3)
    # encoder = branch.branchEncoder(encoder_out=3, useBatchNorm=False, activation= nn.Sigmoid())
    #encoder = branch.branchImageEncoder(encoder_out=3)

    model = branch.sslModel(encoder=encoder)
    model.to(device)
    model.save(save_progress_path, 'start')

    optimizer = torch.optim.SGD(model.params(args.lr), lr=args.lr, momentum=0.9, weight_decay=args.wd)
    lr_scheduler = get_lr_scheduler(args, optimizer, train_orbits_loader)

    print("note: we rescale phi0, H, L by (6.28, 0.25, 1)^(-1)")
    multiplier = np.reshape(1 / np.array([6.28, 0.25, 1]), (1,3)) # equalize variances
    multiplier = torch.from_numpy(multiplier).type(torch.float32).to(device)
    def apply_loss(target, z1, z2, loss_func):
        target = target * multiplier
        z1 = z1 * multiplier
        z2 = z2 * multiplier

        loss = 0.5 * loss_func(target, z1) + 0.5 * loss_func(target, z2)
        return loss
    
    losses = []

    with tqdm.trange(args.epochs) as t:
        for e in range(args.epochs):
            model.train()

            for it, (input1, input2, y) in enumerate(train_orbits_loader):
                model.zero_grad()

                # forward pass
                input1 = input1.type(torch.float32).to(device)
                input2 = input2.type(torch.float32).to(device)
                z1 = model(input1)
                z2 = model(input2)

                target = torch.cat((y["phi0"], y["H"], y["L"]), axis=1).type(torch.float32).to(device)
                loss = apply_loss(target, z1, z2, torch.nn.MSELoss())

                # optimization step
                loss.backward()
                optimizer.step()
                lr_scheduler.step()

            losses.append(t2np(loss).flatten()[0])

            if e in saved_epochs:
                model.save(save_progress_path, f'{e:02d}')
            t.set_postfix(loss=loss.item(), loss50_avg=np.mean(np.array(losses[max(-1 * e, -50):])))
            t.update()

    model.save(save_progress_path, 'final')
    losses = np.array(losses)
    np.save(os.path.join(save_progress_path, "loss.npy"), losses)
    plot_loss(losses, title = args.fname, save_progress_path = save_progress_path)
    
    return encoder


if __name__ == '__main__':
    pass
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=1500, type=int)
    parser.add_argument('--lr', default=0.02, type=float)
    parser.add_argument('--bsz', default=512, type=int)
    parser.add_argument('--wd', default=0.001, type=float)
    parser.add_argument('--warmup_epochs', default=5, type=int)
    parser.add_argument('--fine_tune', default=False, type=bool)
    parser.add_argument('--fname', default='rmse_1500_a' , type = str)
    parser.add_argument('--data_config', default='orbit_config_default.json' , type = str)
    parser.add_argument('--all_epochs', default=False, type=bool)

    args = parser.parse_args()
    training_loop(args)
