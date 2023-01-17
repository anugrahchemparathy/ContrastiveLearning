import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import numpy as np

import os
import shutil
import argparse

from ldcl.models import branch, predictor

from ldcl.tools.seed import set_deterministic
from ldcl.optimizers.lr_scheduler import LR_Scheduler, get_lr_scheduler
from ldcl.data import physics
from ldcl.losses.nce import infoNCE, rmseNCE, normalmseNCE
#from ldcl.losses.simclr import NT_Xent_loss, infoNCE
#from ldcl.losses.simsiam import simsiam
from ldcl.plot.plot import plot_loss
from ldcl.tools.device import get_device, t2np
from ldcl.tools import metrics

import tqdm

device = get_device()

scaler = GradScaler()

import pathlib
SCRIPT_PATH = pathlib.Path(__file__).parent.resolve().as_posix() + "/" # always get this directory

#saved_epochs = list(range(20)) + [20,40,60,80,100,200,300,400,500,600,700,1000,1400]
saved_epochs = list(range(0, 10000, 20))

def training_loop(args):
    global saved_epochs

    """
    drop_last: drop the last non_full batch (potentially useful for training weighting etc.)
    pin_memory: speed dataloader transfer to cuda
    num_workers: multiprocess data loading
    """
    save_progress_path = os.path.join(SCRIPT_PATH, "saved_models", args.fname)
    while os.path.exists(save_progress_path) and not args.override:
        to_del = input("Saved directory already exists. If you continue, you may erase previous training data. Press Ctrl+C to stop now. Otherwise, type 'yes' to continue:")
        if to_del == "yes":
            shutil.rmtree(save_progress_path)
    os.mkdir(save_progress_path)

    data_config_file = "data_configs/" + args.data_config

    train_orbits_dataset, folder = physics.get_dataset(data_config_file, "../saved_datasets")
    print(f"Using dataset {folder}...")
    shutil.copy(data_config_file, os.path.join(save_progress_path, "data_config.json"))
    train_orbits_loader = torch.utils.data.DataLoader(
        dataset = train_orbits_dataset,
        sampler=torch.utils.data.BatchSampler(
                torch.utils.data.RandomSampler(train_orbits_dataset), batch_size=args.bsz, drop_last=True
            ),
    )

    is_natural = isinstance(train_orbits_dataset, physics.NaturalDataset)
    if is_natural:
        train_orbits_dataset2, folder = physics.get_dataset(data_config_file, "../saved_datasets", no_aug=True)
        train_orbits_loader2 = torch.utils.data.DataLoader(
            dataset = train_orbits_dataset2,
            shuffle = False,
            batch_size = args.bsz,
            drop_last=True
        )
        test_orbits_dataset, folder = physics.get_dataset(data_config_file.replace("train", "test"), "../saved_datasets", no_aug=True)
        test_orbits_loader = torch.utils.data.DataLoader(
            dataset = test_orbits_dataset,
            shuffle = False,
            batch_size = args.bsz,
            drop_last=True
        )

    if args.all_epochs:
        saved_epochs = list(range(args.epochs))

    #encoder = branch.branchEncoder(encoder_out=3, useBatchNorm=True)
    #encoder = branch.branchEncoder(encoder_out=3)
    # encoder = branch.branchEncoder(encoder_out=3, useBatchNorm=False, activation= nn.Sigmoid())
    if is_natural:
        encoder = branch.branchImageEncoder(encoder_out=1024, useBatchNorm=True, encoder_hidden=768, num_layers=2)
    else:
        encoder = branch.branchImageEncoder(encoder_out=3)
        #encoder = branch.branchEncoder(encoder_out=3)

    model = branch.sslModel(encoder=encoder)
    model.to(device)
    model.save(save_progress_path, 'start')

    optimizer = torch.optim.SGD(model.params(args.lr), lr=args.lr, momentum=0.9, weight_decay=args.wd)
    lr_scheduler = get_lr_scheduler(args, optimizer, train_orbits_loader)

    def apply_loss(z1, z2, loss_func = normalmseNCE):
        loss = 0.5 * loss_func(z1, z2) + 0.5 * loss_func(z2, z1)
        return loss

    losses = []
    mtrd = {"loss": None, "avg_loss": None}
    saved_metrics = {}

    def update_metrics(t, new_loss=None, losses=None, do_eval=False):
        if new_loss is not None:
            mtrd["loss"] = new_loss
        if losses is not None:
            mtrd["avg_loss"] = np.mean(np.array(losses[max(-1 * len(losses), -50):]))

        if do_eval:
            for name, metric in emtrs.items():
                new_val = metric()
                if name in saved_metrics.keys():
                    saved_metrics[name].append(new_val)
                else:
                    saved_metrics[name] = [new_val]
                mtrd[name] = new_val

        t.set_postfix(**mtrd)

    emtrs = {} # training metrics

    with tqdm.trange(args.epochs * len(train_orbits_loader)) as t:
        update_metrics(t, do_eval=True)
        for e in range(args.epochs):
            model.train()

            for it, (input1, input2, y) in enumerate(train_orbits_loader):
                model.zero_grad()

                # forward pass
                input1 = input1[0].type(torch.float32).to(device)
                input2 = input2[0].type(torch.float32).to(device)

                if args.mixed_precision:
                    with autocast():
                        z1 = model(input1)
                        z2 = model(input2)

                        loss = apply_loss(z1, z2, rmseNCE)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    z1 = model(input1)
                    z2 = model(input2)

                    loss = apply_loss(z1, z2, rmseNCE)

                    loss.backward()
                    optimizer.step()
                lr_scheduler.step()

                losses.append(t2np(loss).flatten()[0])

                if e in saved_epochs and it == 0:
                    model.save(save_progress_path, f'{e:02d}')
                update_metrics(t, new_loss=loss.item(), losses=losses)
                t.update()

            if e % args.eval_every == args.eval_every - 1:
                update_metrics(t, do_eval=True)

    model.save(save_progress_path, 'final')
    losses = np.array(losses)
    np.save(os.path.join(save_progress_path, "loss.npy"), losses)

    for name, slist in saved_metrics.items():
        np.save(os.path.join(save_progress_path, f"{name}.npy"), slist)

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
    parser.add_argument('--eval_every', default=3, type=int)
    parser.add_argument('--mixed_precision', action='store_true')
    parser.add_argument('--override', action='store_true')
    parser.add_argument('--device', default=0, type=int)

    args = parser.parse_args()
    device = get_device(idx=args.device)
    print(args.device, 'arg')
    input(device)
    training_loop(args)
