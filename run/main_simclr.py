import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

import os
import shutil
import argparse

from ldcl.models import branch, predictor
from ldcl.models.main import Branch

from ldcl.tools.seed import set_deterministic
from ldcl.optimizers.lr_scheduler import LR_Scheduler, get_lr_scheduler
from ldcl.data import physics
#from ldcl.losses.nce import infoNCE, rmseNCE, normalmseNCE
from ldcl.losses.simclr import NT_Xent_loss, infoNCE
#from ldcl.losses.simsiam import simsiam
from ldcl.plot.plot import plot_loss
from ldcl.tools.device import get_device, t2np
from ldcl.tools import metrics, utils, main

import tqdm

device = get_device(idx=3)

import pathlib
SCRIPT_PATH = pathlib.Path(__file__).parent.resolve().as_posix() + "/" # always get this directory

saved_epochs = list(range(20)) + [20,40,60,80,100,200,300,400,500,600,700,1000,1400]

def training_loop(args):
    global saved_epochs

    """
    drop_last: drop the last non_full batch (potentially useful for training weighting etc.)
    pin_memory: speed dataloader transfer to cuda
    num_workers: multiprocess data loading
    """
    save_progress_path = os.path.join(SCRIPT_PATH, "saved_models", args.fname)
    while os.path.exists(save_progress_path):
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
                torch.utils.data.RandomSampler(train_orbits_dataset), batch_size=args.bsz, drop_last=False
            ),
    )


    is_natural = isinstance(train_orbits_dataset, physics.NaturalDataset)
    if is_natural:
        train_orbits_dataset2, folder = physics.get_dataset(data_config_file, "../saved_datasets", no_aug=True)
        train_orbits_loader2 = torch.utils.data.DataLoader(
            dataset = train_orbits_dataset2,
            shuffle = False,
            batch_size = args.bsz,
        )
        test_orbits_dataset, folder = physics.get_dataset(data_config_file.replace("train", "test"), "../saved_datasets", no_aug=True)
        test_orbits_loader = torch.utils.data.DataLoader(
            dataset = test_orbits_dataset,
            shuffle = False,
            batch_size = args.bsz
        )

    if args.all_epochs:
        saved_epochs = list(range(args.epochs))

    #encoder = branch.branchEncoder(encoder_out=3, useBatchNorm=True, activation= nn.Sigmoid())
    #encoder = branch.branchEncoder(encoder_out=3, num_layers=15, useBatchNorm=True, encoder_hidden=128)
    #encoder = branch.branchEncoder(encoder_out=3)
    if is_natural:
        #encoder = branch.branchImageEncoder(encoder_out=1024, useBatchNorm=True, encoder_hidden=768, num_layers=2)
        #proj_head = branch.projectionHead(head_in=1024, head_out=1024, num_layers=3, hidden_size=512)
        branchep = Branch()
        encoder = branchep.encoder
        proj_head = branchep.projector
    else:
        encoder = branch.branchImageEncoder(encoder_out=3, useBatchNorm=True)
        proj_head = branch.projectionHead(head_in=3, head_out=4, num_layers=3, hidden_size=128)
    #proj_head = branch.projectionHead(head_in=3, head_out=4)

    model = branch.sslModel(encoder=encoder, projector=proj_head)
    model.to(device)
    model.save(save_progress_path, 'start')

    optimizer = torch.optim.SGD(model.params(args.lr), lr=args.lr, momentum=0.9, weight_decay=args.wd)
    lr_scheduler = get_lr_scheduler(args, optimizer, train_orbits_loader)

    def apply_loss(z1, z2, loss_func):
        loss = loss_func(z1, z2)
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
    
    if is_natural:
        emtrs = {
            "knn_u": lambda: utils.knn_monitor(model.encoder, train_orbits_loader2, test_orbits_loader, device=str(device)),
            "lin_u": lambda: main.eval_loop(model.encoder, train_orbits_loader2, test_orbits_loader, device=device, epochs=5)
        }
        """
            "knn": lambda: metrics.knn_eval(model, train_orbits_loader2, test_orbits_loader, device=device),
            "lin": lambda: metrics.lin_eval(model, train_orbits_loader2, test_orbits_loader, device=device),
            "knn_v": lambda: metrics.knn_eval(model, train_orbits_loader2, train_orbits_loader2, device=device),
            "lin_v": lambda: metrics.lin_eval(model, train_orbits_loader2, train_orbits_loader2, device=device),
        """
    else:
        emtrs = {}

    with tqdm.trange(args.epochs * len(train_orbits_loader)) as t:
        update_metrics(t, do_eval=True)
        for e in range(args.epochs):
            model.train()

            for it, (input1, input2, y) in enumerate(train_orbits_loader):
                model.zero_grad()

                # forward pass
                input1 = input1[0].type(torch.float32).to(device)
                input2 = input2[0].type(torch.float32).to(device)
                z1 = model(input1)
                z2 = model(input2)

                loss = apply_loss(z1, z2, infoNCE)

                # optimization step
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
    # parser.add_argument('--fname', default='simclr_infoNCE_1hidden_head_4dim' , type = str)
    parser.add_argument('--fname', default='simclr_1500_d' , type = str)
    parser.add_argument('--data_config', default='orbit_config_default.json' , type = str)
    parser.add_argument('--all_epochs', default=False, type=bool)
    parser.add_argument('--eval_every', default=3, type=int)

    args = parser.parse_args()
    training_loop(args)
