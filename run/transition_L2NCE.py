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
from ldcl.plot.plot import VisPlot, plot_metric
from ldcl.tools.device import get_device, t2np
from ldcl.tools import metrics


import tqdm
from sklearn.linear_model import LinearRegression
import pickle


device = get_device()

scaler = GradScaler()

import pathlib
SCRIPT_PATH = pathlib.Path(__file__).parent.resolve().as_posix() + "/" # always get this directory


def constrained_training_loop(args, threshhold):


    data_config_file = "data_configs/" + args.data_config
    metric_data_config_file = "data_configs/" + args.metric_config

    train_orbits_dataset, folder = physics.get_dataset(data_config_file, "../saved_datasets")
    print(f"Using dataset {folder}...")
    train_orbits_loader = torch.utils.data.DataLoader(
        dataset = train_orbits_dataset,
        sampler=torch.utils.data.BatchSampler(
                torch.utils.data.RandomSampler(train_orbits_dataset), batch_size=args.bsz, drop_last=True
            ),
    )

    metric_orbits_dataset, folder = physics.get_dataset(metric_data_config_file, "../saved_datasets")
    print(f"Using dataset {folder}...")
    metric_orbits_loader = torch.utils.data.DataLoader(
        dataset = metric_orbits_dataset,
        batch_size = args.bsz
    )


    if args.all_epochs:
        saved_epochs = list(range(args.epochs))


    # encoder = branch.branchEncoder(encoder_out=3, activation=nn.Sigmoid(), useBatchNorm=True)
    encoder = branch.branchEncoder(encoder_out=3, activation=nn.ReLU())
    # encoder = branch.branchEncoder(encoder_out=3, activation=nn.ReLU(), useBatchNorm=True)

    model = branch.sslModel(encoder=encoder)
    model.to(device)

    optimizer = torch.optim.SGD(model.params(args.lr), lr=args.lr, momentum=0.9, weight_decay=args.wd)
    lr_scheduler = get_lr_scheduler(args, optimizer, train_orbits_loader)

    def apply_loss(z1, z2, loss_func = rmseNCE):
        loss = 0.5 * loss_func(z1, z2) + 0.5 * loss_func(z2, z1)
        return loss

    losses = []
    mtrd = {"loss": None, "avg_loss": None}
    saved_metrics = {'H':[], 'L':[]}
    reg = LinearRegression()


    def update_metrics(t, new_loss=None, losses=None, do_eval=False):
        if new_loss is not None:
            mtrd["loss"] = new_loss
        if losses is not None:
            mtrd["avg_loss"] = np.mean(np.array(losses[max(-1 * len(losses), -50):]))


        outputs = []
        labels = {'H':[],'L':[],'phi0':[]}
        model.eval()
        for it, (input1, input2, y) in enumerate(metric_orbits_loader):
            # if it >= 2: break
            input1 = input1.type(torch.float32).to(device)
            output1 = model(torch.squeeze(input1))
            outputs.append(output1)
            for label in ['H','L','phi0']:
                labels[label].append(torch.squeeze(y[label])[:,0])

        outputs = torch.cat(outputs).cpu().detach().numpy()
        for label in ['L']:
            labels[label] = torch.cat(labels[label]).cpu().detach().numpy()


        reg.fit(outputs, labels['L'])
        L_score = reg.score(outputs,labels['L'])
        saved_metrics['L'].append(L_score)
            
        t.update()
        t.set_postfix(**mtrd)

        return L_score



    L_scores = []

    with tqdm.trange(args.epochs) as t:
        update_metrics(t, do_eval=True)
        for e in range(args.epochs):
            model.train()

            epoch_loss = []

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

                epoch_loss.append(t2np(loss).flatten()[0])


            losses.append(sum(epoch_loss))
            L_score = update_metrics(t, new_loss=loss.item(), losses=losses)
            L_scores.append(L_score)

            if sum(L_scores[-3:])/3 >= threshhold: break


    return L_scores



def collect_data(args):
    save_progress_path = os.path.join(SCRIPT_PATH, "saved_models", args.fname)
    while os.path.exists(save_progress_path) and not args.override:
        to_del = input("Saved directory already exists. If you continue, you may erase previous training data. Press Ctrl+C to stop now. Otherwise, type 'yes' to continue:")
        if to_del == "yes":
            shutil.rmtree(save_progress_path)
    os.mkdir(save_progress_path)

    num_trials = 16
    L_scores = []
    for i in range(num_trials):
        print("Trial", i)
        trial_L_scores  = constrained_training_loop(args, threshhold=0.8)
        L_scores.append(trial_L_scores)
    
    plot_metric(L_scores, title = "L fitting with 5\% trajectories", save_progress_path = save_progress_path, ylabel = 'R2 value', save_name = 'L_fitting', legend = [f'trial {i}' for i in range(len(L_scores))])

    with open(os.path.join(save_progress_path, 'L_score_list.pkl'), 'wb') as f:
        pickle.dump(L_scores,f)

    # print(L_scores)



if __name__ == '__main__':
    pass
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=500, type=int)
    parser.add_argument('--lr', default=0.02, type=float)
    parser.add_argument('--bsz', default=512, type=int)
    parser.add_argument('--wd', default=0.001, type=float)
    parser.add_argument('--warmup_epochs', default=5, type=int)
    parser.add_argument('--fine_tune', default=False, type=bool)
    parser.add_argument('--fname', default='transitionA' , type = str)
    parser.add_argument('--data_config', default='orbit_config_default.json' , type = str)
    parser.add_argument('--metric_config', default='metric_config_default.json' , type = str)
    parser.add_argument('--all_epochs', default=False, type=bool)
    parser.add_argument('--eval_every', default=3, type=int)
    parser.add_argument('--mixed_precision', action='store_true')
    parser.add_argument('--override', action='store_true')
    parser.add_argument('--device', default=0, type=int)

    args = parser.parse_args()
    device = get_device(idx=args.device)
    print(args.device, 'arg')
    input(device)
    collect_data(args)
