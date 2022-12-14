import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

import os
import shutil
import argparse
# import sys
# sys.path.append('./') # now can access entire repository, (important when running locally)


from models import branch, predictor



from ldcl.tools.seed import set_deterministic
from ldcl.optimizers.lr_scheduler import LR_Scheduler
from ldcl.data import physics, neworbits
from ldcl.losses.nce import infoNCE, rmseNCE, normalmseNCE
from ldcl.losses.simclr import NT_Xent_loss, infoNCE
from ldcl.tools.device import get_device
from ldcl.losses.simsiam import simsiam

device = get_device()

import pathlib
SCRIPT_PATH = pathlib.Path(__file__).parent.resolve().as_posix() + "/" # always get this directory

saved_epochs = list(range(20)) + [20,40,60,80]

def training_loop(args):

    """
    drop_last: drop the last non_full batch (potentially useful for training weighting etc.)
    pin_memory: speed dataloader transfer to cuda
    num_workers: multiprocess data loading
    """
    save_progress_path = os.path.join(SCRIPT_PATH, "saved_models", args.fname)
    os.mkdir(save_progress_path)

    # dataloader_kwargs = dict(drop_last=True, pin_memory=True, num_workers=16)
    dataloader_kwargs = {}
    data_config_file = "orbit_config_default.json"

    train_orbits_dataset, folder = physics.get_dataset(data_config_file, "../saved_datasets")
    print(f"Using dataset {folder}...")
    shutil.copy(data_config_file, os.path.join(save_progress_path, "data_config.json"))
    train_orbits_loader = torch.utils.data.DataLoader(
        dataset = train_orbits_dataset,
        shuffle = True,
        batch_size = args.bsz,
    )

    encoder = branch.branchEncoder(encoder_out=3)
    proj_head = branch.projectionHead(head_size=3)
    predictor = branch.predictor(size = 3)

    torch.save(encoder, os.path.join(save_progress_path, 'start_encoder.pt'))
    # if args.projhead:
    #     torch.save(proj_head, os.path.join(save_progress_path, 'start_projector.pt'))

    custom_parameters = [{
            'name': 'base',
            'params': list(encoder.parameters()) + list(proj_head.parameters()) + list(predictor.parameters()),
            'lr': args.lr
        }]
    optimizer = torch.optim.SGD(custom_parameters, lr=args.lr, momentum=0.9, weight_decay=args.wd)
    
    lr_scheduler = LR_Scheduler(
        optimizer=optimizer,
        warmup_epochs=args.warmup_epochs,
        warmup_lr=0,
        num_epochs=args.epochs,
        base_lr=args.lr * args.bsz / 256,
        final_lr=0,
        iter_per_epoch=len(train_orbits_loader),
        constant_predictor_lr=True
    )  

    # helpers
    def get_z(x):
        out = proj_head(encoder(x.float()))
        # if args.projhead: out = proj_head(out)
        return out
    def apply_loss(z1, z2, loss_func):
        with torch.no_grad():
            p1 = predictor(z1)
            p2 = predictor(z2)

        loss = 0.5 * loss_func(z1, p2) + 0.5 * loss_func(z2, p1)

        return loss
    
    losses = []

    for e in range(args.epochs):
        # main_branch.train()

        for it, (input1, input2, y) in enumerate(train_orbits_loader):
            encoder.zero_grad()
            proj_head.zero_grad()
            predictor.zero_grad()

            z1 = get_z(input1)
            z2 = get_z(input2)

            loss = apply_loss(z1, z2, simsiam)

            # optimization step
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
        
        #print the loss of the last iteration of that epoch
        print("epoch" + str(e) + "    loss = " + str(loss))

        losses.append(loss.detach().numpy().flatten()[0])

        if e in saved_epochs:
            torch.save(encoder, os.path.join(save_progress_path,f'{e:02d}_encoder.pt'))
            torch.save(proj_head, os.path.join(save_progress_path,f'{e:02d}_projector.pt'))
            torch.save(predictor, os.path.join(save_progress_path,f'{e:02d}_predictor.pt'))


    torch.save(encoder, os.path.join(save_progress_path, 'final_encoder.pt'))
    torch.save(proj_head, os.path.join(save_progress_path, 'final_projector.pt'))
    torch.save(predictor, os.path.join(save_progress_path,'final_predictor.pt'))
    losses = np.array(losses)
    np.save(os.path.join(save_progress_path, "loss.npy"), losses)


    
    
    return encoder




if __name__ == '__main__':
    pass
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--lr', default=0.02, type=float)
    parser.add_argument('--bsz', default=512, type=int)
    parser.add_argument('--wd', default=0.001, type=float)
    parser.add_argument('--warmup_epochs', default=5, type=int)
    parser.add_argument('--fine_tune', default=False, type=bool)
    #parser.add_argument('--projhead', default=False, type=bool)
    # parser.add_argument('--fname', default='default_model' , type = str)
    # parser.add_argument('--fname', default='simclr_infoNCE_1hidden_head_4dim' , type = str)
    parser.add_argument('--fname', default='rmseNCE_3D' , type = str)

    args = parser.parse_args()
    #print(args.projhead)
    training_loop(args)


