import torch
import torch.nn.functional as F
import torch.nn as nn

import os
import argparse
import sys
sys.path.append('./') # now can access entire repository, (important when running locally)


from models import branch, predictor



from cl.tools.seed import set_deterministic
from cl.optimizers.LR_scheduler import LR_Scheduler

from orbit_datasets import neworbits, versatileorbits, staticorbits
from Losses.NCE_losses import infoNCE, rmseNCE, normalmseNCE




FOLDER_ROOT = os.getcwd() + "/cl/ExperimentsCL/"
SCRIPT_PATH = "ExperimentsCL/"

saved_epochs = list(range(20)) + [20,40,60,80]

def training_loop(args):

    """
    drop_last: drop the last non_full batch (potentially useful for training weighting etc.)
    pin_memory: speed dataloader transfer to cuda
    num_workers: multiprocess data loading
    """
    # dataloader_kwargs = dict(drop_last=True, pin_memory=True, num_workers=16)
    dataloader_kwargs = {}
    train_orbits_dataset = neworbits.OrbitsDataset()
    train_orbits_loader = torch.utils.data.DataLoader(
        dataset = train_orbits_dataset,
        shuffle = True,
        batch_size = args.bsz,
        **dataloader_kwargs
    )


    encoder = branch.branchEncoder(encoder_out=3)
    model_type = "3Dorbits_mseNCE_new/"
    save_progress_path = SCRIPT_PATH + "saved_models/" + model_type
    os.mkdir(save_progress_path)

    custom_parameters = [{
            'name': 'base',
            'params': encoder.parameters(),
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
        return encoder(x.float())
    def apply_loss(z1, z2):
        loss = 0.5 * normalmseNCE(z1, z2) + 0.5 * normalmseNCE(z2, z1)
        return loss


    for e in range(args.epochs):
        # main_branch.train()

        for it, (input1, input2, y) in enumerate(train_orbits_loader):
            encoder.zero_grad()

            # forward pass
            # z1 = get_z(input1.cuda())
            # z2 = get_z(input2.cuda())
            z1 = get_z(input1)
            z2 = get_z(input2)
            loss = apply_loss(z1, z2)

            # optimization step
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
        
        #print the loss of the last iteration of that epoch
        print("epoch" + str(e) + "    loss = " + str(loss))

        if e in saved_epochs:
            torch.save(encoder, save_progress_path + str(e) + '_encoder.pt')


    torch.save(encoder, save_progress_path + 'final_encoder.pt')

    
    
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

    args = parser.parse_args()
    training_loop(args)







