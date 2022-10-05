import os
import numpy as np
from numpy.core.numeric import full
import torch
import argparse
import time
import random

import torch.nn.functional as F
import torch.nn as nn


import torchvision
import torchvision.transforms as T

#====================================================



from cl.tools.seed import set_deterministic
from cl.optimizers.LR_scheduler import LR_Scheduler
#====================================================

#added imports
from orbit_datasets import neworbits, versatileorbits, staticorbits
from Losses.NCE_losses import infoNCE, rmseNCE, normalmseNCE




#====================================================
#function from seed.py in /cl/tools/seed.py
set_deterministic(42)


#====================================================
class Branch(nn.Module):
    def __init__(self, encoder_hidden, encoder=None):
        super().__init__()
        """
        encoder_hidden = 64
        proj_hidden = 64
        proj_out = 3
        """

        encoder_output = 4 #or something larger like 32
        #encoder_hidden = 64
        if encoder:
            self.encoder = encoder
        else:
            #try different values for encoder hidden on order of 64?
            #try different values for encoder output
            
            encoder_layers = [nn.Linear(4,encoder_hidden),
                              nn.BatchNorm1d(encoder_hidden),
                              nn.ReLU(inplace=True),
                              nn.Linear(encoder_hidden, encoder_hidden),
                              nn.BatchNorm1d(encoder_hidden),
                              nn.ReLU(inplace=True),
                              nn.Linear(encoder_hidden,encoder_hidden),
                              nn.BatchNorm1d(encoder_hidden),
                              nn.ReLU(inplace=True),
                              nn.Linear(encoder_hidden,encoder_output)
                             ]
            self.encoder = nn.Sequential(*encoder_layers)
        

        #self.projector = ProjectionMLP(encoder_output,proj_hidden,proj_out)
        #self.net = nn.Sequential(self.encoder,self.projector)
        self.net = self.encoder

    def forward(self, x):
        return self.net(x)

class TopPredictor(nn.Module):
    def __init__(self, encoder, predictor=None, predictor_output = 3, fine_tuning = False, predictor_hidden = 64):
        super().__init__()

        if predictor:
            self.predictor = predictor
        else:
            predictor_layers = [nn.Linear(3,predictor_hidden),
                              nn.BatchNorm1d(predictor_hidden),
                              nn.ReLU(inplace=True),
                              nn.Linear(predictor_hidden, predictor_hidden),
                              nn.BatchNorm1d(predictor_hidden),
                              nn.ReLU(inplace=True),
                              nn.Linear(predictor_hidden,predictor_output),
                             ]
            self.predictor = nn.Sequential(*predictor_layers)
        
        self.encoder = encoder
        self.net = nn.Sequential(self.predictor,self.encoder)
        
        if not fine_tuning:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, x):
        return self.net(x)


#====================================================

def contrastivelearning_loop(args, encoder=None):
    FOLDER_ROOT = os.path.expanduser("~/contrastive_learning/experiments/anugrahtests")
    #just equivalent to
    #FOLDER_ROOT = "/home/anugrah/contrastive_learning/experiments/anugrahtests"
    NEW_FOLDER_NAME = "/TESTING_NEW_FILE_SYSTEM"


    save_folder = FOLDER_ROOT+NEW_FOLDER_NAME

    os.mkdir(save_folder)



    def save_performance(epoch):
        #keep track of performance of the final model on the training and test set
        performance_log.write("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ \n")
        performance_log.write("\tSAVING PERFORMANCE FOR EPOCH " + str(epoch) + '\n')
        performance_log.write("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ \n \n")


        performance_log.write("TRAINING PERFORMANCE \n")
        performance_log.write("================================================== \n")
        total_loss = 0
        for it, (inputs1,inputs2,y) in enumerate(train_orbits_loader):
            z1 = get_z(inputs1.cuda())
            z2 = get_z(inputs2.cuda())
            loss = apply_loss(z1, z2)
            total_loss += loss[0].item()

            performance_log.write(("iter" + str(it) + "    loss = " + str(loss)+'\n'))
        
        performance_log.write("TOTAL LOSS = " + str(total_loss) + '\n')

    #keep extra keyword args for use in the dataloader
    dataloader_kwargs = dict(drop_last=True, pin_memory=True, num_workers=16)

    #create a train_loader to load training data
    train_orbits_dataset = neworbits.OrbitsDataset()
    train_orbits_loader = torch.utils.data.DataLoader(
        dataset = train_orbits_dataset,
        shuffle = True,
        batch_size = args.bsz,
        **dataloader_kwargs
    )
    

    #create a branch network
    main_branch = Branch(64, encoder=encoder).cuda()


    # optimization
    optimizer = get_optimizer(
        name='sgd',
        momentum=0.9,
        lr=args.lr,
        model=main_branch,
        weight_decay=args.wd
    )
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
    # macros
    b = main_branch.encoder

    # helpers
    def get_z(x):
        #return b(x.float())
        return b(x.float())

    def apply_loss(z1, z2):
        #using rmseNCE loss
        loss = 0.5 * normalmseNCE(z1, z2) + 0.5 * normalmseNCE(z2, z1)

        return loss


    #logging all print statements in the training to a file for later review
    training_log = open(save_folder+"/training_log.txt", "w")
    #keeping track of performance
    performance_log = open(save_folder+"/performance_log.txt", "w")


    # training

    
    SAVE_PROGRESS_PREFIX = "orbits_InfoNCE_encoder_model"
    
    for e in range(1, args.epochs + 1): #loop through each epoch
        # declaring train
        main_branch.train()

        training_log.write("=================\n")
        print("=================")

        for it, (input1, input2, y) in enumerate(train_orbits_loader):

            # zero the gradients
            main_branch.zero_grad()

            # forward pass
            z1 = get_z(input1.cuda())
            z2 = get_z(input2.cuda())
            loss = apply_loss(z1, z2)

            # optimization step
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
        
        #print the loss of the last iteration of that epoch
        training_log.write("epoch" + str(e) + "    loss = " + str(loss)+'\n')
        print("epoch" + str(e) + "    loss = " + str(loss))

        if e % 20 == 0 and e != 0:
            save_progress_path = save_folder + "/" + SAVE_PROGRESS_PREFIX + "_EPOCH" + str(e) + ".pt"
            torch.save(main_branch.encoder, save_progress_path)
            save_performance(e)

    save_progress_path = save_folder + "/" + SAVE_PROGRESS_PREFIX + "FINAL.pt"
    torch.save(main_branch.encoder, save_progress_path)
    

    training_log.close()
    performance_log.close()
    return main_branch.encoder

def toppredictortraining_loop(args):

    def save_performance(epoch):
        #keep track of performance of the final model on the training and test set
        performance_log.write("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ \n")
        performance_log.write("\tSAVING PERFORMANCE FOR EPOCH " + str(epoch) + '\n')
        performance_log.write("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ \n \n")


        performance_log.write("VALIDATION PERFORMANCE \n")
        performance_log.write("================================================== \n")
        total_loss = 0
        for it, (input1,input2,y) in enumerate(validate_orbits_loader):
            new_input = torch.from_numpy(np.vstack([input1,input2])).to(device)
            new_y = torch.from_numpy(np.vstack([y[:,2:],y[:,2:]])).to(device)

            z1 = get_z(new_input)
            loss = apply_loss(z1, new_y)
            total_loss += loss.item()

            performance_log.write(("iter" + str(it) + "    loss = " + str(loss)+'\n'))
        performance_log.write("TOTAL LOSS = " + str(total_loss) + '\n')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    FOLDER_ROOT = os.path.expanduser("~/contrastive_learning/experiments/anugrahtests")
    #just equivalent to
    #FOLDER_ROOT = "/home/anugrah/contrastive_learning/experiments/anugrahtests"
    NEW_FOLDER_NAME = "/TESTING_NEW_FILE_SYSTEM"


    save_folder = FOLDER_ROOT+NEW_FOLDER_NAME

    os.mkdir(save_folder)


    ENCODER_PATH = FOLDER_ROOT + "/orbits3DwnormalMSE/orbits_InfoNCE_encoder_modelFINAL.pt"
    branch_encoder = torch.load(ENCODER_PATH, map_location=torch.device('cuda'))

    if args.fine_tune:
        PREDICTOR_PATH = FOLDER_ROOT + "/orbitsSupervisedTopPredictorver1/top_predictor_orbits_topPredictor_final.pt"
        top_predictor = torch.load(PREDICTOR_PATH, map_location=torch.device('cuda'))
    
        full_net = TopPredictor(encoder=branch_encoder, predictor=top_predictor, fine_tuning=args.fine_tune, predictor_hidden=64).cuda()
    else:
        full_net = TopPredictor(encoder=branch_encoder, predictor_hidden=64).cuda()

    #create a train_loader to load training data
    train_orbits_dataset = versatileorbits.OrbitsDataset(exclude_values=['H','L'],val_lower=0.25,val_higher=0.75)
    train_orbits_loader = torch.utils.data.DataLoader(
        dataset = train_orbits_dataset,
        shuffle = True,
        batch_size = args.bsz,
    )

    validate_orbits_dataset = versatileorbits.OrbitsDataset()
    validate_orbits_loader = torch.utils.data.DataLoader(
        dataset = validate_orbits_dataset,
        shuffle = True,
        batch_size = args.bsz,
    )    

    # optimization
    optimizer = get_optimizer(
        name='sgd',
        momentum=0.9,
        lr=args.lr,
        model=full_net,
        weight_decay=args.wd
    )
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
    # macros
    fn_encoder = full_net.encoder
    fn_predictor = full_net.predictor

    # helpers
    def get_z(x):
        return fn_predictor(fn_encoder(x.float()).float())

    def apply_loss(inputs, targets):
        loss_function = nn.MSELoss()
        loss = loss_function(inputs.float(),targets.float())
        return loss

    #logging all print statements in the training to a file for later review
    training_log = open(save_folder+"/training_log.txt", "w")
    #keeping track of performance
    performance_log = open(save_folder+"/performance_log.txt", "w")


    # training    
    PREDICTOR_SAVE_PROGRESS_PREFIX = "/top_predictor_orbits_topPredictor"
    ENCODER_SAVE_PROGRESS_PREFIX = "/top_predictor_orbits_Encoder"
    
    
    for e in range(1, args.epochs + 1): #loop through each epoch
        # declaring train
        full_net.train()

        training_log.write("=================\n")
        print("=================")

        for it, (input1, input2, y) in enumerate(train_orbits_loader):

            new_input = torch.from_numpy(np.vstack([input1,input2])).to(device)
            new_y = torch.from_numpy(np.vstack([y[:,2:],y[:,2:]])).to(device)

            # zero the gradients
            full_net.zero_grad()

            # forward pass
            z = get_z(new_input.to(device))
            loss = apply_loss(z,new_y)

            # optimization step
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
        
        #print the loss of the last iteration of that epoch
        training_log.write("epoch" + str(e) + "    loss = " + str(loss)+'\n')
        print("epoch" + str(e) + "    loss = " + str(loss))

        if e % 20 == 0 and e != 0:
            predictor_save_progress_path = save_folder + PREDICTOR_SAVE_PROGRESS_PREFIX + str(e) + ".pt"
            if args.fine_tune:
                encoder_save_progress_path = save_folder + ENCODER_SAVE_PROGRESS_PREFIX + str(e) + ".pt"
                torch.save(full_net.encoder, encoder_save_progress_path)
            torch.save(full_net.predictor, predictor_save_progress_path)
            save_performance(e)

    predictor_save_progress_path_final = save_folder + PREDICTOR_SAVE_PROGRESS_PREFIX + "_final.pt"
    
    if args.fine_tune:
        encoder_save_progress_path_final = save_folder + ENCODER_SAVE_PROGRESS_PREFIX + "_final.pt"
        torch.save(full_net.encoder, encoder_save_progress_path_final)
    torch.save(full_net.predictor, predictor_save_progress_path_final)
    

    training_log.close()
    performance_log.close()


if __name__ == '__main__':
    pass
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--epochs', default=100, type=int)
    # parser.add_argument('--lr', default=0.02, type=float)
    # parser.add_argument('--bsz', default=512, type=int)
    # parser.add_argument('--wd', default=0.001, type=float)
    # parser.add_argument('--warmup_epochs', default=5, type=int)
    # parser.add_argument('--fine_tune', default=False, type=bool)

    # args = parser.parse_args()
    # toppredictortraining_loop(args)