import torch
import torch.nn.functional as F
import torch.nn as nn



from models import branch, predictor



from cl.tools.seed import set_deterministic
from cl.optimizers.LR_scheduler import LR_Scheduler

from orbit_datasets import neworbits, versatileorbits, staticorbits
from Losses.NCE_losses import infoNCE, rmseNCE, normalmseNCE






def training_loop(args, encoder = None):


    dataloader_kwargs = dict(drop_last=True, pin_memory=True, num_workers=16)
    train_orbits_dataset = neworbits.OrbitsDataset()
    train_orbits_loader = torch.utils.data.DataLoader(
        dataset = train_orbits_dataset,
        shuffle = True,
        batch_size = args.bsz,
        **dataloader_kwargs
    )

    if not encoder: encoder = branch.branchEncoder()

    optimizer = torch.optim.SGD(encoder.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
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


    total_loss = 0
    for e in range(1, args.epochs + 1): #loop through each epoch
        # declaring train
        # main_branch.train()

        for it, (input1, input2, y) in enumerate(train_orbits_loader):
            # zero the gradients
            encoder.zero_grad()

            # forward pass
            z1 = get_z(input1.cuda())
            z2 = get_z(input2.cuda())
            loss = apply_loss(z1, z2)
            total_loss += loss.item()

            # optimization step
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
        
        #print the loss of the last iteration of that epoch
        print("epoch" + str(e) + "    loss = " + str(loss))


    save_progress_path = save_folder + "/" + SAVE_PROGRESS_PREFIX + "FINAL.pt"
    torch.save(main_branch.encoder, save_progress_path)
    

    training_log.close()
    performance_log.close()
    return main_branch.encoder











