import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import math
import numpy as np

def adjust_learning_rate(epochs, warmup_epochs, base_lr, optimizer, loader, step):
    max_steps = epochs * len(loader)
    warmup_steps = warmup_epochs * len(loader)
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = 0
        lr = base_lr * q + end_lr * (1 - q)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def eval_loop(encoder, train_loader, test_loader, ind=None, fp16=False, do_print=False, device=None, epochs=100):
    classifier = nn.Linear(1024, 10).to(device)
    optimizer = torch.optim.SGD(
        classifier.parameters(),
        momentum=0.9,
        lr=30,
        weight_decay=0
    )
    scaler = GradScaler()

    # training
    for e in range(1, epochs + 1):
        # declaring train
        classifier.train()
        encoder.eval()
        # epoch
        for it, (inputs, _, y) in enumerate(train_loader, start=(e - 1) * len(train_loader)):
            # adjust
            adjust_learning_rate(epochs=epochs,
                                 warmup_epochs=0,
                                 base_lr=30,
                                 optimizer=optimizer,
                                 loader=train_loader,
                                 step=it)
            # zero grad
            classifier.zero_grad()

            def forward_step():
                nonlocal y
 
                with torch.no_grad():
                    b = encoder(inputs.to(device))
                logits = classifier(b)
                y = y.long()
                loss = F.cross_entropy(logits, y.to(device))
                return loss

            # optimization step
            if fp16:
                with autocast():
                    loss = forward_step()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = forward_step()
                loss.backward()
                optimizer.step()

        if e % 10 == 0:
            accs = []
            classifier.eval()
            for idx, (images, _, labels) in enumerate(test_loader):
                with torch.no_grad():
                    if fp16:
                        with autocast():
                            b = encoder(images.to(device))
                            preds = classifier(b).argmax(dim=1)
                    else:
                        b = encoder(images.to(device))
                        preds = classifier(b).argmax(dim=1)
                    hits = (preds == labels.to(device)).sum().item()
                    accs.append(hits / b.shape[0])
            accuracy = np.mean(accs)
            # final report of the accuracy
            if do_print:
                line_to_print = (
                    f'seed: {ind} | accuracy (%) @ epoch {e}: {accuracy:.2f}'
                )
                print(line_to_print)
    return accuracy
