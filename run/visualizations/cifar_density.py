from ldcl.plot.plot import VisPlot
from ldcl.plot.embed import embed
from ldcl.plot.color import get_cmap

from ldcl.data.physics import get_dataset
from ldcl.tools.device import get_device

from sklearn.decomposition import PCA
import argparse

import subprocess
import math

import torch
import torchvision
import torchvision.transforms as T
import torch.nn.functional as F
import numpy as np

from tqdm import tqdm

from ldcl.models.cifar_resnet import Branch

import faiss

device = get_device(idx=7)

normalize = T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
single_transform = T.Compose([T.ToTensor(), normalize])


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

def eval_loop(encoder, ind=None):
    # dataset
    train_transform = T.Compose([
        T.RandomResizedCrop(32, interpolation=T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalize
    ])
    test_transform = T.Compose([
        T.Resize(36, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(32),
        T.ToTensor(),
        normalize
    ])

    train_loader = torch.utils.data.DataLoader(
        dataset=torchvision.datasets.CIFAR10('../../data', train=True, transform=train_transform, download=True),
        shuffle=True,
        batch_size=256,
        pin_memory=True,
        num_workers=args.num_workers,
        drop_last=True
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=torchvision.datasets.CIFAR10('../../data', train=False, transform=test_transform, download=True),
        shuffle=False,
        batch_size=256,
        pin_memory=True,
        num_workers=args.num_workers
    )

    classifier = torch.nn.Linear(512, 10)
    # optimization
    optimizer = torch.optim.SGD(
        classifier.parameters(),
        momentum=0.9,
        lr=30,
        weight_decay=0
    )

    # training
    for e in range(1, 2):
        # declaring train
        classifier.train()
        encoder.eval()
        # epoch
        for it, (inputs, y) in enumerate(train_loader, start=(e - 1) * len(train_loader)):
            # adjust
            adjust_learning_rate(epochs=100,
                                 warmup_epochs=0,
                                 base_lr=30,
                                 optimizer=optimizer,
                                 loader=train_loader,
                                 step=it)
            # zero grad
            classifier.zero_grad()

            def forward_step():
                with torch.no_grad():
                    b = encoder(inputs.cuda())
                logits = classifier(b)
                loss = F.cross_entropy(logits, y.cuda())
                return loss

            # optimization step
            loss = forward_step()
            loss.backward()
            optimizer.step()

        if e % 10 == 0:
            accs = []
            classifier.eval()
            for idx, (images, labels) in enumerate(test_loader):
                with torch.no_grad():
                    b = encoder(images.cuda())
                    preds = classifier(b).argmax(dim=1)
                    hits = (preds == labels.cuda()).sum().item()
                    accs.append(hits / b.shape[0])
            accuracy = np.mean(accs) * 100
            # final report of the accuracy
            line_to_print = (
                f'seed: {ind} | accuracy (%) @ epoch {e}: {accuracy:.2f}'
            )
            print(line_to_print)

    probs = []
    embeds = []
    lbls = []
    for idx, (images, labels) in enumerate(test_loader):
        with torch.no_grad():
            b = encoder(images.cuda())
            embeds.append(b)
            b = classifier(b)
            b = torch.nn.Softmax(b,dim=1)
            probs.append(b[torch.arange(0,256),labels])
            lbls.append(labels)
    embeds = torch.cat(embeds)
    probs = torch.stack(probs)
    lbls = torch.stack(lbls)
    print(embeds.shape, probs.shape, "sholud be ~(10000,512) and ~(10000,")

    index = faiss.IndexFlatL2(512)
    index.add(embeds.detach().cpu().numpy())
    print(index.ntotal, "should be ~10000")
    dists, _ = index.search(embeds, 21)
    dists = dists[:, 1:]
    print(dists.shape, "should be ~10000 x 20", dists[:10, :10], "some distances")
    dists = np.mean(dists, dim=1)

    return probs, dists, lbls

def main_plot(args):
    test_loader = torch.utils.data.DataLoader(
        dataset=torchvision.datasets.CIFAR10(
            '../../data', train=False, transform=single_transform, download=True,
        ),
        shuffle=False,
        batch_size=512,
        pin_memory=True,
        num_workers=8
    )
    branch = Branch()
    state_dict = torch.load(f"../cifar10_ckpts/011923_epochs_{args.epochs}_croplow_{args.croplow:.1f}/{args.id}.pth", map_location=device)["state_dict"]
    branch.load_state_dict(state_dict)
    branch.to(device)

    probs,dists,labels = eval_loop(branch)
    embeds = np.stack((probs,dists),dim=1)
    vals = { "labels": labels}

    """
    # Dim reduction (2d only).
    pca = PCA(n_components=2) # dimensionality reduction for 2D
    single_orbit_embeds = pca.fit_transform(single_orbit_embeds)
    oneD_span_embeds = pca.transform(oneD_span_embeds)
    """

    # Plot

    def cmap_one():
        plot = VisPlot(2)
        plot.add_with_cmap(embeds, vals, cmap="tab10", cby="labels", size=3, outline=False)
        return plot

    plot = cmap_one()

    plot.show()
    if args.server:
        subprocess.run('python -m http.server', shell=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--croplow', type=float)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--id', default='final', type=str)
    parser.add_argument('--server', action='store_true')

    args = parser.parse_args()
    main_plot(args)

