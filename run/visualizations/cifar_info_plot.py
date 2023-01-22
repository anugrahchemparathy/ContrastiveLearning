from ldcl.plot.plot import VisPlot
from ldcl.plot.embed import embed
from ldcl.plot.color import get_cmap

from ldcl.data.physics import get_dataset
from ldcl.tools.device import get_device

from sklearn.decomposition import PCA
import argparse

import subprocess

import torch
import torchvision
import torchvision.transforms as T
import numpy as np

from tqdm import tqdm

from ldcl.models.cifar_resnet import Branch

device = get_device(idx=7)

normalize = T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
single_transform = T.Compose([T.ToTensor(), normalize])

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
    branch.eval()

    embeds = []
    targets = []
    for batch in tqdm(test_loader):
        ind = batch[0].type(torch.float32).to(device)
        embeds.append(branch(ind))
        targets = targets + batch[1].cpu().numpy().tolist()
    embeds = torch.cat(embeds, dim=0).detach().cpu().numpy()
    targets = np.array(targets)
    vals = {
        "targets": targets
    }

    mask = np.equal(vals["targets"], 2)
    class_embeds = embeds[mask]
    class_vals = {}
    for key in vals:
        class_vals[key] = vals[key][mask]

    """
    # Dim reduction (2d only).
    pca = PCA(n_components=2) # dimensionality reduction for 2D
    single_orbit_embeds = pca.fit_transform(single_orbit_embeds)
    oneD_span_embeds = pca.transform(oneD_span_embeds)
    """

    # Plot

    def cmap_three():
        nonlocal embeds

        plot = VisPlot(3, num_subplots=5) # 3D plot, 2 for 2D plot
        print(embeds.shape)
        plot.add_with_cmap(embeds, vals, cmap=["husl", "viridis", "viridis", "viridis", "viridis"], cby=["phi0", "H", "L", "x", "v.x"], size=1.5, outline=False)
        plot.add_with_cmap(so_embeds, so_vals, cmap=["husl", "viridis", "viridis", "viridis", "viridis"], cby=["phi0", "H", "L", "x", "v.x"], size=2.5, outline=True)
        return plot

    def cmap_one():
        plot = VisPlot(3)
        plot.add_with_cmap(embeds, vals, cmap="tab10", cby="targets", size=1.5, outline=False)
        plot.add_with_cmap(class_embeds, class_vals, cmap="tab10", cby="targets", size=3, outline=True)
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

