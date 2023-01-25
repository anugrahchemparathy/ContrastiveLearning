import numpy as np

from ldcl.plot.plot import VisPlot
from ldcl.plot.embed import embed
from ldcl.plot.color import get_cmap

from ldcl.data.physics import get_dataset
from ldcl.tools.device import get_device

from sklearn.decomposition import PCA
import argparse

import subprocess

device = get_device(idx=7)


def main_plot(args):
    if args.image:
        dataset, _ = get_dataset("../data_configs/orbit_images_medxl2.json", "../../saved_datasets")
        #dataset, _ = get_dataset("../data_configs/orbit_resz_medxl.json", "../../saved_datasets")
        single_orbit, _ = get_dataset("../data_configs/single_orbit_image.json", "../../saved_datasets")
    else:
        dataset, _ = get_dataset("../data_configs/orbit_config_default.json", "../../saved_datasets")
        print(_)
        single_orbit, _ = get_dataset("../data_configs/single_orbit.json", "../../saved_datasets")

    embeds, vals = embed(f"../saved_models/{args.fname}/{args.id}_encoder.pt", dataset, device=device)
    so_embeds, so_vals = embed(f"../saved_models/{args.fname}/{args.id}_encoder.pt", single_orbit, device=device)
    so_embeds = so_embeds[::10]
    for key in so_vals.keys():
        so_vals[key] = so_vals[key][::10]
    
    # mask = np.less(vals['phi0'], 3.14)
    # embeds = embeds[mask]
    # for key in vals.keys():
    #     vals[key] = vals[key][mask]

    """
    # Dim reduction (2d only).
    pca = PCA(n_components=2) # dimensionality reduction for 2D
    single_orbit_embeds = pca.fit_transform(single_orbit_embeds)
    oneD_span_embeds = pca.transform(oneD_span_embeds)
    """

    # Colors

    viridis = get_cmap('viridis')
    plasma = get_cmap('plasma')
    blank = get_cmap('blank')

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
        print(embeds.shape)
        plot.add_with_cmap(embeds, vals, cmap="viridis", cby="y", size=3, outline=True)
        return plot

    #plot = add_demo()
    plot = cmap_three()
    #plot = cmap_one()

    plot.show()
    if args.server:
        subprocess.run('python -m http.server', shell=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fname', type=str)
    parser.add_argument('--image', action='store_true')
    parser.add_argument('--id', default='final', type=str)
    parser.add_argument('--server', action='store_true')

    args = parser.parse_args()
    main_plot(args)

