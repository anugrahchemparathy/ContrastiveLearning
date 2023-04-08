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
    dataset, _ = get_dataset("double_pendulum", "../../saved_datasets")
    embeds, vals = embed(f"../saved_models/{args.fname}/{args.id}_encoder.pt", dataset, device=device)
    so_dataset, _ = get_dataset("double_pendulum_fixed", "../../saved_datasets")
    so_embeds, so_vals = embed(f"../saved_models/{args.fname}/{args.id}_encoder.pt", so_dataset, device=device)

    embeds = embeds[::3]
    for key in vals.keys():
        vals[key] = vals[key][::3]

    if args.low_only:
        mask = vals["one"] < -1.50
        embeds = embeds[mask]
        del vals["params"]
        for key in vals.keys():
            vals[key] = vals[key][mask]

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

        plot = VisPlot(3, num_subplots=3) # 3D plot, 2 for 2D plot
        plot.add_with_cmap(embeds, vals, cmap=["viridis", "viridis", "viridis"], cby=["one", "two", "three"], size=1.5, outline=False)
        plot.add_with_cmap(so_embeds, so_vals, cmap=["plasma", "plasma", "plasma"], cby=["one", "two", "three"], size=4, outline=True)
        #plot.add_with_cmap(so_embeds, so_vals, cmap=["husl", "viridis", "viridis", "viridis"], cby=["phi0", "H", "L", "x"], size=2.5, outline=True)
        return plot

    def cmap_one():
        plot = VisPlot(3)
        plot.add_with_cmap(embeds, vals, cmap="viridis", cby=["d"], size=1.5)
        plot.add_with_cmap(so_embeds, so_vals, cmap="plasma", cby=["d"], size=4)
        #plot.add_with_cmap(so_embeds, so_vals, cmap="black", cby=["L"], size=4.8, outline=True)
        return plot

    #plot = add_demo()
    #plot = cmap_three()
    plot = cmap_one()
    camera = dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=11/7, y=-2.25/7, z=0.25/7)
    )

    #plot.fig.update_layout(scene_camera=camera)

    #plot.set_title([""])
    #plot.fig.write_image("kepler_embed.pdf")
    plot.show()
    if args.server:
        subprocess.run('python -m http.server', shell=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fname', type=str)
    parser.add_argument('--image', action='store_true')
    parser.add_argument('--id', default='final', type=str)
    parser.add_argument('--server', action='store_true')
    parser.add_argument('--low_only', action='store_true')

    args = parser.parse_args()
    main_plot(args)

