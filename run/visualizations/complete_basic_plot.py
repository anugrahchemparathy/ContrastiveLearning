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
        dataset, _ = get_dataset("../data_configs/orbit_completeimages_medxl2.json", "../../saved_datasets")
        #dataset, _ = get_dataset("../data_configs/orbit_resz_medxl.json", "../../saved_datasets")
    else:
        dataset, _ = get_dataset("../data_configs/orbit_config_default.json", "../../saved_datasets")

    embeds, vals = embed(f"../saved_models/{args.fname}/{args.id}_encoder.pt", dataset, device=device)

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

    # option 1: add syntax (more flexible, more work)
    def add_demo():
        plot = VisPlot(3) # 3D plot, 2 for 2D plot

        single_orbit_colors = viridis(single_orbit_vals["x"])
        oneD_span_colors = plasma(oneD_span_vals["x"])

        # Note that a list of sizes can be passed in too.

        plot.add(oneD_span_embeds,
            size=2,
            color=oneD_span_colors,
            label=oneD_span_vals, outline=False)
        plot.add(single_orbit_embeds,
            size=4,
            color=single_orbit_colors,
            label=single_orbit_vals, outline=False)

        return plot

    # option 2: add_with_cmap syntax (fastest, less flexible)
    # note that you can put a list of cmap/cby to make multiple plots
    # this one has both x and H, not just x

    def cmap_three():
        nonlocal embeds

        plot = VisPlot(3, num_subplots=4) # 3D plot, 2 for 2D plot
        print(embeds.shape)
        plot.add_with_cmap(embeds, vals, cmap=["husl", "viridis", "viridis", "viridis"], cby=["phi0", "H", "L", "ecc"], size=1.5, outline=False)
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

