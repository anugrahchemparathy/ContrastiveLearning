from ldcl.plot.plot import VisPlot
from ldcl.plot.embed import embed
from ldcl.plot.color import get_cmap

from ldcl.data.physics import get_dataset
from ldcl.tools.device import get_device

from sklearn.decomposition import PCA

device = get_device()

#dataset, _ = get_dataset("../data_configs/orbit_images_medxl.json", "../../saved_datasets")
dataset, _ = get_dataset("../data_configs/orbit_config_default.json", "../../saved_datasets")
embeds, vals = embed("../saved_models/supervised_test/final_encoder.pt", dataset, device=device)

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
    plot = VisPlot(3, num_subplots=3) # 3D plot, 2 for 2D plot
    plot.add_with_cmap(embeds, vals, cmap="viridis", cby=["phi0", "H", "L"], size=1.5, outline=False)
    return plot

def cmap_one():
    plot = VisPlot(3)
    print(embeds.shape)
    plot.add_with_cmap(embeds, vals, cmap="viridis", cby="L", size=3, outline=True)
    return plot

#plot = add_demo()
plot = cmap_three()

plot.show()
