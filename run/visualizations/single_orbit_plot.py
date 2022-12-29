from ldcl.plot.plot import VisPlot
from ldcl.plot.embed import embed
from ldcl.plot.color import get_cmap

from ldcl.data.physics import get_dataset

from sklearn.decomposition import PCA

single_orbit, _ = get_dataset("../data_configs/single_orbit.json", "../../saved_datasets")
oneD_span, _ = get_dataset("../data_configs/H_vary.json", "../../saved_datasets")

single_orbit_embeds, single_orbit_vals = embed("../saved_models/rmse_vanilla/final_encoder.pt", single_orbit)
oneD_span_embeds, oneD_span_vals = embed("../saved_models/rmse_vanilla/final_encoder.pt", oneD_span)

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

def add_with_cmap_demo():
    plot = VisPlot(3, num_subplots=2) # 3D plot, 2 for 2D plot

    plot.add_with_cmap(oneD_span_embeds, oneD_span_vals, cmap="viridis", cby=["x", "H"], size=0.5, outline=False)
    plot.add_with_cmap(single_orbit_embeds, single_orbit_vals, cmap="plasma", cby=["x", "H"], size=4, outline=False)

    return plot

#plot = add_demo()
plot = add_with_cmap_demo()

plot.show()
