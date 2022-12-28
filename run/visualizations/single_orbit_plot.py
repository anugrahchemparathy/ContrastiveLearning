from ldcl.plot.plot import VisPlot
from ldcl.plot.embed import embed
from ldcl.plot.color import get_cmap

from ldcl.data.physics import get_dataset

from sklearn.decomposition import PCA

single_orbit, _ = get_dataset("../data_configs/single_orbit.json", "../../saved_datasets")
oneD_span, _ = get_dataset("../data_configs/H_vary.json", "../../saved_datasets")

single_orbit_embeds, single_orbit_vals = embed("../saved_models/rmse_vanilla/final_encoder.pt", single_orbit)
oneD_span_embeds, oneD_span_vals = embed("../saved_models/rmse_vanilla/final_encoder.pt", oneD_span)

single_orbit_embeds = PCA(n_components=2).fit_transform(single_orbit_embeds)
oneD_span_embeds = PCA(n_components=2).fit_transform(oneD_span_embeds)

# Colors

viridis = get_cmap('viridis')
plasma = get_cmap('plasma')
#ylorrd = get_cmap('YlOrRd')
#pubugn = get_cmap('PuBuGn')
blank = get_cmap('blank')

# Creating size/colors

single_orbit_colors = viridis(single_orbit_vals["x"])
oneD_span_colors = plasma(oneD_span_vals["x"])

single_orbit_sizes = 4
oneD_span_sizes = 2

# Plot

plot = VisPlot(2) # 3D plot, 2 for 2D plot
plot.add(single_orbit_embeds, 
    size=single_orbit_sizes, 
    color=single_orbit_colors, 
    labels=single_orbit_vals, 
    outline=False)
plot.add(oneD_span_embeds, 
    size=oneD_span_sizes, 
    color=oneD_span_colors, 
    labels=oneD_span_vals, outline=False)

plot.show()
