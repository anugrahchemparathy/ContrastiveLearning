from ldcl.plot.plot import VisPlot
from ldcl.plot.embed import embed
from ldcl.plot.color import get_cmap

from ldcl.data.physics import get_dataset
from ldcl.tools.device import get_device

from sklearn.decomposition import PCA

import imageio
import glob

device = get_device()

dataset, _ = get_dataset("../data_configs/orbit_images_medxl.json", "../../saved_datasets")
#dataset, _ = get_dataset("../data_configs/orbit_config_default.json", "../../saved_datasets")
#embeds, vals = embed("../saved_models/newnoise_test/final_encoder.pt", dataset, device=device)
model = "../saved_models/simsiam_imagetest/"

# Colors

viridis = get_cmap('viridis')
plasma = get_cmap('plasma')
blank = get_cmap('blank')

def cmap_three(embeds, vals):
    plot = VisPlot(3, num_subplots=3) # 3D plot, 2 for 2D plot
    plot.add_with_cmap(embeds, vals, cmap="viridis", cby=["phi0", "H", "L"], size=1.5, outline=False)
    return plot

def cmap_one(embeds, vals):
    plot = VisPlot(3)
    plot.add_with_cmap(embeds, vals, cmap="viridis", cby="phi0", size=3, outline=False)
    return plot

def make_image(embeds, vals, i):
    plot = cmap_one(embeds, vals)
    plot.fig.write_image(f'train_progression/{i}e.png', width=1800, height=1800)
    return imageio.imread(f'train_progression/{i}e.png')

#plot = add_demo()
images = []
embeds, vals = embed(model + "start_encoder.pt", dataset, device=device)
images = images + [make_image(embeds, vals, "start")] * 12
for i in range(len(glob.glob(model + "*_encoder.pt")) - 2):
    embeds, vals = embed(model + f"{i:02d}_encoder.pt", dataset, device=device)
    images.append(make_image(embeds, vals, str(i)))
embeds, vals = embed(model + "final_encoder.pt", dataset, device=device)
images = images + [make_image(embeds, vals, "final")] * 12

imageio.mimsave('training.gif', images, fps=12)
