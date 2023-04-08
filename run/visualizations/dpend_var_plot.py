import numpy as np
from ldcl.plot.embed import embed
#from ldcl.plot.color import get_cmap
import matplotlib.pyplot as plt

from ldcl.data.physics import get_dataset
from ldcl.tools.device import get_device

from sklearn.decomposition import PCA
import argparse
#import diptest
import scipy
from sklearn.linear_model import LinearRegression

import subprocess

device = get_device(idx=7)
bin_size = 0.05

def main_plot(fname, id):
    dataset, _ = get_dataset("double_pendulum", "../../saved_datasets")
    embeds, vals = embed(f"../saved_models/{fname}/{id}_encoder.pt", dataset, device=device)

    min_energy = np.min(vals["one"])
    max_energy = np.max(vals["one"])
    variances = []
    lowers = []
    for lower in np.arange(min_energy, max_energy, step=bin_size):
        lowers.append(lower)
        points = embeds[np.logical_and(np.less_equal(lower, vals["one"]), np.less(vals["one"], lower + bin_size))]
        if np.shape(points)[0] > 0:
            variances.append(np.sum(np.square(np.std(points, axis=0))))
        else:
            del lowers[-1]
    fit = np.polynomial.polynomial.Polynomial.fit(lowers, variances, 4)
    #print(diptest.diptest(np.array(variances)))

    small_embed = np.max(fit(np.arange(min_energy, -1.5)))
    large_embed = np.max(fit(np.arange(0, max_energy)))
    #print(fit(min_energy) / fit(max_energy))
    #print(small_embed / large_embed)
    #return small_embed / large_embed
    print(small_embed - fit(1.0))
    plt.scatter(lowers, variances)
    plt.scatter(np.arange(min_energy, max_energy, step=bin_size),fit(np.arange(min_energy, max_energy, step=bin_size)))
    plt.show()

def main_plot_2(fname, id):
    dataset, _ = get_dataset("double_pendulum", "../../saved_datasets")
    embeds, vals = embed(f"../saved_models/{fname}/{id}_encoder.pt", dataset, device=device)

    mask = vals["one"] < -1.50
    embeds = embeds[mask]
    del vals["params"]
    for key in vals.keys():
        vals[key] = vals[key][mask]

    #linreg1 = np.linalg.lstsq(embeds, vals["two"])[1]
    #linreg2 = np.linalg.lstsq(embeds, vals["three"])[1] / np.shape(embeds)[0]
    #denom = np.square(np.std(vals["three"]))
    #slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(embeds, vals["three"])
    model1 = LinearRegression()
    model1.fit(embeds, vals['two'])
    model2 = LinearRegression()
    model2.fit(embeds, vals['three'])
    return min(model1.score(embeds, vals['two']), model2.score(embeds, vals['three']))

    #return max(linreg1, linreg2)
    #return 1 - linreg2/denom
    #return r_value ** 2

def track_plot(fnames):
    num_epochs = 1500
    interval = 20

    ratios = {fname: [] for fname in fnames}
    for fname in fnames:
        for epoch in range(20, num_epochs, interval):
            ratios[fname].append(main_plot_2(fname, str(epoch)))
    for fname in fnames:
        plt.plot(np.arange(20, num_epochs, step=interval), ratios[fname], label=fname[fname.rindex("_") + 1:])
    plt.legend()
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fname', type=str)
    parser.add_argument('--image', action='store_true')
    parser.add_argument('--id', default='final', type=str)
    parser.add_argument('--server', action='store_true')
    parser.add_argument('--low_only', action='store_true')

    args = parser.parse_args()
    fnames = ["double_pendulum_t" + x for x in ["1", "5", "10", "100"]]
    track_plot([args.fname])
    #main_plot_2(args.fname, args.id)
