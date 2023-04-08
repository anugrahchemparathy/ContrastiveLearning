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


def track_plot(fnames):
    num_epochs = 1500
    interval = 20

    """
    ratios = {fname: [] for fname in fnames}
    for fname in fnames:
        for epoch in range(20, num_epochs, interval):
            ratios[fname].append(main_plot(fname, str(epoch)))
    for fname in fnames:
        plt.plot(np.arange(20, num_epochs, step=interval), ratios[fname], label=fname[fname.rindex("_") + 1:])
    plt.legend()
    plt.show()
    """
    #times = {fname + "_" + str(i): [] for i in ["1", "2", "3"] for fname in fnames}
    times = {fname: [] for fname in fnames}
    for fname in fnames:
        for i in ["1", "2", "3", "4", "5"]:
            for epoch in ["final"] + list(range(20, num_epochs, interval)):
                print(f"{fname}, {i}, {epoch}")
                corr = main_plot(fname + "_" + i, epoch)
                if (corr > 0.5 and epoch != "final") or (corr <= 0.5 and epoch == "final"):
                    times[fname].append(num_epochs if epoch == "final" else int(epoch))
                    break
        #times[fname] = sum(times[fname]) / len(times[fname])
    print(times)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fname', type=str)
    parser.add_argument('--image', action='store_true')
    parser.add_argument('--id', default='final', type=str)
    parser.add_argument('--server', action='store_true')
    parser.add_argument('--low_only', action='store_true')

    args = parser.parse_args()
    #fnames = ["double_pendulum_t" + x for x in ["1", "5", "10", "100"]]
    fnames = ["double_pendulum_t" + x for x in ["05", "08", "1", "3", "5", "10"]]
    track_plot(fnames)
    #main_plot_2(args.fname, args.id)
