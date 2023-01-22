import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
import seaborn as sns

#name = "many_loss_control"
#name = "many_loss_test"
name = "many_loss_noiseshort"

path = "saved_models/" + name + "/"

losses = {
    "5": np.load(path + "five.npy"),
    "10": np.load(path + "ten.npy"),
    "20": np.load(path + "twenty.npy"),
    "50": np.load(path + "fifty.npy"),
    "100": np.load(path + "full.npy")
}
colors = {
    "5": sns.color_palette("husl", 5)[0],
    "10": sns.color_palette("husl", 5)[1],
    "20": sns.color_palette("husl", 5)[2],
    "50": sns.color_palette("husl", 5)[3],
    "100": sns.color_palette("husl", 5)[4],
}

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

print(losses["5"])
for key in losses.keys():
    cidx = 0
    while 'inf' in str(losses[key][0]):
        cidx += 1
        losses[key][0] = losses[key][cidx] # first non inf number
    for idx in range(losses[key].shape[0]):
        if 'inf' in str(losses[key][idx]):
            losses[key][idx] = losses[key][idx - 1]
    #losses[key] = losses[key][5:]
    losses[key] = losses[key] - losses[key][-1]
    losses[key] = losses[key] / losses[key][0]

    #losses[key] = smooth(losses[key], 30)
    losses[key] = savgol_filter(losses[key], 31, 3)
    plt.plot(3 * np.arange(losses[key].shape[0]), losses[key] / losses[key][0], label=key, linewidth=1, color=colors[key])

plt.hlines(0, 0, 1, linewidth=0.2)
plt.legend()
plt.show()

