import matplotlib.pyplot as plt
import numpy as np

res = np.load("cifar_results.npy")

fig, axs = plt.subplots(3,1)
#fig.set_size_inches(18.5, 18.5)
cmap = plt.get_cmap('cool')
ns = [0.2,0.5,0.8,1.0]
ref = [25,50,75,100,150,200,400,800,1600]
show = [0,3,8]

#fig.text(1.1, 0.5, "Similarity", fontsize=12)
for exp in range(len(axs)):
    for n in range(res.shape[1]):
        top = res[show[exp], n, :]
        top = top - np.min(top)
        top = top / np.max(top)
        axs[exp].plot(np.arange(1600, step=int(1600 / res.shape[2])) + int(1600 / res.shape[2]), top, color=cmap(ns[n] * 0.8), label=f"{1 - (ns[n] - 0.2) * 5/4:.1f}")
        axs[exp].set_ylim(-0.1,1.1)
        axs[exp].set_xlim(0,1600)
        axs[exp].spines['top'].set_visible(False)
        axs[exp].spines['right'].set_visible(False)
        #axs[exp].spines['bottom'].set_visible(False)
        axs[exp].spines['left'].set_visible(False)
        axs[exp].set_yticks([])
        axs[exp].yaxis.set_label_position("right")
        axs[exp].set_ylabel(f"Reference: {ref[show[exp]]}e")
        axs[exp].yaxis.label.set_fontsize(8)
        if exp != len(axs) - 1:
            axs[exp].set_xticks([])
        else:
            axs[exp].set_xlabel("Epochs")
            axs[exp].legend(loc='lower right', ncol=4, frameon=False)
            axs[exp].text(0.66,0.35,"Augmentation strength", transform=axs[exp].transAxes)
        if exp == 1:
            ax2 = axs[exp].twinx()
            ax2.yaxis.set_label_position("left")
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            ax2.spines['left'].set_visible(False)
            ax2.set_yticks([])
            ax2.set_ylabel("Normalized Similarity")
            
plt.savefig('cifar_results.pdf')
