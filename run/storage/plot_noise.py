import matplotlib.pyplot as plt
import numpy as np

y = np.load("results/noise.npy")
x = np.array(list(range(10))) * 0.03

plt.plot(x, np.mean(y, axis=1), label="var(x0)+var(x1)+var(x2)")

for i in range(2):
    plt.scatter(x, y[:, i], c='k', s=10, zorder=100)

plt.xlabel("Noise")
plt.legend()
plt.show()
