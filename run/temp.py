from ldcl.data import physics
import matplotlib.pyplot as plt

d, _ = physics.get_dataset("data_configs/short_trajectories.json", "../saved_datasets")
x, y = d.bundle["x"], d.bundle["y"]
print("Remember to change traj sizes in short")
for i in range(10240):
    plt.scatter(x[i], y[i])
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.show()
    input("continue?")
    plt.clf()
