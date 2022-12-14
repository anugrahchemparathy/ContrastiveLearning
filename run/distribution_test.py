from ldcl.data import physics
import torch

import matplotlib.pyplot as plt

train_orbits_dataset, folder = physics.get_dataset("data_configs/eight_mini_cubes_015.json", "../saved_datasets")
print(f"Using dataset {folder}...")

loader = torch.utils.data.DataLoader(
    dataset = train_orbits_dataset,
    shuffle = True,
    batch_size = len(train_orbits_dataset),
)

for data in loader:
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(data[2]["phi0"], data[2]["L"], data[2]["H"], s=0.3)
    for a in [0, 6.28]:
        for b in [0, 1]:
            ax.plot([a, a], [b, b], [-0.5, -0.25], color='k', zorder=100)
    for a in [0, 6.28]:
        for c in [-0.5, -0.25]:
            ax.plot([a, a], [0, 1], [c, c], color='k', zorder=100)
    for b in [0, 1]:
        for c in [-0.5, -0.25]:
            ax.plot([0.0, 6.28], [b, b], [c, c], color='k', zorder=100)

    ax.set_xlabel("phi0")
    ax.set_ylabel("L")
    ax.set_zlabel("H")
    plt.show()
