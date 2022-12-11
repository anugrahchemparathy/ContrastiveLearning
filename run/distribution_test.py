from ldcl.data import physics
import torch

import matplotlib.pyplot as plt

train_orbits_dataset, folder = physics.get_dataset("orbit_config_test.json", "../saved_datasets")
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
    plt.show()
