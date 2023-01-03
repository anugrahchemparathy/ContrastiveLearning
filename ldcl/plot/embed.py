import numpy as np
import torch

def embed(encoder_location_or_encoder, orbits_dataset):
    """
        Embed a dataset into its representations, while also neatly packaging conserved
        quantities and inputs.

        :param encoder_location_or_encoder: either a string (path to encoder)
            or already loaded PyTorch encoder
        :param orbits_dataset: a ConservationDataset object loaded from our datasets
        :return: a tuple of (1) a 2D array of encoder outputs (representations)
            and (2) a dictionary of H, L, phi0, x, y, v.x, v.y
    """

    orbits_loader = torch.utils.data.DataLoader(
        dataset = orbits_dataset,
        shuffle = True,
        batch_size = 1,
    )

    if isinstance(encoder_location_or_encoder, str):
        branch_encoder = torch.load(encoder_location_or_encoder, map_location=torch.device('cpu'))
    else:
        print("using encoder")
        branch_encoder = encoder_location_or_encoder
    branch_encoder.eval()

    data = orbits_dataset.data
    data = data.reshape([data.shape[0] * data.shape[1]] + list(data.shape)[2:])
    data = torch.from_numpy(data)
    encoder_outputs = branch_encoder(data.float()).detach().numpy()
    print(encoder_outputs.shape)

    values = {}
    print([x.shape for x in orbits_dataset.bundle.values()])
    for k, v in orbits_dataset.bundle.items():
        if k == "idxs_":
            continue
        if len(v.shape) == 3:
            values[k] = v[:, :, 0]
        elif len(v.shape) == 2:
            values[k] = v
        else:
            raise NotImplementedError

        if v.shape[1] != encoder_outputs.shape[0] / v.shape[0]: # conserved quantities
            values[k] = np.repeat(values[k], encoder_outputs.shape[0] / v.shape[0], axis=1)
        print(values[k].shape)

        values[k] = values[k].flatten()

    print(f"First output val: {encoder_outputs.flatten()[0]}")

    return encoder_outputs, values
