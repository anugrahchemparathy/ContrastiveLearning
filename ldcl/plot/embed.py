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

    encoder_outputs_list = []
    target_values = []
    inputs = []

    for it, (input1, input2, y) in enumerate(orbits_loader):
        predicted_representation = branch_encoder(input1.float()).detach().numpy()[0]
        inputs.append(input1.float().numpy()[0])
        encoder_outputs_list.append(predicted_representation)

        #append conserved quantities to the end of the representation for plotting, y = (1, )
        #[2=phi0,3=energy,4=angular_momentum] discard [0=eccentricity, 1=semimajor_axis]
        target_values.append(np.array( (y["phi0"].item(),y["H"].item(),y["L"].item()) ))

    encoder_outputs = np.vstack(encoder_outputs_list)
    target_values = np.vstack(target_values)

    phi0_c_values = target_values[:,0]
    energy_c_values = target_values[:,1]
    angular_momentum_c_values = target_values[:,2]

    inputs = np.array(inputs)
    return encoder_outputs, {
        "phi0": phi0_c_values,
        "H": energy_c_values,
        "L": angular_momentum_c_values,
        "x": inputs[:, 0],
        "y": inputs[:, 1],
        "v.x": inputs[:, 2],
        "v.y": inputs[:, 3]
    }
