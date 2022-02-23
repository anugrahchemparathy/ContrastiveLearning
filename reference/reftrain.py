import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import tqdm

from utils import *
from flow_models import SimpleAffine, StackSimpleAffine, RealNVP

plt.style.use('ggplot')



"""
========================================================================================
========================================================================================
"""


# Very simple training loop
def train(model, data, epochs = 100, batch_size = 64):
    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size)
    optimizer = torch.optim.Adam(model.parameters())
    
    losses = []
    with tqdm.tqdm(range(epochs), unit=' Epoch') as tepoch:
        epoch_loss = 0
        for epoch in tepoch:
            for batch_index, training_sample in enumerate(train_loader):
                log_prob = model.log_probability(training_sample)
                loss = - log_prob.mean(0)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss
            epoch_loss /= len(train_loader)
            losses.append(np.copy(epoch_loss.detach().numpy()))
            tepoch.set_postfix(loss=epoch_loss.detach().numpy())

    return model, losses


"""
========================================================================================
========================================================================================
"""

torch.manual_seed(2)
np.random.seed(0)

num_layers= 4
masks = torch.nn.functional.one_hot(torch.tensor([i % 2 for i in range(num_layers)])).float()
hidden_size = 32

data = FlowDataset('Moons')
NVP_model = RealNVP(masks, hidden_size)
moon_model, loss = train(NVP_model, data, epochs = 1000)
plot_density(moon_model, mesh_size=2.2)