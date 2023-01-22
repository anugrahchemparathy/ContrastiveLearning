import torch
import numpy as np
from ldcl.losses.nce import infoNCE, rmseNCE
import tqdm

model = torch.nn.Sequential(
    torch.nn.Linear(1, 64),
    torch.nn.ReLU(inplace=True),
    torch.nn.Linear(64, 64),
    torch.nn.ReLU(inplace=True),
    torch.nn.Linear(64, 1),
)
model.train()

pts = torch.arange(0,1,step=0.001,requires_grad=True)

def process(tens):
    tens = torch.floor(tens * 1000).to(torch.int32)
    tens = torch.maximum(tens, torch.tensor([0]))
    tens = torch.minimum(tens, torch.tensor([999]))
    return pts[tens]

rng = np.random.default_rng(12345)

data_size = 100000

data = rng.uniform(0, 1, size=(data_size,))

noise_f = np.full((data_size,), 1)
noise_f = data * data * 0.5
#noise = rng.normal(size=(data_size,num_copies)) * noise_f[:, np.newaxis]
#noise_f = np.where(data < 0.5, np.full(data.shape,0.01), np.full(data.shape,0.1))

#data = data + noise
data = torch.tensor(data)
data_loader = torch.utils.data.DataLoader(
    dataset = np.stack((data,noise_f),axis=1),
    shuffle = True,
    batch_size = 5000
)

#optimizer = torch.optim.SGD(model.parameters(), lr=0.0003)
current_lr = 0.01
optimizer = torch.optim.Adam([pts], lr=current_lr)

loss_history = []
pts_history = []
with tqdm.trange(90) as t:
    for e in range(90):
        losses = []
        for batch in data_loader:
            bdata = batch[:, 0]
            bnoise = batch[:, 1]

            bnoise = torch.tensor(rng.normal(size=(bdata.shape[0],2))) * bnoise[:, np.newaxis]

            z1 = torch.unsqueeze(process(bdata + bnoise[:, 0]), dim=1)
            z2 = torch.unsqueeze(process(bdata + bnoise[:, 1]), dim=1)

            loss = rmseNCE(z1, z2)
            losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        loss_history.append(torch.mean(torch.tensor(losses)).item())
        pts_history.append(pts)
        t.set_postfix({
            "loss": torch.mean(torch.tensor(losses)).item(),
            "lr": current_lr,
            "pos": [f"{x:.2f}" for x in pts[::199].detach().cpu().numpy().tolist()]
        })
        t.update()

        """
        if e > 5 and loss_history[-1] > loss_history[-2] and loss_history[-2] > loss_history[-3]:
            current_lr *= 0.5
            optimizer = torch.optim.Adam([pts], lr=current_lr)
            loss_history = loss_history[:-2]
            pts_history = pts_history[:-2]
            pts = pts_history[loss_history.index(min(loss_history))]
        """

model.eval()
new_data = torch.unsqueeze(torch.arange(0, 1,step=0.001), dim=1)
out = process(new_data)

print(torch.quantile(out,0.99) - torch.quantile(out,0.01))
np.save("noise_out", out.detach().cpu().numpy())
