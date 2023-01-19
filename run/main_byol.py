import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

import os
import shutil
import argparse

from ldcl.models import branch, predictor, byol_model

from ldcl.tools.seed import set_deterministic
from ldcl.optimizers.lr_scheduler import LR_Scheduler, get_lr_scheduler
from ldcl.data import physics
#from ldcl.losses.nce import infoNCE, rmseNCE, normalmseNCE
#from ldcl.losses.simclr import NT_Xent_loss, infoNCE
from ldcl.losses.byol_loss import byol_loss
#from ldcl.losses.simsiam import simsiam
from ldcl.plot.plot import plot_loss
from ldcl.tools.device import get_device, t2np

import tqdm

device = get_device()

import pathlib

SCRIPT_PATH = pathlib.Path(
  __file__).parent.resolve().as_posix() + "/"  # always get this directory

saved_epochs = list(
  range(20)) + [20, 40, 60, 80, 100, 200, 300, 400, 500, 1000, 1400]


def training_loop(args):
  global saved_epochs
  """
    drop_last: drop the last non_full batch (potentially useful for training weighting etc.)
    pin_memory: speed dataloader transfer to cuda
    num_workers: multiprocess data loading
    """
  save_progress_path = os.path.join(SCRIPT_PATH, "saved_models", args.fname)
  os.mkdir(save_progress_path)

  data_config_file = "data_configs/" + args.data_config

  train_orbits_dataset, folder = physics.get_dataset(data_config_file,
                                                     "../saved_datasets")
  print(f"Using dataset {folder}...")
  shutil.copy(data_config_file,
              os.path.join(save_progress_path, "data_config.json"))
  train_orbits_loader = torch.utils.data.DataLoader(
    dataset=train_orbits_dataset,
    shuffle=True,
    batch_size=args.bsz,
  )

  if args.all_epochs:
    saved_epochs = list(range(args.epochs))

  # online network
  # target network (outputs have stop grad)
  #online = branch.branchEncoder(encoder_out=3, useBatchNorm=True)

  #encoder = branch.branchEncoder(encoder_out=3, useBatchNorm=True, activation= nn.Sigmoid())
  #encoder = branch.branchEncoder(encoder_out=3, num_layers=15, useBatchNorm=True, encoder_hidden=128)
  #encoder = branch.branchEncoder(encoder_out=3)
  encoder = branch.branchEncoder(encoder_out=3, useBatchNorm=True)
  proj_head = branch.projectionHead(head_in=3, head_out=3)

  online_model = branch.sslModel(encoder=encoder, projector=proj_head)
  momentum_model = byol_model.moving_average_network(online_model)
  predictor_network = predictor.predictor()

  for model in [online_model, momentum_model, predictor_network]:
    model.to(device)
  
  online_model.save(save_progress_path, 'start')

  sgd_params = online_model.params(args.lr)
  sgd_params[0]['params'] += list(predictor_network.parameters())
  optimizer = torch.optim.SGD(sgd_params, lr=args.lr, momentum=0.9, weight_decay=args.wd)
  lr_scheduler = get_lr_scheduler(args, optimizer, train_orbits_loader)

  def apply_loss(z1, z2, loss_func):
    loss = loss_func(z1, z2)
    return loss

  losses = []

  with tqdm.trange(args.epochs) as t:
    for e in range(args.epochs):
      online_model.train()

      for it, (input1, input2, y) in enumerate(train_orbits_loader):
        online_model.zero_grad()

        # forward pass
        input1 = input1.type(torch.float32).to(device)
        input2 = input2.type(torch.float32).to(device)

        
        online1 = predictor_network(online_model(input1))
        online2 = predictor_network(online_model(input2))
        with torch.no_grad():
          momentum1 = momentum_model(input1).detach()
          momentum2 = momentum_model(input2).detach()

        loss = apply_loss(online1, momentum2, byol_loss) + apply_loss(momentum1, online2, byol_loss)

        # optimization step
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        momentum_model.update_weights()

      losses.append(t2np(loss).flatten()[0])

      if e in saved_epochs:
        online_model.save(save_progress_path, f'{e:02d}')
      t.set_postfix(loss=loss.item(),
                    loss50_avg=np.mean(np.array(losses[max(-1 * e, -50):])))
      t.update()

  online_model.save(save_progress_path, 'final')
  losses = np.array(losses)
  np.save(os.path.join(save_progress_path, "loss.npy"), losses)
  plot_loss(losses, title=args.fname, save_progress_path=save_progress_path)

  return encoder


if __name__ == '__main__':
  pass
  parser = argparse.ArgumentParser()
  parser.add_argument('--epochs', default=1500, type=int)
  parser.add_argument('--lr', default=0.02, type=float)
  parser.add_argument('--bsz', default=512, type=int)
  parser.add_argument('--wd', default=0.001, type=float)
  parser.add_argument('--warmup_epochs', default=5, type=int)
  parser.add_argument('--fine_tune', default=False, type=bool)
  # parser.add_argument('--fname', default='simclr_infoNCE_1hidden_head_4dim' , type = str)
  parser.add_argument('--fname', default='byol_test1', type=str)
  parser.add_argument('--data_config',
                      default='orbit_config_default.json',
                      type=str)
  parser.add_argument('--all_epochs', default=False, type=bool)

  args = parser.parse_args()
  training_loop(args)