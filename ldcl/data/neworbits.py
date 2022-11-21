import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
import time

rng = np.random.default_rng(8)  # manually seed random number generator
from scipy.special import ellipj

MAX_ITERATIONS = 10000


def eccentric_anomaly_from_mean(e, M, tol=1e-14):
    """Convert mean anomaly to eccentric anomaly.
    Implemented from [A Practical Method for Solving the Kepler Equation][1]
    by Marc A. Murison from the U.S. Naval Observatory
    [1]: http://murison.alpheratz.net/dynamics/twobody/KeplerIterations_summary.pdf
    """
    Mnorm = np.fmod(M, 2 * np.pi)
    E0 = np.fmod(M + (-1 / 2 * e ** 3 + e + (e ** 2 + 3 / 2 * np.cos(M) * e ** 3) * np.cos(M)) * np.sin(M), 2 * np.pi)
    dE = tol + 1
    count = 0
    while np.any(dE > tol):
        t1 = np.cos(E0)
        t2 = -1 + e * t1
        t3 = np.sin(E0)
        t4 = e * t3
        t5 = -E0 + t4 + Mnorm
        t6 = t5 / (1 / 2 * t5 * t4 / t2 + t2)
        E = E0 - t5 / ((1 / 2 * t3 - 1 / 6 * t1 * t6) * e * t6 + t2)
        dE = np.abs(E - E0)
        E0 = np.fmod(E, 2 * np.pi)
        count += 1
        if count == MAX_ITERATIONS:
            print('Current dE: ', dE[dE > tol])
            print('eccentricity: ', np.repeat(e, dE.shape[-1], axis=-1)[dE > tol])
            raise RuntimeError(f'Did not converge after {MAX_ITERATIONS} iterations with tolerance {tol}.')
    return E


def orbits_train_gen(batch_size, traj_samples=100, noise=0., shuffle=True, check=False, H=None, L=None, phi0=None):
    #print('batch size =', batch_size, ' traj_samples =', traj_samples)
    # randomly sampled observation times
    t = rng.uniform(0, 10. * traj_samples, size=(batch_size, traj_samples))
    # t = np.cumsum(rng.exponential(scale=10., size=(batch_size, traj_samples)), axis=-1)

    mu = 1.  # standard gravitational parameter, i.e. G*M

    H = -mu / 2 * (0.5 + 0.5 * rng.uniform(size=(batch_size, 1))) if H is None else H * np.ones((batch_size, 1))
    L = rng.uniform(size=(batch_size, 1)) if L is None else L * np.ones((batch_size, 1))

    a = -mu / (2 * H)  # semi-major axis
    e = np.sqrt(1 - L ** 2 / (mu * a))

    #print('eshape =', e.shape)
    #print('tshape =', t.shape)

    phi0 = 2 * np.pi * rng.uniform(size=(batch_size, 1)) if phi0 is None else phi0 * np.ones((batch_size, 1))

    # https://downloads.rene-schwarz.com/download/M001-Keplerian_Orbit_Elements_to_Cartesian_State_Vectors.pdf
    T = 2 * np.pi * np.sqrt(a ** 3 / mu)  # period
    M = np.fmod(2 * np.pi * t / T, 2 * np.pi)  # mean anomaly
    E = None
    while E is None:
        try:
            E = eccentric_anomaly_from_mean(e, M)  # eccentric anomaly
        except:
            print("data generation failed.")
            pass
    phi = 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2), np.sqrt(1 - e) * np.cos(E / 2))  # true anomaly/angle
    r = (a * (1 - e ** 2)) / (1 + e * np.cos(phi))  # radius
    pos = np.stack((r * np.cos(phi + phi0), r * np.sin(phi + phi0)), axis=-1)  # position rotated by phi0

    vel = np.expand_dims(np.sqrt(mu * a) / r, axis=-1) * np.stack((-np.sin(E), np.sqrt(1 - e ** 2) * np.cos(E)),
                                                                  axis=-1)  # velocity
    c, s = np.cos(phi0), np.sin(phi0)
    R = np.stack((c, -s, s, c), axis=-1).reshape(batch_size, 1, 2, 2)
    vel = np.squeeze(R @ np.expand_dims(vel, axis=-1), axis=-1)  # rotated by phi0

    data = np.concatenate((pos, vel), axis=-1)
    #print('data shape =', data.shape)

    if check:
        assert np.allclose(M, E - e * np.sin(E))

        p = np.sqrt(mu * (2 / r - 1 / a))  # speed/specific momentum
        diffp = p - np.linalg.norm(vel, axis=-1)
        # print(e[np.any(np.isnan(diffp),axis=-1)])
        assert np.allclose(diffp, np.zeros_like(diffp))

        L = np.sqrt(mu * a * (1 - e ** 2))  # specific angular momentum
        diffL = L - np.cross(pos, vel)
        assert np.allclose(diffL, np.zeros_like(diffL))

        H = -mu / (2 * a)  # specific energy
        diffH = H - (0.5 * np.linalg.norm(vel, axis=-1) ** 2 - mu / np.linalg.norm(pos, axis=-1))
        assert np.allclose(diffH, np.zeros_like(diffH))

    if shuffle:
        for x in data:
            rng.shuffle(x, axis=0)

    if noise > 0:
        data += noise * rng.standard_normal(size=data.shape)
    return [e, a, phi0, H, L], data


class OrbitsDataset(torch.utils.data.Dataset):
    def __init__(self, size=10240, check=False, gen_batch=128, transform=None):
        self.transform = transform
        self.size = size
        start = time.time()
        num_batch = size // gen_batch
        self.params = [[], [], [], [], []]
        self.data = []
        for _ in range(num_batch):
            p, d = orbits_train_gen(gen_batch, check=check)
            for i in range(len(self.params)):
                self.params[i].append(p[i])
            self.data.append(d)
        self.data = np.concatenate(self.data, axis=0)
        for i in range(len(self.params)):
            self.params[i] = np.concatenate(self.params[i], axis=0)
        self.params = np.concatenate(self.params, axis=1)  # e, a, phi0, H, L
        print(f'It took {time.time() - start} time to finish the job.')

    def __getitem__(self, idx):
        if idx < self.size:
            x_data = self.data[idx]
            random_x_rows = np.random.randint(0,x_data.shape[0],2)

            x_output = x_data[random_x_rows]
            x_view1 = x_output[0]
            x_view2 = x_output[1]

            # [input = [x1, x2, p1, p2]?, target = [e, a, phi0, H, L]
            """
            e = eccentricity
            a = semimajor axis
            phi0 = orientation of the orbit
            H = energy
            L = angular momentum
            """
            return [x_view1,x_view2,self.params[idx]]

    def __len__(self):
        return self.size


if __name__ == '__main__':
    orbits_dataset = OrbitsDataset()
    params = orbits_dataset.params
    data = orbits_dataset.data
    
    train_orbits_loader = torch.utils.data.DataLoader(
        dataset = orbits_dataset,
        shuffle = True,
        batch_size = 5,
    )

    print(len(train_orbits_loader))
    for inp1,inp2,y in train_orbits_loader:
        print(len(inp1))
        print(type(inp1[0]))
        break

