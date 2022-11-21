import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
import time

rng = np.random.default_rng(10)  # manually seed random number generator
from scipy.special import ellipj

MAX_ITERATIONS = 1000


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
            #return None
    return E

def excluded_uniform_distribution(lower,higher,batch_size):
    """
    Generate a distribution from 0 - 1 excluding (lower, higher)
    """
    total = lower + (1-higher)
    lower_bsz = int(np.floor(lower/total * batch_size))
    higher_bsz = int(np.floor((1-higher)/total*batch_size))
    if lower_bsz + higher_bsz < batch_size:
        higher_bsz += 1

    lower_vals = rng.uniform(low=0.0, high=lower, size=(lower_bsz,1))
    higher_vals = rng.uniform(low=higher, high = 1.0, size=(higher_bsz,1))

    return_vals = np.vstack([lower_vals,higher_vals])

    #return_vals = return_vals[:, np.random.permutation(return_vals.shape[1])]
    np.random.shuffle(return_vals)

    return return_vals



def orbits_train_gen(batch_size, traj_samples=100, noise=0., shuffle=True, check=False, H_val=None, L_val=None, phi0_val=None,exclude_values=[], val_lower=None, val_higher = None):

    # t = np.cumsum(rng.exponential(scale=10., size=(batch_size, traj_samples)), axis=-1)

    mu = 1.  # standard gravitational parameter, i.e. G*M

    E = None
    while E is None:

        # randomly sampled observation times
        t = rng.uniform(0, 10. * traj_samples, size=(batch_size, traj_samples))

        if H_val is None: #energy
            if "H" in exclude_values:
                #print("H domain partially excluded!")
                H = -mu / 2 * (0.5 + 0.5 * excluded_uniform_distribution(val_lower,val_higher,batch_size))
            else:
                H = -mu / 2 * (0.5 + 0.5 * rng.uniform(size=(batch_size, 1)))
        else: 
            H_val * np.ones((batch_size, 1))
            

        if L_val is None: #angular momentum
            if "L" in exclude_values:
                L = excluded_uniform_distribution(val_lower,val_higher,batch_size)
            else:
                L = rng.uniform(size=(batch_size, 1))
        else:
            L = L_val * np.ones((batch_size, 1))
            #L standard value = 0.5

        a = -mu / (2 * H)  # semi-major axis
        e = np.sqrt(1 - L ** 2 / (mu * a)) #eccentricity?

        """
            y target indices correspond to:

            0 = e = eccentricity
            1 = a = semimajor axis


            most relevant
            2 = phi0 = orientation of the orbit
            3 = H = energy
            4 = L = angular momentum
        """

        if phi0_val is None:
            if "phi0" in exclude_values:
                phi0 = 2 * np.pi * excluded_uniform_distribution(val_lower,val_higher,batch_size)
            else:
                phi0 = 2 * np.pi * rng.uniform(size=(batch_size, 1))
        else:
            phi0 = phi0_val * np.ones((batch_size, 1))
            #phi0 standard value = np.pi
        
        # https://downloads.rene-schwarz.com/download/M001-Keplerian_Orbit_Elements_to_Cartesian_State_Vectors.pdf
        T = 2 * np.pi * np.sqrt(a ** 3 / mu)  # period
        M = np.fmod(2 * np.pi * t / T, 2 * np.pi)  # mean anomaly

        try:
            E = eccentric_anomaly_from_mean(e, M)  # eccentric anomaly
        except:
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
    def __init__(self, size=10240, check=False, gen_batch=128, transform=None,H_val=None,L_val=None,phi0_val=None,exclude_values=[], val_lower=None, val_higher = None, visual = False):
        """
        phiO=True,H=True,L=True to decide whether or not to fix phi0,H,L
        H = Energy
        L = Angular Momentum

        L standard value = 0.5
        phi0 standard value = np.pi
        """
        self.visual = visual
        self.transform = transform
        self.size = size
        start = time.time()
        num_batch = size // gen_batch
        self.params = [[], [], [], [], []]
        self.data = []
        for _ in range(num_batch):
            p, d = orbits_train_gen(gen_batch, check=check,H_val=H_val,L_val=L_val,phi0_val=phi0_val,exclude_values=exclude_values, val_lower=val_lower, val_higher = val_higher)
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
            img_size = 32

            if self.visual:
                data_1 = x_output[:3]
                data_2 = x_output[3:6]

                pxls_1 = 255*np.ones((3,img_size + 2, img_size + 2))

                for idx, (x1,x2,p1,p2) in enumerate(data_1):
                    #map (-2,-2) to (2,2) onto a 32 x 32 grid
                    newx1 = np.floor((x1 + 2)/4 * 32)
                    newx2 = np.floor((x2 + 2)/4 * 32)
                    for dx in (-1,0,1):
                        for dy in (-1,0,1):
                            pxls_1[idx,newx1+dx,newx2+dy] = 0


                    
            else:
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
    orbits_dataset = OrbitsDataset(exclude_values=['H','phi0'],val_lower=0.25,val_higher=0.75, size=128)
    params = orbits_dataset.params
    data = orbits_dataset.data
    print(params)
    
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

