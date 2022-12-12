import numpy as np

from munch import DefaultMunch
import json
from PIL import Image

from scipy.special import ellipj

import time
import glob
import os
from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import torch

#rng = np.random.default_rng()
rng = np.random.default_rng(9)  # manually seed random number generator
verbose = True
MAX_ITERATIONS = 100

def read_config(f):
    """
        Read config files. Implement this in a function in case we need to change this at some point.

        We should probably move this somewhere else at some point, as the training loop will need it as well.
        
        :param: f: path to config file to be read
        :return: x: an object with attributes that are the defined parameters
    """

    with open(f, "r") as stream:
        x = json.load(stream)

    def convert_keys(d):
        for key, value in d.items():
            if value in ["None", "True", "False"]:
                d[key] = eval(d[key])
            elif isinstance(value, str) and "," in value:
                d[key] = eval("[" + d[key] + "]")
            elif isinstance(value, dict):
                convert_keys(value)
            elif isinstance(value, list) and all([isinstance(x, dict) for x in value]):
                for i, x in enumerate(value):
                    convert_keys(x)

    convert_keys(x)

    return DefaultMunch.fromDict(x, object())

def interval_generator(dist):
    intervals = []

    if dist.mode == "even_space": # evenly spaced gaps
        for k in range(dist.dims):
            intervals.append([[(2 * i) / (2 * dist.intervals[k] - 1), (2 * i + 1) / (2 * dist.intervals[k] - 1)] for i in range(0, dist.intervals[k])])
            intervals[k] = np.array(intervals[k])
            intervals[k] = intervals[k] * (dist.max[k] - dist.min[k]) + dist.min[k]
    elif dist.mode == "explicit": # explicitly described gaps
        for k in range(dist.dims):
            intervals.append(np.array(dist.intervals[k]))
    else:
        raise ValueError("dist interval specification unrecognized")

    return intervals

def in_intervals(dist, data):
    intervals = interval_generator(dist)

    data = np.transpose(data)

    is_interval = []
    for k in range(dist.dims):
        is_interval.append(np.logical_and(data[k][:, np.newaxis] > intervals[k][np.newaxis, :, 0], data[k][:, np.newaxis] < intervals[k][np.newaxis, :, 1]))
        is_interval[k] = np.any(is_interval[k], axis=1)
    is_interval = np.array(is_interval)

    if dist.dims == 1 or dist.combine == "all":
        is_interval = np.all(is_interval, axis=0)
    elif dist.combine == "any":
        is_interval = np.any(is_interval, axis=0)
    else:
        raise ValueError("dist.combine unrecognized")
    return is_interval

def sample_distribution(dist, num):
    if dist.type == "uniform":
        ret = []

        if dist.dims == 1:
            dist.max = [dist.max]
            dist.min = [dist.min]

        for dim in range(dist.dims):
            ret.append(rng.uniform(dist.min[dim], dist.max[dim], size=num))
        
        if dist.dims == 1:
            dist.max = dist.max[0] # Undo modifications
            dist.min = dist.min[0]
            return ret[0]
        else:
            return np.stack(ret, axis=-1)
    elif dist.type == "uniform_with_intervals":
        # Note: this is uniform across the union of all intervals,
        # i.e. if some intervals are longer than other intervals,
        # they will be more likely.

        intervals = interval_generator(dist)

        final = np.zeros((0, dist.dims))
        while np.shape(final)[0] < num:
            ret = []
            for k in range(dist.dims):
                ret.append(rng.uniform(np.min(intervals[k]), np.max(intervals[k]), size=num - np.shape(final)[0]))

            is_interval = in_intervals(dist, np.transpose(np.array(ret)))

            ret = np.swapaxes(np.array(ret), 0, 1)
            ret = ret[is_interval]

            final = np.concatenate((final, ret))
        final = final[:num]

        return final
    elif dist.type == "exponential":
        if dist.dims == 1:
            return dist.shift + rng.exponential(dist.scale, size=num)
        else:
            raise NotImplementedError
    elif dist.type == "stack":
        ret = []
        for smalld in dist.dists:
            ret.append(sample_distribution(smalld, num))
            if len(ret[-1].shape) == 1:
                ret[-1] = ret[-1][:, np.newaxis]

        ret = np.concatenate(ret, axis=1)

        if dist.reorder_axes != None:
            if len(dist.reorder_axes) != ret.shape[1]:
                raise ValueError("reorder axes not of correct length")
            ret = ret[:, dist.reorder_axes]

        return ret
    elif dist.type == "single":
        return np.array([dist.value] * num)
    else:
        raise NotImplementedError # implement other kinds of distributions

def is_in_distribution(config_or_dist, arr):
    reduce_to_one = False
    if isinstance(arr, int) or isinstance(arr, float):
        reduce_to_one = True
        arr = [arr]

    if len(arr.shape) == 1:
        arr = arr[:, np.newaxis]

    if not isinstance(config_or_dist, str) and "dims" in vars(config_or_dist):
        dist = config_or_dist
    else:
        if isinstance(config_or_dist, str):
            config = read_config(config_or_dist)

        for key, value in vars(config).items():
            if isinstance(value, dict) and "traj_distr" in value:
                dist = DefaultMunch.fromDict(value["traj_distr"], object())

    if dist.type == "uniform":
        if isinstance(dist.min, float) or isinstance(dist.min, int):
            dist.min = [dist.min]
            dist.max = [dist.max]

        dist.min = np.array(dist.min)
        dist.max = np.array(dist.max)
        ret = np.logical_and(arr > dist.min[np.newaxis, :], arr < dist.max[np.newaxis, :])
        ret = np.all(ret, axis=1)
    elif dist.type == "uniform_with_intervals":
        ret = in_intervals(dist, arr)
    elif dist.type == "exponential":
        if isinstance(dist.shift, float) or isinstance(dist.shift, int):
            dist.shift = [dist.shift]
            dist.scale = [dist.scale]

        dist.shift = np.array(dist.shift)
        dist.scale = np.array(dist.scale)
        ret = np.all(np.logical_and(arr > dist.shift[np.newaxis, :], arr < (dist.shift[np.newaxis, :] + dist.scale[np.newaxis, :])), axis=1)
    elif dist.type == "stack":
        if dist.reorder_axes != None:
            inverted = [dist.reorder_axes.index(i) for i in range(len(dist.reorder_axes))]
            arr = arr[:, inverted]

        startid = 0
        in_dists = []
        for smalld in dist.dists:
            in_dists.append(is_in_distribution(smalld, arr[:, startid:startid + smalld.dims]))
            startid += smalld.dims

        ret = np.all(np.array(in_dists), axis=0)
        #ret = np.array(in_dists)
    elif dist.type == "single":
        ret = np.equal(arr, np.array(dist.value)[np.newaxis, :])
        ret = np.all(ret, axis=1)
    else:
        raise NotImplementedError

    if reduce_to_one:
        ret = ret[0]

    return ret

def pendulum_num_gen(config):
    """
        pendulum numerical (time, energy, position, momentum, etc.) generation
    
        :param config: configuration details
        :return: energy, data=(angle, angular momentum), predata=(time, energy)
    """
    settings = config.pendulum_settings

    t = np.reshape(sample_distribution(settings.t_distr, settings.num_ts * settings.num_trajs), (settings.num_trajs, settings.num_ts))
    if config.modality == "image":
        t = np.stack((t, t + config.pendulum_imagen_settings.diff_time), axis=-1) # time steps
    k2 = sample_distribution(settings.traj_distr, settings.num_trajs)[:, np.newaxis, np.newaxis]

    sn, cn, dn, _ = ellipj(t, k2) # fix this more
    q = 2 * np.arcsin(np.sqrt(k2) * sn)
    p = 2 * np.sqrt(k2) * cn * dn / np.sqrt(1 - k2 * sn ** 2) # anglular momentum
    data = np.stack((q, p), axis=-1)

    if settings.pq_resample_condition is not None:
        raise NotImplementedError # Resampling is going to be hard!4

    if settings.shuffle:
        for x in data:
            rng.shuffle(x, axis=0)

    """
    I don't know what this code is for.
        if check_energy:
            H = 0.5 * p ** 2 - np.cos(q) + 1
            diffH = H - 2 * k2
            print("max diffH = ", np.max(np.abs(diffH)))
            assert np.allclose(diffH, np.zeros_like(diffH))
    """

    if settings.noise > 0:
        data += settings.noise * rng.standard_normal(size=data.shape)

    return {
        "k2": k2,
        "data": data
    }

def pendulum_img_gen(config, bundle):
    """
        pendulum image generation: vectorized !

        :param config: all the config details
        :param bundle: dictionary output from pendulum_num_gen
        :return: a dictionary containing energies (k2), image data (imgs), and positions (q)
    """
    settings = config.pendulum_imagen_settings

    data_size = bundle["data"].shape[0]
    traj_samples = bundle["data"].shape[1]
    q = bundle["data"][..., 0]

    if settings.crop != 1.0: # Cropping: create "bigger" image, then crop after
        if settings.crop_c == [-1, -1]:
            settings.crop_c = [1 - settings.crop / 2, 1 - settings.crop / 2]
        big_img = np.floor(settings.img_size / settings.crop + 4).astype('int32')
        left = np.floor(settings.crop_c[0] * big_img - settings.img_size / 2)
        top = np.floor(settings.crop_c[1] * big_img - settings.img_size / 2)
    else:
        big_img = settings.img_size

    center_x = big_img // 2
    center_y = big_img // 2
    str_len = big_img - 4 - big_img // 2 - settings.bob_size
    bob_area = (2 * settings.bob_size + 1)**2

    pxls = np.ones((data_size, traj_samples, settings.img_size + 2, settings.img_size + 2, 3))
    if config.verbose:
        print("[Dataset] Blank images created")

    x = center_x + np.round(np.cos(q) * str_len)
    y = center_y + np.round(np.sin(q) * str_len)
    
    idx = np.indices((data_size, traj_samples))
    idx = np.expand_dims(idx, [0, 1, 5])

    bob_idx = np.indices((2 * settings.bob_size + 1, 2 * settings.bob_size + 1)) - settings.bob_size
    bob_idx = np.swapaxes(bob_idx, 0, 2)
    bob_idx = np.expand_dims(bob_idx, [3, 4, 5])

    pos = np.expand_dims(np.stack((x, y), axis=0), [0, 1])
    pos = pos + bob_idx
    pos = np.reshape(pos, (bob_area, 2, data_size, traj_samples, 2))
    pos = np.expand_dims(pos, 0)

    c = np.expand_dims(np.array([[1, 1], [0, 2]]), [1, 2, 3, 4])

    idx, pos, c = np.broadcast_arrays(idx, pos, c)
    c = np.expand_dims(c[:, :, 0, :, :, :], 2)
    idx_final = np.concatenate((idx, pos, c), axis=2)

    # Translation noise. Disabling for now.
    """
    if settings.noise > 0:
        translation_noise = rng.uniform(0, 1, size=(2, data_size, traj_samples))
        translation_noise = np.minimum(np.ones(translation_noise.shape),
                                np.floor(np.abs(translation_noise) / (1 - 2 * settings.noise))) * translation_noise / np.abs(translation_noise)
        translation_noise = np.concatenate((np.zeros(translation_noise.shape), translation_noise, np.zeros(translation_noise.shape)), axis=0)
        translation_noise = translation_noise[:5]
        translation_noise = np.expand_dims(translation_noise, [0, 1, 5])
        idx_final = idx_final + translation_noise
    """

    idx_final = np.swapaxes(idx_final, 0, 2)
    idx_final = np.reshape(idx_final, (5, 4 * data_size * traj_samples * bob_area))
    idx_final = idx_final.astype('int32')

    if config.verbose:
        print("[Dataset] Color indices computed")

    if settings.crop == 1.0:
        pxls[idx_final[0], idx_final[1], idx_final[2] + 1, idx_final[3] + 1, idx_final[4]] = 0
    else:
        idx_final[2] = idx_final[2] - left.astype('int32') + 1
        idx_final[3] = idx_final[3] - top.astype('int32') + 1
        idx_final[2] = np.maximum(idx_final[2], np.array(0))
        idx_final[3] = np.maximum(idx_final[3], np.array(0))
        idx_final[2] = np.minimum(idx_final[2], np.array(settings.img_size + 1))
        idx_final[3] = np.minimum(idx_final[3], np.array(settings.img_size + 1))

        pxls[idx_final[0], idx_final[1], idx_final[2], idx_final[3], idx_final[4]] = 0
    pxls = pxls[:, :, 1:settings.img_size + 1, 1:settings.img_size + 1, :]

    if settings.noise > 0:
        pxls = pxls + settings.noise / 4 * rng.standard_normal(size=pxls.shape)
        tint_noise = rng.uniform(- settings.noise / 8, settings.noise / 8, size=(data_size, traj_samples, 1, 1, 3))
        pxls = pxls + tint_noise
        pxls = np.minimum(np.ones(pxls.shape), np.maximum(np.zeros(pxls.shape), pxls))

    if config.verbose:
        print("[Dataset] Images computed")

    pxls = np.swapaxes(pxls, 4, 2)
    
    return {
        "k2": np.broadcast_arrays(bundle["k2"], q)[0],
        "q": q,
        "data": pxls
    }

def eccentric_anomaly_from_mean(e, M, tol=1e-13):
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

def orbits_num_gen(config):
    #batch_size, traj_samples=100, noise=0., shuffle=True, check=False, H=None, L=None, phi0=None):
    settings = config.orbit_settings

    mu = settings.mu  # standard gravitational parameter, i.e. G*M

    E = None
    while E is None:
        #t = rng.uniform(0, 10. * traj_samples, size=(batch_size, traj_samples))
        t = sample_distribution(settings.t_distr, settings.num_trajs * settings.num_ts).reshape((settings.num_trajs, settings.num_ts))

        all_conserved = sample_distribution(settings.traj_distr, settings.num_trajs)
        H, L, phi0 = tuple([all_conserved[..., i][..., np.newaxis] for i in range(3)])

        #H = -mu / 2 * (0.5 + 0.5 * rng.uniform(size=(batch_size, 1))) if H is None else H * np.ones((batch_size, 1)) # samples uniformly from -1/4 to -1/2
        #L = rng.uniform(size=(batch_size, 1)) if L is None else L * np.ones((batch_size, 1))

        a = -mu / (2 * H)  # semi-major axis
        e = np.sqrt(1 - L ** 2 / (mu * a))

        #print('eshape =', e.shape)
        #print('tshape =', t.shape)

        #phi0 = 2 * np.pi * rng.uniform(size=(batch_size, 1)) if phi0 is None else phi0 * np.ones((batch_size, 1))

        # https://downloads.rene-schwarz.com/download/M001-Keplerian_Orbit_Elements_to_Cartesian_State_Vector/.pdf
        T = 2 * np.pi * np.sqrt(a ** 3 / mu)  # period
        M = np.fmod(2 * np.pi * t / T, 2 * np.pi)  # mean anomaly
        
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
    R = np.stack((c, -s, s, c), axis=-1).reshape(-1, 1, 2, 2)
    vel = np.squeeze(R @ np.expand_dims(vel, axis=-1), axis=-1)  # rotated by phi0

    data = np.concatenate((pos, vel), axis=-1)
    #print('data shape =', data.shape)

    if settings.check:
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

    if settings.shuffle:
        for x in data:
            rng.shuffle(x, axis=0)

    if settings.noise > 0:
        data += settings.noise * rng.standard_normal(size=data.shape)

    return {
        "e": e,
        "a": a,
        "phi0": phi0,
        "H": H,
        "L": L,
        "data": data
    }

"""
# Visualize images
imgs = np.swapaxes(imgs, 4, 2)
imgs = imgs * 255
imgs = imgs.astype(np.uint8)
for j in range(0, config.pendulum_settings.num_trajs):
    for i in range(0, config.pendulum_settings.num_ts):
        img = Image.fromarray(imgs[i,j,:,:,:], 'RGB')
        img.show()
        print(ret_imgs[0].shape, ret_imgs[0][i, j])
        input("continue...")
"""

class ConservationDataset(torch.utils.data.Dataset):
    def __init__(self, bundle):
        self.size = bundle["data"].shape[0]
        self.data = bundle["data"] # assuming this is where the actual data is
        del bundle["data"]
        self.bundle = bundle

    def __getitem__(self, idx):
        if idx < self.size:
            x_data = self.data[idx]
            random_x_rows = np.random.randint(0,x_data.shape[0],2) # generate two different views

            x_output = x_data[random_x_rows]
            x_view1 = x_output[0]
            x_view2 = x_output[1]

            return [x_view1,x_view2, {k: v[idx] for k, v in self.bundle.items()}]
        else:
            raise ValueError

    def __len__(self):
        return self.size
        

def get_dataset(config, saved_dir):
    """
        General dataset generation.

        Eventually, we'll probably read the config file in the training loop and then just pass it in here as an object...

        :param config: Configuration file to receive parameters in, or already parsed configuration object.
        :return: Dataset object, and then name of folder inside saved_dir where data can be found
    """
    if isinstance(config, str):
        config = read_config(config)
    #print(saved_dir)

    # If cache, check if exists
    bundle = None
    if config.use_cached_data:
        for other_file in glob.glob(saved_dir + "/*/config.json"):
            if json.dumps(read_config(other_file)) == json.dumps(config):
                p = Path(other_file)
                other_file = p.parents[0]
                folder_name = str(os.path.join(other_file, ''))
                
                bundle = {}
                for npfile in glob.glob(os.path.join(other_file, "*.npy")):
                    name = Path(npfile).with_suffix('').name
                    bundle[name] = np.load(npfile)

    # Generate data
    if bundle is None:
        if config.dynamics == "pendulum":
            bundle = pendulum_num_gen(config)
            if config.modality == "image":
                bundle = pendulum_img_gen(config, bundle)
            elif config.modality == "numerical":
                bundle = bundle
            else:
                raise ValueError("Config modality not specified")
        elif config.dynamics == "orbits":
            bundle = orbits_num_gen(config)
            if config.modality == "image":
                raise NotImplementedError
            elif config.modality == "numerical":
                bundle = bundle
            else:
                raise ValueError("Config modality not specified")
        else:
            raise ValueError("Config dynamics not specified")

        # Save dataset
        folder_name = saved_dir + "/" + time.strftime("%Y%m%d") + "-"
        idx = 0
        while os.path.exists(folder_name + str(idx) + "/"):
            idx += 1
        folder_name = folder_name + str(idx)
        os.mkdir(folder_name)
        for key in bundle.keys():
            np.save(folder_name + "/" + key, bundle[key])

        with open(folder_name + "/config.json", "w") as f:
            json.dump(config, f)
    
    dataset = ConservationDataset(bundle)

    return dataset, folder_name

#get_dataset("orbit_config_default.json")

# Template to test distribution generation.
"""class TestDistribution:
    def __init__(self):
        self.type = "uniform_with_intervals"
        self.mode = "explicit"
        self.dims = 2
        self.intervals = [[[-1,0],[0.5,1],[2,3]], [[0,1],[2,3]]]
        self.combine = "any"

dist = TestDistribution()

sample = sample_distribution(dist, 100000)
plt.scatter(sample[:, 0], sample[:, 1], s=0.1)
plt.show()"""
