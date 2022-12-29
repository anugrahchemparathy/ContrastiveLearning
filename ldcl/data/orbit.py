import numpy as np
import copy
from PIL import Image

from .dists import sample_distribution
from .config import read_config

MAX_ITERATIONS = 100
rng = np.random.default_rng(9)

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
    settings = config.orbit_settings

    mu = settings.mu  # standard gravitational parameter, i.e. G*M

    E = None
    while E is None:
        t = sample_distribution(settings.t_distr, settings.num_trajs * settings.num_ts).reshape((settings.num_trajs, settings.num_ts))
        if config.modality == "image":
            #t = np.stack((t, t + config.pendulum_imagen_settings.diff_time), axis=-1) # time steps
            factor = config.orbits_imagen_settings.diff_time / config.orbits_imagen_settings.num_pts
            t = np.stack(tuple([t + i * factor for i in range(0, config.orbits_imagen_settings.num_pts)]), axis=-1)

        all_conserved = sample_distribution(settings.traj_distr, settings.num_trajs)
        H, L, phi0 = tuple([all_conserved[..., i][..., np.newaxis] for i in range(3)])

        if config.modality == "image":
            H, L, phi0 = H[..., np.newaxis], L[..., np.newaxis], phi0[..., np.newaxis]

        a = -mu / (2 * H)  # semi-major axis
        e = np.sqrt(1 - L ** 2 / (mu * a))

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

    if config.modality == "image":
        R = np.stack((c, -s, s, c), axis=-1).reshape(-1, 1, 1, 2, 2)
    else:
        R = np.stack((c, -s, s, c), axis=-1).reshape(-1, 1, 2, 2)
    vel = np.squeeze(R @ np.expand_dims(vel, axis=-1), axis=-1)  # rotated by phi0

    data = np.concatenate((pos, vel), axis=-1)

    if settings.check:
        assert np.allclose(M, E - e * np.sin(E))

        p = np.sqrt(mu * (2 / r - 1 / a))  # speed/specific momentum
        diffp = p - np.linalg.norm(vel, axis=-1)
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

def orbits_img_gen(config, bundle):
    """
        Orbits image generation.

        :param config: all the config details
        :param bundle: dictionary output from pendulum_num_gen
        :return: a dictionary containing conserved qs, some other values, and a dataset of images.
    """
    settings = config.orbits_imagen_settings

    pxls = np.ones((config.orbit_settings.num_trajs, config.orbit_settings.num_ts, settings.img_size + 2, settings.img_size + 2, 3)) # easier to plot with some boundaries

    pos = bundle["data"][:, :, :, 0:2]
    pos = (settings.img_size / 2) / settings.scale * pos
    pos = pos + settings.img_size / 2 + 1

    frac = np.ceil(pos + 0.5) - 0.5 - pos # amount bottom/right
    pos = np.ceil(pos + 0.5) - 2 # bottom-right corner
    pos = pos.astype('int')
    pos = np.where(pos < 0, -1, pos)
    pos = np.where(pos >= settings.img_size + 1, -1, pos)

    start = np.array(settings.start_color)
    end = np.array(settings.end_color)
    cols = np.repeat(start[np.newaxis, :], settings.num_pts, axis=0)
    if settings.num_pts > 1:
        cols = (np.arange(settings.num_pts) / (settings.num_pts - 1))[:, np.newaxis] * (end - start)[np.newaxis, :]
    else:
        cols = np.array([1])[:, np.newaxis] * (end - start)[np.newaxis, :]
        print("only one point??")
    
    num_na = 0
    for traj in range(config.orbit_settings.num_trajs):
        for t in range(config.orbit_settings.num_ts):
            for p in range(settings.num_pts):
                if np.any(np.equal(pos[traj, t, p], -1)):
                    num_na += 1
                    continue

                ratio = np.full((2, 2), 1.0) # calculate ratio to add new color 
                ratio[:, 0] = ratio[:, 0] * frac[traj, t, p, 1]
                ratio[:, 1] = ratio[:, 1] * (1 - frac[traj, t, p, 1])
                ratio[0, :] = ratio[0, :] * frac[traj, t, p, 0]
                ratio[1, :] = ratio[1, :] * (1 - frac[traj, t, p, 0])

                ratio = ratio[:, :, np.newaxis]
                pxls[traj, t, pos[traj, t, p, 0]:pos[traj, t, p, 0] + 2, pos[traj, t, p, 1]:pos[traj, t, p, 1] + 2, :] = ratio * cols[np.newaxis, np.newaxis, p] + (1 - ratio) * pxls[traj, t, pos[traj, t, p, 0]:pos[traj, t, p, 0] + 2, pos[traj, t, p, 1]:pos[traj, t, p, 1] + 2, :] # change pxl values

    pxls = pxls[:, :, 1:-1, 1:-1, :] # remove borders

    new_bundle = copy.deepcopy(bundle)
    new_bundle["data"] = pxls # replace with the correct data

    if settings.noise != 0:
        raise NotImplementedError
    if settings.continuous != True:
        raise NotImplementedError
    if settings.not_visible != "do_nothing":
        raise NotImplementedError

    if settings.not_visible == "do_nothing":
        print(f"Invisible handling: {settings.not_visible}. {num_na} out of {config.orbit_settings.num_trajs * config.orbit_settings.num_ts * settings.num_pts} points are invisible.")
    
    return new_bundle

    """

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
    """

"""Test image generation."""

"""
config = read_config("image_gen_test.json")

#bundle = orbits_img_gen(config, {
#    "data": np.array([[-1.0,1.0,0.0,0.0], [0.0,0.0,0.0,0.0]])[np.newaxis, np.newaxis, :, :]
#})
bundle = orbits_img_gen(config, orbits_num_gen(config))

img = bundle["data"][0, 0]
img = (img * 255).astype(np.uint8)
img = Image.fromarray(img, 'RGB')
img.show()

# Visualize images
#imgs = np.swapaxes(imgs, 4, 2)
imgs = bundle["data"]
imgs = imgs * 255
imgs = imgs.astype(np.uint8)
for j in range(0, config.orbit_settings.num_trajs):
    img = np.sum(imgs[j, :, :, :, :], axis=0)
    print(img)
    img = np.maximum(img - config.orbit_settings.num_ts * 255 + 255, 0)
    print(img)
    img = img.astype(np.uint8)
    img = Image.fromarray(img, 'RGB')
    img.show()
    input("Continue...")
for j in range(0, config.orbit_settings.num_trajs):
    for i in range(0, config.orbit_settings.num_ts):
        img = Image.fromarray(imgs[i,j,:,:,:], 'RGB')
        img.show()
        input("continue...")
"""