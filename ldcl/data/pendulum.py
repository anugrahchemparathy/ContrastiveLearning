import numpy as np
from PIL import Image

from scipy.special import ellipj

from .dists import sample_distribution

rng = np.random.default_rng(9)  # manually seed random number generator
verbose = True

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

