import numpy as np

from munch import DefaultMunch
import json

from scipy.special import ellipj

MAX_ITERATIONS = 1000
rng = np.random.default_rng()

def read_config(f):
    """
        Read config files. Implement this in a function in case we need to change this at some point.
        
        :param: f: path to config file to be read
        :return: x: an object with attributes that are the defined parameters
    """

    with open(f, "r") as stream:
        x = json.load(stream)
    return DefaultMunch.fromDict(x, object())

def sample_distribution(dist, num):
    if dist.type == "uniform":
        if dist.dims == 1:
            return rng.uniform(dist.min, dist.max, size=num)
        else:
            raise NotImplementedError # implement multi-d
    else:
        raise NotImplementedError # implement other kinds of distributions

def eccentric_anomaly_from_mean(e, M, tol=1e-14):
    """
    Convert mean anomaly to eccentric anomaly.

    @anugrah can you please help me here i have no idea what is going on  : ( 

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

def pendulum_num_gen(config):
    """
        pendulum numerical (time, energy, position, momentum, etc.) generation
    
        :param config: configuration details
        :return: energy, data=(angle, angular momentum), predata=(time, energy)
    """
    settings = config.pendulum_settings

    t = sample_distribution(settings.t_distr, settings.num_ts)
    k2 = sample_distribution(settings.energy_distr, settings.num_energies)

    sn, cn, dn, _ = ellipj(t, k2) # fix this more
    q = 2 * np.arcsin(np.sqrt(k2) * sn) # angle
    p = 2 * np.sqrt(k2) * cn * dn / np.sqrt(1 - k2 * sn ** 2) # anglular momentum
    data = np.stack((q, p), axis=-1)

    if settings.shuffle:
        for x in data:
            rng.shuffle(x, axis=0)

    if check_energy:
        H = 0.5 * p ** 2 - np.cos(q) + 1
        diffH = H - 2 * k2
        print("max diffH = ", np.max(np.abs(diffH)))
        assert np.allclose(diffH, np.zeros_like(diffH))

    if noise > 0:
        data += noise * rng.standard_normal(size=data.shape)

    return k2, data

def pendulum_img_gen(config, tk2):
    """
        pendulum image generation

        :param config: all the config details
        :param tk2: generated t (time) and k2 (energy) pairs from pendulum numerical generation
        :return: energy, images
    """

    raise NotImplementedError
    

def pendulum_train_gen(data_size, traj_samples=10, gnoise=0., nnoise=0., uniform=False,
        shuffle=True, check_energy=False, k2=None, image=True,
        blur=False, img_size=32, diff_time=0.5, bob_size=1, continuous=False,
        gaps=[-1,-1], crop=1.0, crop_c=[-1,-1], t_window=[-1,-1], t_range=-1, mink=0, maxk=1):
    """
        pendulum dataset generation

        :param data_size: number of pendulums
        :param traj_samples: number of samples per pendulum
        :param noise: Gaussian noise
        :param shuffle: if shuffle data
        :param check_energy: check the energy (numerical mode only)
        :param k2: specify energy
        :param image: use image/graphical mode
        :param blur: whether to use motion blur mode (otherwise, use two-frame mode) [not implemented]
        :param img_size: size of (square) image (graphical mode only)
        :param diff_time: time difference between two images (graphical mode only)
        :param bob_size: bob = square of length bob_size * 2 + 1
        :param continuous: whether to use continuous generation [not implemented]
        :param gaps: generate gaps in data (graphical mode only)
                        (-1 if not used; otherwise [# gaps, total gap len as proportion])
        :param crop: proportion of image cutout returned
        :param crop_c: center of the image cutout
        :param time_window: specify a certain window of time
        :param time_range: specify max window of time per energy
                (-1 if not used)
        :param mink: minimum energy
        :param maxk: maximum energy
        :return: energy, data
    """
    raise NotImplementedError

    gparams = read_config(config_file)

    # setting up random seeds

    if not image:
        t = rng.uniform(0, 10. * traj_samples, size=(data_size, traj_samples))
        k2 = rng.uniform(size=(data_size, 1)) if k2 is None else k2 * np.ones((data_size, 1))  # energies (conserved)

        sn, cn, dn, _ = ellipj(t, k2)
        q = 2 * np.arcsin(np.sqrt(k2) * sn) # angle
        p = 2 * np.sqrt(k2) * cn * dn / np.sqrt(1 - k2 * sn ** 2) # anglular momentum
        data = np.stack((q, p), axis=-1)

        if shuffle:
            for x in data:
                rng.shuffle(x, axis=0)

        if check_energy:
            H = 0.5 * p ** 2 - np.cos(q) + 1
            diffH = H - 2 * k2
            print("max diffH = ", np.max(np.abs(diffH)))
            assert np.allclose(diffH, np.zeros_like(diffH))

        if noise > 0:
            data += noise * rng.standard_normal(size=data.shape)

        return k2, data

    elif image and not blur:
        if t_window != [-1, -1] and t_range != -1:
            raise UserWarning("Cannot use both time windows & ranges at the same time")
        elif t_window != [-1, -1]:
            t = rng.uniform(t_window[0], t_window[1], size=(data_size, traj_samples))
            t = np.stack((t, t + diff_time), axis=-1) # time steps
        elif t_range != -1:
            t_base = rng.uniform(0, 10. * traj_samples, size=(data_size, 1))
            t_rng = rng.uniform(0, t_range, size=(data_size, traj_samples))
            t = t_base + t_rng
            t = np.stack((t, t + diff_time), axis=-1) # time steps
        else:
            t = rng.uniform(0, 10. * traj_samples, size=(data_size, traj_samples))
            t = np.stack((t, t + diff_time), axis=-1) # time steps

        if gaps == [-1,-1]:
            if not uniform:
                k2 = rng.uniform(mink, maxk, size=(data_size, 1, 1)) if k2 is None else k2 * np.ones((data_size, 1, 1))  # energies (conserved)
            else:
                if k2 == None:
                    k2 = np.linspace(mink, maxk, num=data_size)
                    k2 = np.reshape(k2, (data_size, 1, 1))
                else:
                    k2 = k2 * np.ones((data_size, 1, 1))
        else:
            if k2 == None:
                if not uniform:
                    k2 = rng.uniform(0, 1 - gaps[1], size=(data_size, 1, 1))
                    prefix = np.floor(k2 / (1 - gaps[1]) * (gaps[0] + 1)) * ((1 - gaps[1]) / (gaps[0] + 1) + gaps[1] / gaps[0])
                    frac = k2 - np.floor(k2 / (1 - gaps[1]) * (gaps[0] + 1)) * (1 - gaps[1]) / (gaps[0] + 1)
                    k2 = prefix + frac
                    k2 = k2 * (maxk - mink) + mink
                else:
                    k2 = np.linspace(0, 1 - gaps[1], num=data_size, endpoint=False)
                    k2 = np.reshape(k2, (data_size, 1, 1))
                    prefix = np.floor(k2 / (1 - gaps[1]) * (gaps[0] + 1)) * ((1 - gaps[1]) / (gaps[0] + 1) + gaps[1] / gaps[0])
                    frac = k2 - np.floor(k2 / (1 - gaps[1]) * (gaps[0] + 1)) * (1 - gaps[1]) / (gaps[0] + 1)
                    k2 = prefix + frac
                    k2 = k2 * (maxk - mink) + mink
            else:
                k2 = k2 * np.ones((data_size, 1, 1))

        sn, cn, dn, _ = ellipj(t, k2)
        q = 2 * np.arcsin(np.sqrt(k2) * sn)

        if verbose:
            print("[Dataset] Numerical generation complete")

        if shuffle:
            for x in q:
                rng.shuffle(x, axis=0) # TODO: check if the shapes work out

        if nnoise > 0:
            q += nnoise * rng.standard_normal(size=q.shape)

        # Image generation begins here
        if crop != 1.0:
            if crop_c == [-1, -1]:
                crop_c = [1 - crop / 2, 1 - crop / 2]
            big_img = np.floor(img_size / crop + 4).astype('int32')
            left = np.floor(crop_c[0] * big_img - img_size / 2)
            top = np.floor(crop_c[1] * big_img - img_size / 2)
        else:
            big_img = img_size

        center_x = big_img // 2
        center_y = big_img // 2
        str_len = big_img - 4 - big_img // 2 - bob_size
        bob_area = (2 * bob_size + 1)**2

        pxls = np.ones((data_size, traj_samples, img_size + 2, img_size + 2, 3))
        if verbose:
            print("[Dataset] Blank images created")

        x = center_x + np.round(np.cos(q) * str_len)
        y = center_y + np.round(np.sin(q) * str_len)

        idx = np.indices((data_size, traj_samples))
        idx = np.expand_dims(idx, [0, 1, 5])

        bob_idx = np.indices((2 * bob_size + 1, 2 * bob_size + 1)) - bob_size
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

        if gnoise > 0:
            translation_noise = rng.uniform(0, 1, size=(2, data_size, traj_samples))
            translation_noise = np.minimum(np.ones(translation_noise.shape),
                                    np.floor(np.abs(translation_noise) / (1 - 2 * gnoise))) * translation_noise / np.abs(translation_noise)
            translation_noise = np.concatenate((np.zeros(translation_noise.shape), translation_noise, np.zeros(translation_noise.shape)), axis=0)
            translation_noise = translation_noise[:5]
            translation_noise = np.expand_dims(translation_noise, [0, 1, 5])
            idx_final = idx_final + translation_noise

        idx_final = np.swapaxes(idx_final, 0, 2)
        idx_final = np.reshape(idx_final, (5, 4 * data_size * traj_samples * bob_area))
        idx_final = idx_final.astype('int32')

        if verbose:
            print("[Dataset] Color indices computed")
        if crop == 1.0:
            pxls[idx_final[0], idx_final[1], idx_final[2] + 1, idx_final[3] + 1, idx_final[4]] = 0
        else:
            #bigpxls = np.ones((data_size, traj_samples, big_img + 2, big_img + 2, 3))
            #bigpxls[idx_final[0], idx_final[1], idx_final[2] + 1, idx_final[3] + 1, idx_final[4]] = 0
            idx_final[2] = idx_final[2] - left.astype('int32') + 1
            idx_final[3] = idx_final[3] - top.astype('int32') + 1
            idx_final[2] = np.maximum(idx_final[2], np.array(0))
            idx_final[3] = np.maximum(idx_final[3], np.array(0))
            idx_final[2] = np.minimum(idx_final[2], np.array(img_size + 1))
            idx_final[3] = np.minimum(idx_final[3], np.array(img_size + 1))

            pxls[idx_final[0], idx_final[1], idx_final[2], idx_final[3], idx_final[4]] = 0
        pxls = pxls[:, :, 1:img_size + 1, 1:img_size + 1, :]

        if gnoise > 0:
            pxls = pxls + gnoise / 4 * rng.standard_normal(size=pxls.shape)
            tint_noise = rng.uniform(- gnoise / 8, gnoise / 8, size=(data_size, traj_samples, 1, 1, 3))
            pxls = pxls + tint_noise
            pxls = np.minimum(np.ones(pxls.shape), np.maximum(np.zeros(pxls.shape), pxls))

        if verbose:
            print("[Dataset] Images computed")

        """pxls = pxls * 255
        pxls = pxls.astype(np.uint8)
        #bigpxls = bigpxls * 255
        #bigpxls = bigpxls.astype(np.uint8)
        for j in range(0, traj_samples):
            for i in range(0, data_size):
                img = Image.fromarray(pxls[i,j,:,:,:], 'RGB')
                img.show()
                #fig, axs = plt.subplots(2)
                #axs[0].imshow(pxls[i,j,:,:,:])
                #axs[1].imshow(bigpxls[i,j,:,:,:])
                #plt.show()
                input("continue...")
                #plt.clf()"""

        pxls = np.swapaxes(pxls, 4, 2)
        return np.broadcast_arrays(k2, q)[0], pxls, q
