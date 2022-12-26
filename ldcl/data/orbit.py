import numpy as np
from .dists import sample_distribution

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

        all_conserved = sample_distribution(settings.traj_distr, settings.num_trajs)
        H, L, phi0 = tuple([all_conserved[..., i][..., np.newaxis] for i in range(3)])

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

