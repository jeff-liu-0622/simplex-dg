import numpy as np


def compute_advection_dt(h, N, speed, cfl=0.05):
    """
    dt = CFL * h / (N^2 * speed)
    """
    if speed <= 0.0:
        return np.inf

    return cfl * h / (N**2 * speed)


def compute_advection_dt_from_velocity(h, N, u, v, cfl=0.05):
    """
    For variable velocity fields.
    """
    speed_max = np.max(np.sqrt(u**2 + v**2))

    if speed_max <= 0.0:
        return np.inf

    return cfl * h / (N**2 * speed_max)