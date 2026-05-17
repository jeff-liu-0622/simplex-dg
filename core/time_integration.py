import numpy as np

RKA = np.array([
    0.0,
    -567301805773.0 / 1357537059087.0,
    -2404267990393.0 / 2016746695238.0,
    -3550918686646.0 / 2091501179385.0,
    -1275806237668.0 / 842570457699.0
])

RKB = np.array([
    1432997174477.0 / 9575080441755.0,
    5161836677717.0 / 13612068292357.0,
    1720146321549.0 / 2090206949498.0,
    3134564353537.0 / 4481467310338.0,
    2277821191437.0 / 14882151754819.0
])

RKC = np.array([
    0.0,
    1432997174477.0 / 9575080441755.0,
    2526269341429.0 / 6820363962896.0,
    2006345519317.0 / 3224310063776.0,
    2802321613138.0 / 2924317926151.0
])


def lsrk54_step(q, res, t, dt, compute_rhs_func, **kwargs):
    """
    Low-storage Runge-Kutta 5-stage 4th-order single step.

    compute_rhs_func must have signature:

        rhs = compute_rhs_func(q, t, **kwargs)

    and rhs must have the same shape as q.
    """
    q = np.asarray(q)
    res = np.asarray(res)

    if res.shape != q.shape:
        raise ValueError(f"res shape {res.shape} must match q shape {q.shape}.")

    for stage in range(5):
        t_local = t + RKC[stage] * dt

        rhs_val = compute_rhs_func(q, t_local, **kwargs)

        if rhs_val.shape != q.shape:
            raise ValueError(
                f"RHS shape {rhs_val.shape} does not match q shape {q.shape}."
            )

        res = RKA[stage] * res + dt * rhs_val
        q = q + RKB[stage] * res

    return q, res