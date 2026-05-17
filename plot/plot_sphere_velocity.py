import numpy as np
import matplotlib.pyplot as plt

from core.geometry.sphere_mapping import (
    map_unit_triangle_to_sphere,
    unit_triangle_to_lonlat,
)
from core.geometry.sphere_velocity import solid_body_velocity_lonlat


def sample_unit_triangle(n=8):
    """
    Sample interior points in the unit triangle:
        xi >= 0, eta >= 0, xi + eta <= 1
    """
    xi_list = []
    eta_list = []

    for i in range(1, n):
        for j in range(1, n - i):
            xi = i / n
            eta = j / n

            if xi + eta < 1.0:
                xi_list.append(xi)
                eta_list.append(eta)

    return np.array(xi_list), np.array(eta_list)


def lonlat_velocity_to_cartesian(u, v, lam, theta):
    """
    Convert sphere tangent velocity components (u, v) into Cartesian components.

    Basis:
        e_lambda = (-sin(lam), cos(lam), 0)
        e_theta  = (-sin(theta) cos(lam),
                    -sin(theta) sin(lam),
                     cos(theta))

    Then:
        V = u * e_lambda + v * e_theta
    """
    e_lambda_x = -np.sin(lam)
    e_lambda_y =  np.cos(lam)
    e_lambda_z =  np.zeros_like(lam)

    e_theta_x = -np.sin(theta) * np.cos(lam)
    e_theta_y = -np.sin(theta) * np.sin(lam)
    e_theta_z =  np.cos(theta)

    Vx = u * e_lambda_x + v * e_theta_x
    Vy = u * e_lambda_y + v * e_theta_y
    Vz = u * e_lambda_z + v * e_theta_z

    return Vx, Vy, Vz


def draw_sphere_wireframe(ax, R=1.0):
    u = np.linspace(0.0, 2.0 * np.pi, 80)
    v = np.linspace(-0.5 * np.pi, 0.5 * np.pi, 40)
    uu, vv = np.meshgrid(u, v)

    Xs = R * np.cos(vv) * np.cos(uu)
    Ys = R * np.cos(vv) * np.sin(uu)
    Zs = R * np.sin(vv)

    ax.plot_wireframe(Xs, Ys, Zs, linewidth=0.25, alpha=0.12)


def plot_velocity_case(ax, alpha, title, R=1.0, u0=1.0):
    xi, eta = sample_unit_triangle(n=8)

    draw_sphere_wireframe(ax, R=R)

    X_all = []
    Y_all = []
    Z_all = []
    Vx_all = []
    Vy_all = []
    Vz_all = []

    for patch_id in range(8):
        X, Y, Z = map_unit_triangle_to_sphere(xi, eta, patch_id, R=R)
        lam, theta = unit_triangle_to_lonlat(xi, eta, patch_id)

        u, v = solid_body_velocity_lonlat(
            lam,
            theta,
            R=R,
            u0=u0,
            alpha=alpha,
        )

        Vx, Vy, Vz = lonlat_velocity_to_cartesian(u, v, lam, theta)

        X_all.append(X)
        Y_all.append(Y)
        Z_all.append(Z)
        Vx_all.append(Vx)
        Vy_all.append(Vy)
        Vz_all.append(Vz)

    X_all = np.concatenate(X_all)
    Y_all = np.concatenate(Y_all)
    Z_all = np.concatenate(Z_all)
    Vx_all = np.concatenate(Vx_all)
    Vy_all = np.concatenate(Vy_all)
    Vz_all = np.concatenate(Vz_all)

    # 畫 patch sample points
    ax.scatter(X_all, Y_all, Z_all, s=10, alpha=0.65)

    # 畫速度箭頭
    ax.quiver(
        X_all,
        Y_all,
        Z_all,
        Vx_all,
        Vy_all,
        Vz_all,
        length=0.18,
        normalize=True,
        linewidth=0.8,
    )

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_box_aspect([1, 1, 1])


def plot_sphere_velocity():
    R = 1.0
    u0 = 1.0

    fig = plt.figure(figsize=(14, 6))

    ax1 = fig.add_subplot(121, projection="3d")
    plot_velocity_case(
        ax1,
        alpha=0.0,
        title="Sphere velocity field (alpha = 0)",
        R=R,
        u0=u0,
    )

    ax2 = fig.add_subplot(122, projection="3d")
    plot_velocity_case(
        ax2,
        alpha=np.pi / 4.0,
        title="Sphere velocity field (alpha = pi/4)",
        R=R,
        u0=u0,
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_sphere_velocity()