import numpy as np

from test.test_manifold_geometry_sphere import build_octahedral_sphere_diagnostics


def solid_body_rotation_velocity(xyz_nodes, u0=1.0, alpha=np.pi / 4.0):
    """
    Compute the 3D solid-body rotation velocity V = omega x X.

    The angular velocity convention matches the manifold report direction:

        omega = (-u0 sin(alpha), 0, u0 cos(alpha)).
    """
    omega = np.array(
        [
            -u0 * np.sin(alpha),
            0.0,
            u0 * np.cos(alpha),
        ],
        dtype=float,
    )

    return np.cross(omega[None, :], xyz_nodes)


def compute_velocity_diagnostics(nsub=8, order=4, R=1.0, u0=1.0, alpha=np.pi / 4.0):
    """
    Geometry-only velocity diagnostic on the octahedral sphere manifold mesh.

    This does not build a sphere RHS and does not perform time integration.
    """
    _, diagnostics = build_octahedral_sphere_diagnostics(
        nsub=nsub,
        order=order,
        R=R,
    )

    max_tangency_error = 0.0
    max_reconstruction_error = 0.0
    max_speed = 0.0

    for k, (xyz_nodes, geometry) in enumerate(diagnostics):
        V3D = solid_body_rotation_velocity(
            xyz_nodes,
            u0=u0,
            alpha=alpha,
        )

        tangency_error = np.abs(np.sum(V3D * xyz_nodes, axis=1))

        a1 = geometry["a1"]
        a2 = geometry["a2"]
        ac1 = geometry["a_contra_1"]
        ac2 = geometry["a_contra_2"]

        u_tilde = np.sum(ac1 * V3D, axis=1)
        v_tilde = np.sum(ac2 * V3D, axis=1)

        assert np.all(np.isfinite(V3D)), f"element {k}: non-finite V3D"
        assert np.all(np.isfinite(u_tilde)), f"element {k}: non-finite u_tilde"
        assert np.all(np.isfinite(v_tilde)), f"element {k}: non-finite v_tilde"

        V_rec = u_tilde[:, None] * a1 + v_tilde[:, None] * a2
        reconstruction_error = np.linalg.norm(V_rec - V3D, axis=1)
        speed = np.linalg.norm(V3D, axis=1)

        max_tangency_error = max(max_tangency_error, np.max(tangency_error))
        max_reconstruction_error = max(
            max_reconstruction_error,
            np.max(reconstruction_error),
        )
        max_speed = max(max_speed, np.max(speed))

    return {
        "nsub": nsub,
        "order": order,
        "R": R,
        "u0": u0,
        "alpha": alpha,
        "max_tangency_error": max_tangency_error,
        "max_reconstruction_error": max_reconstruction_error,
        "max_speed": max_speed,
    }


def test_solid_body_velocity_is_tangent_and_reconstructable():
    result = compute_velocity_diagnostics(
        nsub=8,
        order=4,
        R=1.0,
        u0=1.0,
        alpha=np.pi / 4.0,
    )

    print("\n" + "=" * 88)
    print("Manifold sphere velocity diagnostic")
    print("=" * 88)
    print(f"nsub = {result['nsub']}")
    print(f"order = {result['order']}")
    print(f"R = {result['R']:.6e}")
    print(f"u0 = {result['u0']:.6e}")
    print(f"alpha = {result['alpha']:.12e}")
    print(f"max tangency error = {result['max_tangency_error']:.6e}")
    print(f"max reconstruction error = {result['max_reconstruction_error']:.6e}")
    print(f"max speed = {result['max_speed']:.6e}")
    print("=" * 88)

    assert result["max_tangency_error"] < 1.0e-12, (
        "solid-body velocity is not tangent to the sphere: "
        f"max error = {result['max_tangency_error']:.3e}"
    )
    assert result["max_reconstruction_error"] < 1.0e-2, (
        "contravariant velocity reconstruction error is too large: "
        f"max error = {result['max_reconstruction_error']:.3e}"
    )
    assert np.isfinite(result["max_speed"])
    assert result["max_speed"] > 0.0


def run_all_tests():
    print("\n" + "=" * 88)
    print("Manifold sphere velocity diagnostic")
    print("=" * 88)

    test_solid_body_velocity_is_tangent_and_reconstructable()
    print("solid-body velocity is tangent and reconstructable")

    print("=" * 88)
    print("test_manifold_velocity_sphere.py passed")


if __name__ == "__main__":
    run_all_tests()
