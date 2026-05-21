import numpy as np

from core.geometry.connectivity import build_connectivity
from core.geometry.sphere_mapping import map_unit_triangle_to_sphere
from core.mesh_octahedron import create_octahedral_layout_mesh
from core.operators import build_local_operators
from core.geometry.manifold_metrics import compute_manifold_geometry
from core.operators_sphere import compute_manifold_skew_volume_rhs
from test.test_manifold_geometry_sphere import (
    build_octahedral_sphere_diagnostics,
    build_projected_octahedral_sphere_diagnostics,
    map_reference_nodes_to_subelement,
)
from test.test_manifold_velocity_sphere import solid_body_rotation_velocity


NEAR_ZERO_RHS_TOL = 1.0e-8
PATCH_BOUNDARY_TOL = 1.0e-12
NEAR_VERTEX_FACTOR = 1.5
REFERENCE_FACE_NORMALS = np.array(
    [
        [0.0, -1.0],
        [1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)],
        [-1.0, 0.0],
    ],
    dtype=float,
)
OCTAHEDRON_VERTICES = np.array(
    [
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
    ],
    dtype=float,
)


def build_octahedral_sphere_diagnostics_with_metadata(nsub=8, order=4, R=1.0):
    engine = build_local_operators(N=order, n=order, rule="table1")
    _, _, EToV, patch_ids, element_local_coords = create_octahedral_layout_mesh(nsub)

    diagnostics = []

    for k in range(EToV.shape[0]):
        patch_id = int(patch_ids[k])
        patch0 = patch_id - 1
        tri_vertices = element_local_coords[k]

        xi, eta = map_reference_nodes_to_subelement(
            engine.r,
            engine.s,
            tri_vertices,
        )

        X, Y, Z = map_unit_triangle_to_sphere(xi, eta, patch0, R=R)
        xyz_nodes = np.column_stack([X, Y, Z])
        geometry = compute_manifold_geometry(engine, xyz_nodes)

        diagnostics.append(
            {
                "element": k,
                "patch_id": patch_id,
                "xi": xi,
                "eta": eta,
                "xyz_nodes": xyz_nodes,
                "geometry": geometry,
            }
        )

    return engine, diagnostics


def _distance_to_original_octahedron_vertex(xyz, R=1.0):
    vertices = R * OCTAHEDRON_VERTICES
    distances = np.linalg.norm(vertices - xyz[None, :], axis=1)
    return np.min(distances)


def _patch_boundary_distance(xi, eta):
    return min(float(xi), float(eta), float(1.0 - xi - eta))


def _summarize_values(values):
    values = np.asarray(values, dtype=float)

    if values.size == 0:
        return {
            "count": 0,
            "max_abs_divJv": np.nan,
            "rms_divJv": np.nan,
            "mean_divJv": np.nan,
        }

    return {
        "count": int(values.size),
        "max_abs_divJv": float(np.max(np.abs(values))),
        "rms_divJv": float(np.sqrt(np.mean(values**2))),
        "mean_divJv": float(np.mean(values)),
    }


def collect_divergence_samples(
    nsub,
    order=4,
    R=1.0,
    u0=1.0,
    alpha=np.pi / 4.0,
):
    engine, diagnostics = build_octahedral_sphere_diagnostics_with_metadata(
        nsub=nsub,
        order=order,
        R=R,
    )

    near_vertex_tol = NEAR_VERTEX_FACTOR / nsub
    samples = []

    for entry in diagnostics:
        xyz_nodes = entry["xyz_nodes"]
        geometry = entry["geometry"]
        V3D = solid_body_rotation_velocity(
            xyz_nodes,
            u0=u0,
            alpha=alpha,
        )
        q = np.ones(engine.num_nodes)

        _, divJv_over_J, u_tilde, v_tilde = compute_manifold_skew_volume_rhs(
            engine=engine,
            geometry=geometry,
            V3D=V3D,
            q=q,
        )

        assert np.all(np.isfinite(divJv_over_J))
        assert np.all(np.isfinite(u_tilde))
        assert np.all(np.isfinite(v_tilde))

        for node in range(engine.num_nodes):
            xi = float(entry["xi"][node])
            eta = float(entry["eta"][node])
            xyz = xyz_nodes[node]
            boundary_distance = _patch_boundary_distance(xi, eta)
            vertex_distance = _distance_to_original_octahedron_vertex(xyz, R=R)

            samples.append(
                {
                    "element": int(entry["element"]),
                    "node": node,
                    "patch_id": int(entry["patch_id"]),
                    "xi": xi,
                    "eta": eta,
                    "X": xyz.copy(),
                    "J": float(geometry["J"][node]),
                    "u_tilde": float(u_tilde[node]),
                    "v_tilde": float(v_tilde[node]),
                    "divJv": float(divJv_over_J[node]),
                    "is_face_node": node < engine.num_boundary_nodes,
                    "is_interior_node": node >= engine.num_boundary_nodes,
                    "patch_boundary_distance": boundary_distance,
                    "near_patch_boundary": boundary_distance < PATCH_BOUNDARY_TOL,
                    "octahedron_vertex_distance": vertex_distance,
                    "near_octahedron_vertex": vertex_distance <= near_vertex_tol,
                }
            )

    return samples


def compute_constant_state_rhs_diagnostics(
    nsub=8,
    order=4,
    R=1.0,
    u0=1.0,
    alpha=np.pi / 4.0,
    mesh_kind="layout",
):
    if mesh_kind == "layout":
        engine, diagnostics = build_octahedral_sphere_diagnostics(
            nsub=nsub,
            order=order,
            R=R,
        )
    elif mesh_kind == "projected":
        engine, diagnostics = build_projected_octahedral_sphere_diagnostics(
            nsub=nsub,
            order=order,
            R=R,
        )
    else:
        raise ValueError("mesh_kind must be 'layout' or 'projected'.")

    rhs_values = []
    divJv_values = []

    for k, (xyz_nodes, geometry) in enumerate(diagnostics):
        V3D = solid_body_rotation_velocity(
            xyz_nodes,
            u0=u0,
            alpha=alpha,
        )
        q = np.ones(engine.num_nodes)

        rhs_vol, divJv_over_J, u_tilde, v_tilde = compute_manifold_skew_volume_rhs(
            engine=engine,
            geometry=geometry,
            V3D=V3D,
            q=q,
        )

        assert np.all(np.isfinite(V3D)), f"element {k}: non-finite V3D"
        assert np.all(np.isfinite(u_tilde)), f"element {k}: non-finite u_tilde"
        assert np.all(np.isfinite(v_tilde)), f"element {k}: non-finite v_tilde"
        assert np.all(np.isfinite(rhs_vol)), f"element {k}: non-finite RHS"
        assert np.all(np.isfinite(divJv_over_J)), f"element {k}: non-finite divJv/J"

        rhs_values.append(rhs_vol)
        divJv_values.append(divJv_over_J)

    rhs = np.concatenate(rhs_values)
    divJv = np.concatenate(divJv_values)

    max_abs_rhs = np.max(np.abs(rhs))
    rms_rhs = np.sqrt(np.mean(rhs**2))
    mean_rhs = np.mean(rhs)
    max_abs_divJv = np.max(np.abs(divJv))
    rms_divJv = np.sqrt(np.mean(divJv**2))
    mean_divJv = np.mean(divJv)

    near_zero = max_abs_rhs < NEAR_ZERO_RHS_TOL and rms_rhs < NEAR_ZERO_RHS_TOL

    return {
        "nsub": nsub,
        "order": order,
        "R": R,
        "u0": u0,
        "alpha": alpha,
        "mesh_kind": mesh_kind,
        "max_abs_rhs": max_abs_rhs,
        "rms_rhs": rms_rhs,
        "mean_rhs": mean_rhs,
        "max_abs_divJv": max_abs_divJv,
        "rms_divJv": rms_divJv,
        "mean_divJv": mean_divJv,
        "near_zero": near_zero,
    }


def test_projected_mesh_constant_state_metric_divergence_diagnostic():
    result = compute_constant_state_rhs_diagnostics(
        nsub=16,
        order=4,
        R=1.0,
        u0=1.0,
        alpha=np.pi / 4.0,
        mesh_kind="projected",
    )

    print("\n" + "=" * 100)
    print("Projected octahedron mesh q=1 metric divergence diagnostic")
    print("=" * 100)
    print(f"mesh_kind = {result['mesh_kind']}")
    print(f"nsub = {result['nsub']}")
    print(f"order = {result['order']}")
    print(f"max_abs_divJv = {result['max_abs_divJv']:.6e}")
    print(f"rms_divJv = {result['rms_divJv']:.6e}")
    print(f"mean_divJv = {result['mean_divJv']:.6e}")
    print(f"max_abs_rhs = {result['max_abs_rhs']:.6e}")
    print(f"rms_rhs = {result['rms_rhs']:.6e}")
    print("=" * 100)

    assert np.isfinite(result["max_abs_divJv"])
    assert np.isfinite(result["rms_divJv"])
    assert np.isfinite(result["mean_divJv"])
    assert result["max_abs_divJv"] < 1.0e-4, (
        "projected mesh metric divergence is larger than expected: "
        f"{result['max_abs_divJv']:.3e}"
    )


def compute_divergence_refinement_diagnostic(
    nsub,
    order=4,
    R=1.0,
    u0=1.0,
    alpha=np.pi / 4.0,
):
    samples = collect_divergence_samples(
        nsub=nsub,
        order=order,
        R=R,
        u0=u0,
        alpha=alpha,
    )
    divJv = np.array([sample["divJv"] for sample in samples])
    max_location = max(samples, key=lambda sample: abs(sample["divJv"]))

    return {
        "nsub": nsub,
        "order": order,
        "R": R,
        "max_abs_divJv": float(np.max(np.abs(divJv))),
        "rms_divJv": float(np.sqrt(np.mean(divJv**2))),
        "mean_divJv": float(np.mean(divJv)),
        "max_location": max_location,
    }


def compute_boundary_vs_interior_diagnostic(nsub, order=4, R=1.0):
    samples = collect_divergence_samples(nsub=nsub, order=order, R=R)

    categories = {
        "interior": [s["divJv"] for s in samples if s["is_interior_node"]],
        "face_boundary": [s["divJv"] for s in samples if s["is_face_node"]],
        "near_patch_boundary": [
            s["divJv"] for s in samples if s["near_patch_boundary"]
        ],
        "near_octahedron_vertex": [
            s["divJv"] for s in samples if s["near_octahedron_vertex"]
        ],
    }

    summaries = {
        name: _summarize_values(values)
        for name, values in categories.items()
    }

    top_locations = sorted(
        samples,
        key=lambda sample: abs(sample["divJv"]),
        reverse=True,
    )[:10]

    return {
        "nsub": nsub,
        "summaries": summaries,
        "top_locations": top_locations,
    }


def _face_alignment(xyz_M, xyz_P):
    direct = np.max(np.linalg.norm(xyz_M - xyz_P, axis=1))
    reverse = np.max(np.linalg.norm(xyz_M - xyz_P[::-1], axis=1))

    if reverse < direct:
        return np.arange(xyz_P.shape[0] - 1, -1, -1)

    return np.arange(xyz_P.shape[0])


def _summarize_jumps(jumps):
    return _summarize_values(jumps)


def compute_element_metric_fluxes(engine, diagnostics, u0=1.0, alpha=np.pi / 4.0):
    element_fluxes = []

    for entry in diagnostics:
        xyz_nodes = entry["xyz_nodes"]
        geometry = entry["geometry"]
        V3D = solid_body_rotation_velocity(
            xyz_nodes,
            u0=u0,
            alpha=alpha,
        )
        q = np.ones(engine.num_nodes)

        _, _, u_tilde, v_tilde = compute_manifold_skew_volume_rhs(
            engine=engine,
            geometry=geometry,
            V3D=V3D,
            q=q,
        )

        Fr = geometry["J"] * u_tilde
        Fs = geometry["J"] * v_tilde

        element_fluxes.append(
            {
                "Fr": Fr,
                "Fs": Fs,
                "u_tilde": u_tilde,
                "v_tilde": v_tilde,
            }
        )

    return element_fluxes


def collect_face_metric_flux_jump_samples(
    nsub=16,
    order=4,
    R=1.0,
    u0=1.0,
    alpha=np.pi / 4.0,
):
    engine = build_local_operators(N=order, n=order, rule="table1")
    _, _, EToV, patch_ids, element_local_coords = create_octahedral_layout_mesh(nsub)
    EToE, EToF = build_connectivity(EToV)

    diagnostics = []

    for k in range(EToV.shape[0]):
        patch_id = int(patch_ids[k])
        patch0 = patch_id - 1
        tri_vertices = element_local_coords[k]
        xi, eta = map_reference_nodes_to_subelement(
            engine.r,
            engine.s,
            tri_vertices,
        )
        X, Y, Z = map_unit_triangle_to_sphere(xi, eta, patch0, R=R)
        xyz_nodes = np.column_stack([X, Y, Z])
        geometry = compute_manifold_geometry(engine, xyz_nodes)

        diagnostics.append(
            {
                "element": k,
                "patch_id": patch_id,
                "xi": xi,
                "eta": eta,
                "xyz_nodes": xyz_nodes,
                "geometry": geometry,
            }
        )

    element_fluxes = compute_element_metric_fluxes(
        engine,
        diagnostics,
        u0=u0,
        alpha=alpha,
    )

    near_vertex_tol = NEAR_VERTEX_FACTOR / nsub
    samples = []

    for kM in range(EToE.shape[0]):
        for fM in range(3):
            kP = int(EToE[kM, fM])
            fP = int(EToF[kM, fM])

            if kP == kM:
                continue

            if (kM, fM) > (kP, fP):
                continue

            face_nodes_M = np.arange(
                engine.edge_slices[fM].start,
                engine.edge_slices[fM].stop,
            )
            face_nodes_P = np.arange(
                engine.edge_slices[fP].start,
                engine.edge_slices[fP].stop,
            )

            xyz_M = diagnostics[kM]["xyz_nodes"][face_nodes_M]
            xyz_P_unaligned = diagnostics[kP]["xyz_nodes"][face_nodes_P]
            align = _face_alignment(xyz_M, xyz_P_unaligned)
            face_nodes_P = face_nodes_P[align]

            nr_M, ns_M = REFERENCE_FACE_NORMALS[fM]
            nr_P, ns_P = REFERENCE_FACE_NORMALS[fP]

            Fn_M = (
                nr_M * element_fluxes[kM]["Fr"][face_nodes_M]
                + ns_M * element_fluxes[kM]["Fs"][face_nodes_M]
            )
            Fn_P = (
                nr_P * element_fluxes[kP]["Fr"][face_nodes_P]
                + ns_P * element_fluxes[kP]["Fs"][face_nodes_P]
            )
            jump = Fn_M + Fn_P

            for i, (node_M, node_P) in enumerate(zip(face_nodes_M, face_nodes_P)):
                X_M = diagnostics[kM]["xyz_nodes"][node_M]
                X_P = diagnostics[kP]["xyz_nodes"][node_P]
                xi_M = float(diagnostics[kM]["xi"][node_M])
                eta_M = float(diagnostics[kM]["eta"][node_M])
                xi_P = float(diagnostics[kP]["xi"][node_P])
                eta_P = float(diagnostics[kP]["eta"][node_P])

                patch_boundary_distance = min(
                    _patch_boundary_distance(xi_M, eta_M),
                    _patch_boundary_distance(xi_P, eta_P),
                )
                vertex_distance = min(
                    _distance_to_original_octahedron_vertex(X_M, R=R),
                    _distance_to_original_octahedron_vertex(X_P, R=R),
                )
                near_patch_boundary = (
                    patch_boundary_distance < PATCH_BOUNDARY_TOL
                    or diagnostics[kM]["patch_id"] != diagnostics[kP]["patch_id"]
                )
                near_octahedron_vertex = vertex_distance <= near_vertex_tol

                samples.append(
                    {
                        "element_M": kM,
                        "element_P": kP,
                        "face_M": fM,
                        "face_P": fP,
                        "node_M": int(node_M),
                        "node_P": int(node_P),
                        "local_face_node": i,
                        "patch_id_M": int(diagnostics[kM]["patch_id"]),
                        "patch_id_P": int(diagnostics[kP]["patch_id"]),
                        "X_M": X_M.copy(),
                        "X_P": X_P.copy(),
                        "Fn_M": float(Fn_M[i]),
                        "Fn_P": float(Fn_P[i]),
                        "jump_Fn": float(jump[i]),
                        "near_patch_boundary": near_patch_boundary,
                        "near_octahedron_vertex": near_octahedron_vertex,
                        "patch_boundary_distance": patch_boundary_distance,
                        "octahedron_vertex_distance": vertex_distance,
                    }
                )

    return samples


def compute_face_metric_flux_jump_diagnostic(nsub=16):
    samples = collect_face_metric_flux_jump_samples(nsub=nsub)
    jumps = np.array([sample["jump_Fn"] for sample in samples])

    categories = {
        "all_shared_faces": jumps,
        "not_near_patch_boundary": [
            sample["jump_Fn"]
            for sample in samples
            if not sample["near_patch_boundary"]
        ],
        "near_patch_boundary": [
            sample["jump_Fn"]
            for sample in samples
            if sample["near_patch_boundary"]
        ],
        "near_octahedron_vertex": [
            sample["jump_Fn"]
            for sample in samples
            if sample["near_octahedron_vertex"]
        ],
    }

    summaries = {
        name: _summarize_jumps(values)
        for name, values in categories.items()
    }

    top_locations = sorted(
        samples,
        key=lambda sample: abs(sample["jump_Fn"]),
        reverse=True,
    )[:10]

    return {
        "nsub": nsub,
        "summaries": summaries,
        "top_locations": top_locations,
    }


def test_divergence_refinement_table_and_location():
    nsubs = [2, 4, 8, 16]
    rows = []
    previous_max = None
    previous_rms = None

    print("\n" + "=" * 106)
    print("Manifold sphere divJv refinement diagnostic")
    print("=" * 106)
    print(
        f"{'nsub':>8s} "
        f"{'max_abs_divJv':>16s} "
        f"{'rms_divJv':>16s} "
        f"{'mean_divJv':>16s} "
        f"{'order(max)':>12s} "
        f"{'order(rms)':>12s}"
    )
    print("-" * 106)

    for nsub in nsubs:
        result = compute_divergence_refinement_diagnostic(nsub=nsub)

        if previous_max is None:
            order_max = None
            order_rms = None
            order_max_text = "-"
            order_rms_text = "-"
        else:
            order_max = np.log(previous_max / result["max_abs_divJv"]) / np.log(2.0)
            order_rms = np.log(previous_rms / result["rms_divJv"]) / np.log(2.0)
            order_max_text = f"{order_max:.4f}"
            order_rms_text = f"{order_rms:.4f}"

        rows.append((result, order_max, order_rms))

        print(
            f"{nsub:8d} "
            f"{result['max_abs_divJv']:16.6e} "
            f"{result['rms_divJv']:16.6e} "
            f"{result['mean_divJv']:16.6e} "
            f"{order_max_text:>12s} "
            f"{order_rms_text:>12s}"
        )

        previous_max = result["max_abs_divJv"]
        previous_rms = result["rms_divJv"]

    print("-" * 106)

    finest = rows[-1][0]
    loc = finest["max_location"]

    print("Max |divJv| location on finest mesh:")
    print(f"  nsub = {finest['nsub']}")
    print(f"  element = {loc['element']}")
    print(f"  node = {loc['node']}")
    print(f"  patch_id = {loc['patch_id']}")
    print(f"  xi = {loc['xi']:.12e}")
    print(f"  eta = {loc['eta']:.12e}")
    print(
        "  X = "
        f"({loc['X'][0]:.12e}, {loc['X'][1]:.12e}, {loc['X'][2]:.12e})"
    )
    print(f"  J = {loc['J']:.12e}")
    print(f"  u_tilde = {loc['u_tilde']:.12e}")
    print(f"  v_tilde = {loc['v_tilde']:.12e}")
    print(f"  divJv = {loc['divJv']:.12e}")
    print(f"  patch_boundary_distance = {loc['patch_boundary_distance']:.12e}")
    print(f"  near_patch_boundary = {loc['near_patch_boundary']}")
    print(f"  octahedron_vertex_distance = {loc['octahedron_vertex_distance']:.12e}")
    print(f"  near_octahedron_vertex = {loc['near_octahedron_vertex']}")
    print("=" * 106)

    for result, order_max, order_rms in rows:
        assert np.isfinite(result["max_abs_divJv"])
        assert np.isfinite(result["rms_divJv"])
        assert np.isfinite(result["mean_divJv"])
        if order_max is not None:
            assert np.isfinite(order_max)
            assert np.isfinite(order_rms)


def _print_category_row(nsub, category, summary):
    print(
        f"{nsub:8d} "
        f"{category:<24s} "
        f"{summary['count']:10d} "
        f"{summary['max_abs_divJv']:16.6e} "
        f"{summary['rms_divJv']:16.6e} "
        f"{summary['mean_divJv']:16.6e}"
    )


def _print_location(rank, loc):
    print(f"#{rank}")
    print(f"  element = {loc['element']}")
    print(f"  node = {loc['node']}")
    print(f"  patch_id = {loc['patch_id']}")
    print(f"  xi = {loc['xi']:.12e}")
    print(f"  eta = {loc['eta']:.12e}")
    print(
        "  X = "
        f"({loc['X'][0]:.12e}, {loc['X'][1]:.12e}, {loc['X'][2]:.12e})"
    )
    print(f"  J = {loc['J']:.12e}")
    print(f"  u_tilde = {loc['u_tilde']:.12e}")
    print(f"  v_tilde = {loc['v_tilde']:.12e}")
    print(f"  divJv = {loc['divJv']:.12e}")
    print(f"  near_patch_boundary = {loc['near_patch_boundary']}")
    print(f"  near_octahedron_vertex = {loc['near_octahedron_vertex']}")
    print(f"  patch_boundary_distance = {loc['patch_boundary_distance']:.12e}")
    print(f"  octahedron_vertex_distance = {loc['octahedron_vertex_distance']:.12e}")


def test_boundary_vs_interior_divergence_diagnostic():
    nsubs = [2, 4, 8, 16]
    all_results = []

    print("\n" + "=" * 104)
    print("Manifold sphere boundary vs interior divJv diagnostic")
    print("=" * 104)
    print(
        f"{'nsub':>8s} "
        f"{'category':<24s} "
        f"{'count':>10s} "
        f"{'max_abs_divJv':>16s} "
        f"{'rms_divJv':>16s} "
        f"{'mean_divJv':>16s}"
    )
    print("-" * 104)

    for nsub in nsubs:
        result = compute_boundary_vs_interior_diagnostic(nsub=nsub)
        all_results.append(result)

        for category in [
            "interior",
            "face_boundary",
            "near_patch_boundary",
            "near_octahedron_vertex",
        ]:
            _print_category_row(
                nsub,
                category,
                result["summaries"][category],
            )

    print("-" * 104)

    finest = all_results[-1]
    print("Top 10 |divJv| locations on finest mesh:")
    print("-" * 104)

    top_locations = finest["top_locations"]

    for rank, loc in enumerate(top_locations, start=1):
        _print_location(rank, loc)

    top10_boundary_count = sum(
        1 for loc in top_locations if loc["near_patch_boundary"]
    )
    top10_vertex_count = sum(
        1 for loc in top_locations if loc["near_octahedron_vertex"]
    )

    print("-" * 104)
    print(f"top10_near_patch_boundary_count = {top10_boundary_count}")
    print(f"top10_near_octahedron_vertex_count = {top10_vertex_count}")
    print("=" * 104)

    for result in all_results:
        for summary in result["summaries"].values():
            assert summary["count"] > 0
            assert np.isfinite(summary["max_abs_divJv"])
            assert np.isfinite(summary["rms_divJv"])
            assert np.isfinite(summary["mean_divJv"])


def _print_jump_summary_row(nsub, category, summary):
    print(
        f"{nsub:8d} "
        f"{category:<28s} "
        f"{summary['count']:10d} "
        f"{summary['max_abs_divJv']:16.6e} "
        f"{summary['rms_divJv']:16.6e} "
        f"{summary['mean_divJv']:16.6e}"
    )


def _print_jump_location(rank, loc):
    print(f"#{rank}")
    print(f"  element_M = {loc['element_M']}")
    print(f"  element_P = {loc['element_P']}")
    print(f"  face_M = {loc['face_M']}")
    print(f"  face_P = {loc['face_P']}")
    print(f"  node_M = {loc['node_M']}")
    print(f"  node_P = {loc['node_P']}")
    print(f"  local_face_node = {loc['local_face_node']}")
    print(f"  patch_id_M = {loc['patch_id_M']}")
    print(f"  patch_id_P = {loc['patch_id_P']}")
    print(
        "  X_M = "
        f"({loc['X_M'][0]:.12e}, {loc['X_M'][1]:.12e}, {loc['X_M'][2]:.12e})"
    )
    print(
        "  X_P = "
        f"({loc['X_P'][0]:.12e}, {loc['X_P'][1]:.12e}, {loc['X_P'][2]:.12e})"
    )
    print(f"  Fn_M = {loc['Fn_M']:.12e}")
    print(f"  Fn_P = {loc['Fn_P']:.12e}")
    print(f"  jump_Fn = {loc['jump_Fn']:.12e}")
    print(f"  near_patch_boundary = {loc['near_patch_boundary']}")
    print(f"  near_octahedron_vertex = {loc['near_octahedron_vertex']}")
    print(f"  patch_boundary_distance = {loc['patch_boundary_distance']:.12e}")
    print(f"  octahedron_vertex_distance = {loc['octahedron_vertex_distance']:.12e}")


def test_face_metric_flux_continuity_diagnostic():
    nsubs = [2, 4, 8, 16]
    results = []

    print("\n" + "=" * 110)
    print("Manifold sphere face metric-flux continuity diagnostic")
    print("=" * 110)
    print(
        f"{'nsub':>8s} "
        f"{'category':<28s} "
        f"{'count':>10s} "
        f"{'max_abs_jump_Fn':>16s} "
        f"{'rms_jump_Fn':>16s} "
        f"{'mean_jump_Fn':>16s}"
    )
    print("-" * 110)

    for nsub in nsubs:
        result = compute_face_metric_flux_jump_diagnostic(nsub=nsub)
        results.append(result)

        for category in [
            "all_shared_faces",
            "not_near_patch_boundary",
            "near_patch_boundary",
            "near_octahedron_vertex",
        ]:
            _print_jump_summary_row(
                nsub,
                category,
                result["summaries"][category],
            )

    print("-" * 110)

    finest = results[-1]
    print("Top 10 |jump_Fn| locations on finest mesh:")
    print("-" * 110)

    for rank, loc in enumerate(finest["top_locations"], start=1):
        _print_jump_location(rank, loc)

    top10_patch_boundary_count = sum(
        1 for loc in finest["top_locations"] if loc["near_patch_boundary"]
    )
    top10_vertex_count = sum(
        1 for loc in finest["top_locations"] if loc["near_octahedron_vertex"]
    )

    print("-" * 110)
    print(f"top10_near_patch_boundary_count = {top10_patch_boundary_count}")
    print(f"top10_near_octahedron_vertex_count = {top10_vertex_count}")
    print("=" * 110)

    for result in results:
        for summary in result["summaries"].values():
            if summary["count"] == 0:
                continue
            assert np.isfinite(summary["max_abs_divJv"])
            assert np.isfinite(summary["rms_divJv"])
            assert np.isfinite(summary["mean_divJv"])


def test_constant_state_volume_rhs_diagnostic_is_finite():
    result = compute_constant_state_rhs_diagnostics(
        nsub=8,
        order=4,
        R=1.0,
        u0=1.0,
        alpha=np.pi / 4.0,
    )

    status = "completed" if result["near_zero"] else "needs investigation"

    print("\n" + "=" * 92)
    print("Manifold sphere constant-state volume RHS diagnostic")
    print("=" * 92)
    print(f"status = {status}")
    print(f"nsub = {result['nsub']}")
    print(f"order = {result['order']}")
    print(f"R = {result['R']:.6e}")
    print(f"u0 = {result['u0']:.6e}")
    print(f"alpha = {result['alpha']:.12e}")
    print(f"max_abs_rhs = {result['max_abs_rhs']:.6e}")
    print(f"rms_rhs = {result['rms_rhs']:.6e}")
    print(f"mean_rhs = {result['mean_rhs']:.6e}")
    print(f"max_abs_divJv = {result['max_abs_divJv']:.6e}")
    print(f"near_zero_tolerance = {NEAR_ZERO_RHS_TOL:.6e}")
    print("=" * 92)

    assert np.isfinite(result["max_abs_rhs"])
    assert np.isfinite(result["rms_rhs"])
    assert np.isfinite(result["mean_rhs"])
    assert np.isfinite(result["max_abs_divJv"])


def run_all_tests():
    print("\n" + "=" * 92)
    print("Manifold sphere constant-state volume RHS diagnostic")
    print("=" * 92)

    test_constant_state_volume_rhs_diagnostic_is_finite()
    print("constant-state volume RHS diagnostic completed")

    test_projected_mesh_constant_state_metric_divergence_diagnostic()
    print("projected mesh constant-state metric divergence diagnostic completed")

    test_divergence_refinement_table_and_location()
    print("divJv refinement diagnostic completed")

    test_boundary_vs_interior_divergence_diagnostic()
    print("boundary vs interior divJv diagnostic completed")

    test_face_metric_flux_continuity_diagnostic()
    print("face metric-flux continuity diagnostic completed")

    print("=" * 92)
    print("test_manifold_constant_rhs_sphere.py completed")


if __name__ == "__main__":
    run_all_tests()
