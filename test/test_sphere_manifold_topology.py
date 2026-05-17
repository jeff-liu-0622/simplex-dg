import numpy as np

from core.geometry.connectivity import build_connectivity
from core.geometry.sphere_manifold_topology import (
    create_projected_octahedron_sphere_mesh,
    map_reference_nodes_to_projected_sphere,
)
from core.operators import build_local_operators


def _aligned_face_error(xyz_M, xyz_P):
    direct = np.max(np.linalg.norm(xyz_M - xyz_P, axis=1))
    reverse = np.max(np.linalg.norm(xyz_M - xyz_P[::-1], axis=1))

    return min(direct, reverse), reverse < direct


def compute_shared_face_match_diagnostic(nsub, order=4, R=1.0):
    engine = build_local_operators(N=order, n=order, rule="table1")
    _, _, _, EToV, patch_ids, nodes_xyz = create_projected_octahedron_sphere_mesh(
        nsub=nsub,
        R=R,
    )

    EToE, EToF = build_connectivity(EToV)
    xyz_nodes = map_reference_nodes_to_projected_sphere(
        nodes_xyz=nodes_xyz,
        EToV=EToV,
        r=engine.r,
        s=engine.s,
        R=R,
    )

    errors = []
    worst_faces = []

    for kM in range(EToE.shape[0]):
        for fM in range(3):
            kP = int(EToE[kM, fM])
            fP = int(EToF[kM, fM])

            if kP == kM:
                raise AssertionError(f"unexpected boundary face ({kM}, {fM})")

            if (kM, fM) > (kP, fP):
                continue

            nodes_M = np.arange(
                engine.edge_slices[fM].start,
                engine.edge_slices[fM].stop,
            )
            nodes_P = np.arange(
                engine.edge_slices[fP].start,
                engine.edge_slices[fP].stop,
            )

            face_xyz_M = xyz_nodes[kM, nodes_M, :]
            face_xyz_P = xyz_nodes[kP, nodes_P, :]
            error, flipped = _aligned_face_error(face_xyz_M, face_xyz_P)

            errors.append(error)
            worst_faces.append(
                {
                    "error": float(error),
                    "elem_M": int(kM),
                    "face_M": int(fM),
                    "elem_P": int(kP),
                    "face_P": int(fP),
                    "patch_M": int(patch_ids[kM]),
                    "patch_P": int(patch_ids[kP]),
                    "flipped": bool(flipped),
                }
            )

    errors = np.asarray(errors, dtype=float)
    worst_faces = sorted(worst_faces, key=lambda row: row["error"], reverse=True)

    return {
        "nsub": nsub,
        "K": int(EToV.shape[0]),
        "Nv": int(nodes_xyz.shape[0]),
        "num_shared_faces": int(errors.size),
        "max_face_match_error": float(np.max(errors)),
        "rms_face_match_error": float(np.sqrt(np.mean(errors**2))),
        "worst_faces": worst_faces[:5],
    }


def test_projected_octahedron_shared_faces_match_in_3d():
    nsubs = [2, 4, 8, 16]
    results = []

    print("\n" + "=" * 112)
    print("Projected octahedron sphere topology face-continuity diagnostic")
    print("=" * 112)
    print(
        f"{'nsub':>8s} "
        f"{'Nv':>8s} "
        f"{'K':>8s} "
        f"{'shared':>8s} "
        f"{'max_face_match_error':>22s} "
        f"{'rms_face_match_error':>22s}"
    )
    print("-" * 112)

    for nsub in nsubs:
        result = compute_shared_face_match_diagnostic(nsub=nsub)
        results.append(result)

        print(
            f"{result['nsub']:8d} "
            f"{result['Nv']:8d} "
            f"{result['K']:8d} "
            f"{result['num_shared_faces']:8d} "
            f"{result['max_face_match_error']:22.6e} "
            f"{result['rms_face_match_error']:22.6e}"
        )

    print("-" * 112)
    print("Worst faces on finest mesh:")

    for row in results[-1]["worst_faces"]:
        print(
            "  "
            f"err={row['error']:.6e}, "
            f"M=({row['elem_M']}, face {row['face_M']}, patch {row['patch_M']}), "
            f"P=({row['elem_P']}, face {row['face_P']}, patch {row['patch_P']}), "
            f"flipped={row['flipped']}"
        )

    print("=" * 112)

    for result in results:
        assert result["max_face_match_error"] < 1.0e-12
        assert result["rms_face_match_error"] < 1.0e-12


if __name__ == "__main__":
    test_projected_octahedron_shared_faces_match_in_3d()
