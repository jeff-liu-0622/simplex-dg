import numpy as np
import matplotlib.pyplot as plt

from core.mesh_octahedron import create_octahedral_layout_mesh
from core.geometry.sphere_mapping import map_unit_triangle_to_sphere
from core.operators import build_local_operators


def reference_to_unit_triangle(r, s):
    """
    Map reference triangle
        (-1,-1), (1,-1), (-1,1)
    to unit triangle
        (0,0), (1,0), (0,1)
    """
    a = 0.5 * (r + 1.0)
    b = 0.5 * (s + 1.0)
    return a, b


def map_reference_nodes_to_subelement(r, s, tri_vertices):
    """
    Map reference nodes (r,s) on the standard reference triangle
    to one subelement inside the unit triangle.

    Parameters
    ----------
    r, s:
        Arrays of reference nodes on
            (-1,-1), (1,-1), (-1,1)

    tri_vertices:
        shape (3,2), the three vertices of one subelement
        in unit-triangle coordinates (xi, eta),
        ordered as local v0, v1, v2.

    Returns
    -------
    xi, eta:
        Arrays of mapped nodal coordinates in the parent patch.
    """
    a, b = reference_to_unit_triangle(r, s)

    v0 = tri_vertices[0]
    v1 = tri_vertices[1]
    v2 = tri_vertices[2]

    xi = v0[0] + a * (v1[0] - v0[0]) + b * (v2[0] - v0[0])
    eta = v0[1] + a * (v1[1] - v0[1]) + b * (v2[1] - v0[1])

    return xi, eta


def plot_octahedral_sphere_element_nodes(
    nsub=5,
    order=4,
    rule="table1",
    R=1.0,
    show_nodes=True,
    show_edges=True,
):
    """
    Plot 3D sphere submesh with high-order element nodes.

    nsub:
        number of subdivisions per Ti patch edge

    order:
        polynomial degree / local operator degree

    rule:
        nodal rule passed to build_local_operators

    R:
        sphere radius
    """
    VX, VY, EToV, patch_ids, element_local_coords = create_octahedral_layout_mesh(nsub)

    engine = build_local_operators(N=order, n=order, rule=rule)
    r = engine.r
    s = engine.s

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    cmap = plt.get_cmap("tab10")

    # faint sphere wireframe
    u = np.linspace(0.0, 2.0 * np.pi, 80)
    v = np.linspace(-0.5 * np.pi, 0.5 * np.pi, 40)
    uu, vv = np.meshgrid(u, v)

    Xs = R * np.cos(vv) * np.cos(uu)
    Ys = R * np.cos(vv) * np.sin(uu)
    Zs = R * np.sin(vv)

    ax.plot_wireframe(Xs, Ys, Zs, linewidth=0.2, alpha=0.08)

    # draw each spherical subelement
    for k in range(EToV.shape[0]):
        pid = patch_ids[k]          # 1..8
        patch0 = pid - 1            # sphere_mapping uses 0..7
        color = cmap((pid - 1) % 10)

        tri_vertices = element_local_coords[k]   # shape (3,2)

        # --- plot subelement edges on sphere ---
        if show_edges:
            edge_t = np.linspace(0.0, 1.0, 40)

            # local subelement edges in unit triangle coordinates
            edge_defs = [
                # v0 -> v1
                (
                    tri_vertices[0][0] + edge_t * (tri_vertices[1][0] - tri_vertices[0][0]),
                    tri_vertices[0][1] + edge_t * (tri_vertices[1][1] - tri_vertices[0][1]),
                ),
                # v1 -> v2
                (
                    tri_vertices[1][0] + edge_t * (tri_vertices[2][0] - tri_vertices[1][0]),
                    tri_vertices[1][1] + edge_t * (tri_vertices[2][1] - tri_vertices[1][1]),
                ),
                # v2 -> v0
                (
                    tri_vertices[2][0] + edge_t * (tri_vertices[0][0] - tri_vertices[2][0]),
                    tri_vertices[2][1] + edge_t * (tri_vertices[0][1] - tri_vertices[2][1]),
                ),
            ]

            for xi_edge, eta_edge in edge_defs:
                Xe, Ye, Ze = map_unit_triangle_to_sphere(xi_edge, eta_edge, patch0, R=R)
                ax.plot(Xe, Ye, Ze, color=color, linewidth=0.8, alpha=0.9)

        # --- map high-order DG nodes to this subelement ---
        if show_nodes:
            xi_nodes, eta_nodes = map_reference_nodes_to_subelement(r, s, tri_vertices)
            Xn, Yn, Zn = map_unit_triangle_to_sphere(xi_nodes, eta_nodes, patch0, R=R)

            ax.scatter(
                Xn,
                Yn,
                Zn,
                s=8,
                color=color,
                alpha=0.85,
                edgecolors="none",
            )

    # patch labels
    for pid in range(1, 9):
        patch0 = pid - 1
        xi_c = np.array([1.0 / 3.0])
        eta_c = np.array([1.0 / 3.0])
        Xc, Yc, Zc = map_unit_triangle_to_sphere(xi_c, eta_c, patch0, R=R)

        ax.text(
            Xc[0],
            Yc[0],
            Zc[0],
            f"T{pid}",
            fontsize=16,
            weight="bold",
            color="black",
        )

    # axes arrows
    L = 1.25 * R
    ax.quiver(0, 0, 0, L, 0, 0, arrow_length_ratio=0.08, linewidth=2.2)
    ax.quiver(0, 0, 0, 0, L, 0, arrow_length_ratio=0.08, linewidth=2.2)
    ax.quiver(0, 0, 0, 0, 0, L, arrow_length_ratio=0.08, linewidth=2.2)

    ax.text(L, 0, 0, "+X", fontsize=14)
    ax.text(0, L, 0, "+Y", fontsize=14)
    ax.text(0, 0, L, "+Z", fontsize=14)

    ax.set_title(
        f"3D sphere submesh with element nodes "
        f"(nsub={nsub}, {rule}, order={order})"
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_box_aspect([1, 1, 1])

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_octahedral_sphere_element_nodes(
        nsub=5,
        order=4,
        rule="table1",
        R=1.0,
        show_nodes=True,
        show_edges=True,
    )