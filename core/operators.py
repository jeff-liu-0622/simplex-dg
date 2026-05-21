import numpy as np
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

from core.quadrature import get_reference_nodes
from core.basis import build_vandermonde2d


@dataclass
class ReferenceElement:
    """
    Reference triangle operator engine for the SDG implementation.

    Reference triangle:
        vertices = (-1, -1), (1, -1), (-1, 1)

    Domain:
        r >= -1,
        s >= -1,
        r + s <= 0

    Area:
        |T_ref| = 2

    Edge ordering follows quadrature.py:

        edge 1: s = -1
            from (-1, -1) to (1, -1)

        edge 2: r + s = 0
            from (1, -1) to (-1, 1)

        edge 3: r = -1
            from (-1, 1) to (-1, -1)

    Boundary nodes are ordered as:

        edge1 nodes, edge2 nodes, edge3 nodes, interior nodes

    Hence the extraction matrix E = [I | 0] is implicit.
    """

    # Basic parameters
    N: int
    n: int
    rule: str

    # Coordinates
    xi: np.ndarray
    eta: np.ndarray
    r: np.ndarray
    s: np.ndarray

    # Quadrature weights
    w_s: np.ndarray
    w_e: Optional[np.ndarray]

    # Reference geometry
    area: float
    edge_lengths: np.ndarray

    # Boundary structure
    num_edge_nodes: int

    # Basis matrices
    V: np.ndarray
    Vr: np.ndarray
    Vs: np.ndarray

    # Quadrature and mass matrix
    W: np.ndarray
    M_modal: np.ndarray
    invM_modal: np.ndarray

    # Nodal differentiation matrices on the reference triangle
    Dr: np.ndarray
    Ds: np.ndarray

    @property
    def num_nodes(self) -> int:
        """Total number of quadrature / nodal points."""
        return self.r.size

    @property
    def num_basis(self) -> int:
        """Number of modal basis functions."""
        return self.V.shape[1]

    @property
    def num_boundary_nodes(self) -> int:
        """Total number of boundary nodes on all three edges."""
        return 3 * self.num_edge_nodes

    @property
    def boundary_slice(self) -> slice:
        """Slice selecting all boundary nodes."""
        return slice(0, self.num_boundary_nodes)

    @property
    def edge_slices(self) -> Tuple[slice, slice, slice]:
        """
        Slices for edge1, edge2, edge3 boundary nodes.

        Boundary node ordering:
            edge1, edge2, edge3, interior
        """
        Nfp = self.num_edge_nodes
        return (
            slice(0, Nfp),
            slice(Nfp, 2 * Nfp),
            slice(2 * Nfp, 3 * Nfp),
        )

    def get_modal_coeffs(self, u_nodes: np.ndarray) -> np.ndarray:
        """
        Galerkin projection from nodal values to modal coefficients.

        Formula:
            a = M^{-1} |T| V^T W u

        Supports:
            u_nodes shape = (num_nodes,)
            u_nodes shape = (num_nodes, num_components)
        """
        u_nodes = np.asarray(u_nodes)

        if u_nodes.shape[0] != self.num_nodes:
            raise ValueError(
                f"Expected u_nodes first dimension {self.num_nodes}, "
                f"got {u_nodes.shape[0]}."
            )

        if u_nodes.ndim == 1:
            rhs = self.area * self.V.T @ (self.w_s * u_nodes)
        elif u_nodes.ndim == 2:
            rhs = self.area * self.V.T @ (self.w_s[:, None] * u_nodes)
        else:
            raise ValueError("u_nodes must be a 1D or 2D array.")

        return self.invM_modal @ rhs

    def modal_to_nodal(self, coeffs: np.ndarray) -> np.ndarray:
        """
        Convert modal coefficients to nodal values.

        Formula:
            u = V a
        """
        return self.V @ coeffs

    def boundary_values(self, u_nodes: np.ndarray) -> np.ndarray:
        """
        Extract boundary nodal values.

        Because quadrature.py orders nodes as:
            edge1 + edge2 + edge3 + interior

        this is simply:
            u[:3*num_edge_nodes]
        """
        u_nodes = np.asarray(u_nodes)

        if u_nodes.shape[0] != self.num_nodes:
            raise ValueError(
                f"Expected u_nodes first dimension {self.num_nodes}, "
                f"got {u_nodes.shape[0]}."
            )

        return u_nodes[self.boundary_slice]

    def boundary_weight_diag(
        self,
        edge_lengths: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Construct diagonal entries of W_b.

        Formula:

            W_b =
            diag(|edge_1|, |edge_2|, |edge_3|)
            kron diag(w_e)

        For the reference triangle:
            edge_lengths = [2, 2*sqrt(2), 2]

        Parameters
        ----------
        edge_lengths:
            Optional physical edge lengths.
            If None, uses the reference edge lengths.

        Returns
        -------
        Wb_diag:
            1D array of length 3*num_edge_nodes.
        """
        if self.w_e is None:
            raise ValueError(
                "Boundary quadrature weights w_e are missing. "
                "Use rule='table1' for the SDG boundary penalty."
            )

        if edge_lengths is None:
            edge_lengths = self.edge_lengths
        else:
            edge_lengths = np.asarray(edge_lengths, dtype=float)

        if edge_lengths.shape != (3,):
            raise ValueError("edge_lengths must have shape (3,).")

        return np.repeat(edge_lengths, self.num_edge_nodes) * np.tile(self.w_e, 3)

    def lift_boundary_penalty(
        self,
        p_boundary: np.ndarray,
        edge_lengths: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Lift boundary penalty back to volume nodes.

        This implements the SDG boundary term:

            V M^{-1} V^T E^T W_b p

        where p is ordered as:

            [edge1 values, edge2 values, edge3 values]

        Parameters
        ----------
        p_boundary:
            Boundary penalty vector.

            Shape can be:
                (3*num_edge_nodes,)
                or
                (3*num_edge_nodes, num_components)

        edge_lengths:
            Optional physical edge lengths.
            If None, uses reference edge lengths:
                [2, 2*sqrt(2), 2]

        Returns
        -------
        lifted:
            Penalty contribution at all volume nodes.

            Shape:
                (num_nodes,)
                or
                (num_nodes, num_components)

        Important
        ---------
        This function already multiplies by W_b, including edge lengths
        and edge quadrature weights.

        Do not multiply edge lengths again in rhs.py unless you intentionally
        pass edge_lengths=np.ones(3).
        """
        p_boundary = np.asarray(p_boundary)

        Nb = self.num_boundary_nodes

        if p_boundary.shape[0] != Nb:
            raise ValueError(
                f"Expected p_boundary first dimension {Nb}, "
                f"got {p_boundary.shape[0]}."
            )

        Wb_diag = self.boundary_weight_diag(edge_lengths=edge_lengths)

        if p_boundary.ndim == 1:
            weighted_boundary = Wb_diag * p_boundary

            full = np.zeros(self.num_nodes)
            full[:Nb] = weighted_boundary

        elif p_boundary.ndim == 2:
            weighted_boundary = Wb_diag[:, None] * p_boundary

            full = np.zeros((self.num_nodes, p_boundary.shape[1]))
            full[:Nb, :] = weighted_boundary

        else:
            raise ValueError("p_boundary must be a 1D or 2D array.")

        return self.V @ (self.invM_modal @ (self.V.T @ full))


def build_local_operators(
    N: int,
    n: int,
    rule: str = "table1",
) -> ReferenceElement:
    """
    Build local SDG operators on the reference triangle.

    Parameters
    ----------
    N:
        Polynomial degree.

    n:
        Quadrature parameter from the SDG notes.

    rule:
        'table1':
            Degree 2n-1 volume quadrature.
            Includes boundary nodes and edge weights.
            This is the natural choice for the SDG penalty form E=[I|0].

        'table2':
            Degree 2n volume quadrature.
            Does not include boundary edge weights.
            Not directly suitable for the simple SDG boundary penalty.
    """

    if N < 1:
        raise ValueError("Polynomial degree N must be >= 1.")

    if rule == "table1":
        if n < N:
            warnings.warn(
                f"Table1 quadrature has degree 2n-1. "
                f"Current n={n}, N={N}. This may be under-integrated."
            )

        if n < N + 1:
            warnings.warn(
                f"For exact modal mass with degree N={N}, "
                f"Table1 would need n >= N+1 because mass entries involve "
                f"degree 2N polynomials. Current n={n}. "
                f"This may still be acceptable for the collocated-boundary "
                f"SDG construction, but be aware of aliasing."
            )

    elif rule == "table2":
        if n < N:
            warnings.warn(
                f"Table2 quadrature has degree 2n. "
                f"For degree N={N}, usually need n >= N."
            )

        warnings.warn(
            "rule='table2' has no boundary nodes or edge weights. "
            "The simple SDG penalty lift E=[I|0] cannot be used directly."
        )

    else:
        raise ValueError("rule must be 'table1' or 'table2'.")

    # ------------------------------------------------------------
    # 1. Quadrature nodes and weights
    # ------------------------------------------------------------
    nodes = get_reference_nodes(n, rule=rule)

    xi = nodes["xi"]
    eta = nodes["eta"]
    r = nodes["r"]
    s = nodes["s"]
    w_s = nodes["w_s"]
    w_e = nodes["w_e"]
    num_edge_nodes = nodes["num_edge_nodes"]

    # ------------------------------------------------------------
    # 2. Reference geometry
    #
    # Reference triangle:
    #     (-1,-1), (1,-1), (-1,1)
    #
    # Area:
    #     2
    #
    # Edge lengths:
    #     edge1: from (-1,-1) to (1,-1)     length = 2
    #     edge2: from (1,-1) to (-1,1)      length = 2sqrt(2)
    #     edge3: from (-1,1) to (-1,-1)     length = 2
    # ------------------------------------------------------------
    area = 2.0
    edge_lengths = np.array([2.0, 2.0 * np.sqrt(2.0), 2.0])

    # ------------------------------------------------------------
    # 3. Basis matrices
    # ------------------------------------------------------------
    V, Vr, Vs = build_vandermonde2d(N, r, s)

    num_nodes = r.size
    num_basis = V.shape[1]

    if num_nodes < num_basis:
        raise ValueError(
            f"Number of quadrature nodes is smaller than number of basis functions: "
            f"num_nodes={num_nodes}, num_basis={num_basis}."
        )

    rank = np.linalg.matrix_rank(V)
    if rank < num_basis:
        raise ValueError(
            f"Vandermonde matrix is rank deficient: rank={rank}, "
            f"num_basis={num_basis}."
        )

    # ------------------------------------------------------------
    # 4. Mass matrix
    #
    # M = |T| V^T W V
    # ------------------------------------------------------------
    W = np.diag(w_s)

    M_modal = area * (V.T @ W @ V)
    invM_modal = np.linalg.inv(M_modal)

    # ------------------------------------------------------------
    # 5. Nodal differentiation matrices
    #
    # Dr = |T| Vr M^{-1} V^T W
    # Ds = |T| Vs M^{-1} V^T W
    #
    # These differentiate with respect to reference coordinates r, s.
    # ------------------------------------------------------------
    Dr = area * Vr @ invM_modal @ (V.T @ W)
    Ds = area * Vs @ invM_modal @ (V.T @ W)

    return ReferenceElement(
        N=N,
        n=n,
        rule=rule,
        xi=xi,
        eta=eta,
        r=r,
        s=s,
        w_s=w_s,
        w_e=w_e,
        area=area,
        edge_lengths=edge_lengths,
        num_edge_nodes=num_edge_nodes,
        V=V,
        Vr=Vr,
        Vs=Vs,
        W=W,
        M_modal=M_modal,
        invM_modal=invM_modal,
        Dr=Dr,
        Ds=Ds,
    )

def compute_manifold_volume_rhs_fast(q, state):
    engine = state["engine"]

    J = state["J_array"]
    J_u = state["J_u"]
    J_v = state["J_v"]
    div_Jv = state["div_Jv"]

    Dr = engine.Dr
    Ds = engine.Ds

    Dr_q = q @ Dr.T
    Ds_q = q @ Ds.T

    Dr_Juq = (J_u * q) @ Dr.T
    Ds_Jvq = (J_v * q) @ Ds.T

    rhs_Jq = (
        -0.5 * (Dr_Juq + Ds_Jvq)
        -0.5 * (J_u * Dr_q + J_v * Ds_q)
        -0.5 * q * div_Jv
    )

    rhs_q = rhs_Jq / J

    return rhs_q