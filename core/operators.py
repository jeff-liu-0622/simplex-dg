import numpy as np
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

from core.quadrature import get_face_quadrature, get_reference_nodes
from core.basis import build_vandermonde2d


@dataclass
class ReferenceElement:
    """Reference triangle operator engine for the SDG implementation."""

    N: int
    n: int
    rule: str
    xi: np.ndarray
    eta: np.ndarray
    r: np.ndarray
    s: np.ndarray
    w_s: np.ndarray
    w_e: Optional[np.ndarray]
    area: float
    edge_lengths: np.ndarray
    num_edge_nodes: int
    face_r: np.ndarray
    face_s: np.ndarray
    face_w: np.ndarray
    R_faces: Tuple[np.ndarray, np.ndarray, np.ndarray]
    R_all: np.ndarray
    V: np.ndarray
    Vr: np.ndarray
    Vs: np.ndarray
    W: np.ndarray
    M_modal: np.ndarray
    invM_modal: np.ndarray
    Dr: np.ndarray
    Ds: np.ndarray

    @property
    def num_nodes(self) -> int:
        return self.r.size

    @property
    def num_basis(self) -> int:
        return self.V.shape[1]

    @property
    def num_boundary_nodes(self) -> int:
        return 3 * self.num_edge_nodes

    @property
    def boundary_slice(self) -> slice:
        return slice(0, self.num_boundary_nodes)

    @property
    def face_slices(self) -> Tuple[slice, slice, slice]:
        nfp = self.num_edge_nodes
        return (
            slice(0, nfp),
            slice(nfp, 2 * nfp),
            slice(2 * nfp, 3 * nfp),
        )

    @property
    def edge_slices(self) -> Tuple[slice, slice, slice]:
        """Backward-compatible volume-node slices for Table 1 only."""
        if self.rule != "table1":
            raise ValueError(
                "rule='table2' has independent face nodes. "
                "Use engine.face_values(u_nodes, edge_id)."
            )
        return self.face_slices

    def get_modal_coeffs(self, u_nodes: np.ndarray) -> np.ndarray:
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
        return self.V @ coeffs

    def face_values(self, u_nodes: np.ndarray, edge_id: int) -> np.ndarray:
        """Extract/interpolate volume nodal values onto one face."""
        u_nodes = np.asarray(u_nodes)
        if edge_id not in (0, 1, 2):
            raise ValueError("edge_id must be 0, 1, or 2.")
        if u_nodes.shape[0] != self.num_nodes:
            raise ValueError(
                f"Expected u_nodes first dimension {self.num_nodes}, "
                f"got {u_nodes.shape[0]}."
            )
        return self.R_faces[edge_id] @ u_nodes

    def boundary_values(self, u_nodes: np.ndarray) -> np.ndarray:
        """Map volume nodal values to the concatenated three-face vector."""
        u_nodes = np.asarray(u_nodes)
        if u_nodes.shape[0] != self.num_nodes:
            raise ValueError(
                f"Expected u_nodes first dimension {self.num_nodes}, "
                f"got {u_nodes.shape[0]}."
            )
        return self.R_all @ u_nodes

    def boundary_weight_diag(
        self,
        edge_lengths: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if self.w_e is None:
            raise ValueError("Boundary quadrature weights w_e are missing.")
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
        """Lift face-quadrature penalties to the volume nodal representation."""
        p_boundary = np.asarray(p_boundary)
        nb = self.num_boundary_nodes
        if p_boundary.shape[0] != nb:
            raise ValueError(
                f"Expected p_boundary first dimension {nb}, "
                f"got {p_boundary.shape[0]}."
            )

        wb_diag = self.boundary_weight_diag(edge_lengths=edge_lengths)
        if p_boundary.ndim == 1:
            weighted_face = wb_diag * p_boundary
        elif p_boundary.ndim == 2:
            weighted_face = wb_diag[:, None] * p_boundary
        else:
            raise ValueError("p_boundary must be a 1D or 2D array.")

        # General modal lift V M^{-1} V_f^T W_b p.
        # For Table 1, R_all is the original E=[I|0] extraction.
        v_face = self.R_all @ self.V
        return self.V @ (self.invM_modal @ (v_face.T @ weighted_face))


def build_local_operators(
    N: int,
    n: int,
    rule: str = "table1",
) -> ReferenceElement:
    """Build Table 1 or Table 2 local operators on the reference triangle."""
    if N < 1:
        raise ValueError("Polynomial degree N must be >= 1.")

    if rule == "table1":
        if n < N:
            warnings.warn(
                f"Table1 quadrature has degree 2n-1. Current n={n}, N={N}. "
                "This may be under-integrated."
            )
        if n < N + 1:
            warnings.warn(
                f"For exact modal mass with degree N={N}, Table1 would need "
                f"n >= N+1. Current n={n}."
            )
    elif rule == "table2":
        if n < N:
            warnings.warn(
                f"Table2 quadrature has degree 2n. For degree N={N}, "
                f"usually need n >= N."
            )
    else:
        raise ValueError("rule must be 'table1' or 'table2'.")

    nodes = get_reference_nodes(n, rule=rule)
    xi = nodes["xi"]
    eta = nodes["eta"]
    r = nodes["r"]
    s = nodes["s"]
    w_s = nodes["w_s"]
    w_e = nodes["w_e"]
    num_edge_nodes = nodes["num_edge_nodes"]

    area = 2.0
    edge_lengths = np.array([2.0, 2.0 * np.sqrt(2.0), 2.0])

    V, Vr, Vs = build_vandermonde2d(N, r, s)
    num_nodes = r.size
    num_basis = V.shape[1]
    if num_nodes < num_basis:
        raise ValueError(
            f"num_nodes={num_nodes} is smaller than num_basis={num_basis}."
        )
    rank = np.linalg.matrix_rank(V)
    if rank < num_basis:
        raise ValueError(
            f"Vandermonde matrix is rank deficient: rank={rank}, "
            f"num_basis={num_basis}."
        )

    W = np.diag(w_s)
    M_modal = area * (V.T @ W @ V)
    invM_modal = np.linalg.inv(M_modal)
    P = invM_modal @ (area * V.T @ W)

    if rule == "table1":
        volume_face_slices = (
            slice(0, num_edge_nodes),
            slice(num_edge_nodes, 2 * num_edge_nodes),
            slice(2 * num_edge_nodes, 3 * num_edge_nodes),
        )
        face_r = np.vstack([r[sl] for sl in volume_face_slices])
        face_s = np.vstack([s[sl] for sl in volume_face_slices])
        face_w = w_e
        r_faces = []
        for f in range(3):
            rf = np.zeros((num_edge_nodes, num_nodes))
            start = f * num_edge_nodes
            rf[np.arange(num_edge_nodes), start + np.arange(num_edge_nodes)] = 1.0
            r_faces.append(rf)
    else:
        face = get_face_quadrature(N)
        face_r = face["r_face"]
        face_s = face["s_face"]
        face_w = face["w_e"]
        w_e = face_w
        num_edge_nodes = N + 1
        r_faces = []
        for f in range(3):
            V_face, _, _ = build_vandermonde2d(N, face_r[f], face_s[f])
            r_faces.append(V_face @ P)

    R_faces = tuple(r_faces)
    R_all = np.vstack(R_faces)

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
        face_r=face_r,
        face_s=face_s,
        face_w=face_w,
        R_faces=R_faces,
        R_all=R_all,
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
    return rhs_Jq / J
