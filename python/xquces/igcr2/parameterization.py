"""Irreducible GCR-2 parameterization.

Flat parameter layout:
    [left_rotation | double_params | pair_params | right_rotation]

    left_rotation:  norb*(norb-1) real params  (θ, φ per Givens pair, all p<q)
    double_params:  norb - 1 real params        (β_p, p=1..norb-1)
    pair_params:    norb*(norb-1)/2 real params  (γ_{pq}, 0 ≤ p < q)
    right_rotation: 2*nocc*nvirt real params     (θ, φ per OV Givens pair)

Total = norb*(norb-1) + (norb-1) + norb*(norb-1)/2 + 2*nocc*nvirt
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from xquces.igcr2.gates import apply_igcr2
from xquces.orbitals import (
    givens_decomposition,
)


# ---------------------------------------------------------------------------
# Givens ↔ flat-parameter helpers
# ---------------------------------------------------------------------------

def _full_givens_params_from_unitary(mat: np.ndarray) -> np.ndarray:
    """Extract (θ, φ) Givens parameters from a general unitary.

    Returns a flat array of length norb*(norb-1) = 2 * C(norb, 2).
    Layout: for each pair (p, q) with p < q in lexicographic order,
    two consecutive entries [θ_{pq}, φ_{pq}].

    The Givens decomposition produces a product of adjacent-pair rotations
    plus diagonal phases.  Here we re-synthesize into the (p,q)-indexed
    Givens network used by the circuit form, by computing
        κ = logm(mat)  (anti-Hermitian generator, zero diagonal)
    and reading off the real parameters.
    """
    mat = np.asarray(mat, dtype=np.complex128)
    norb = mat.shape[0]
    # Use matrix log to get anti-Hermitian generator
    from scipy.linalg import logm

    K = logm(mat)
    # Project to anti-Hermitian
    K = 0.5 * (K - K.conj().T)
    # Zero diagonal (gauge-fixed)
    np.fill_diagonal(K, 0.0)

    params = []
    for p in range(norb):
        for q in range(p + 1, norb):
            # κ_{pq} = Re + i*Im  →  θ = |κ_{pq}|, φ = arg(κ_{pq})
            # But for the Givens form: κ_{pq} corresponds to
            #   θ_{pq} (a†_p a_q - a†_q a_p) rotated by phase φ_{pq}
            kpq = K[p, q]
            theta = abs(kpq)
            phi = np.angle(kpq) if abs(kpq) > 1e-14 else 0.0
            params.extend([theta, phi])
    return np.array(params, dtype=np.float64)


def _unitary_from_full_givens_params(params: np.ndarray, norb: int) -> np.ndarray:
    """Reconstruct a unitary from the (θ, φ) Givens parameterization.

    Inverse of _full_givens_params_from_unitary.
    """
    K = np.zeros((norb, norb), dtype=np.complex128)
    idx = 0
    for p in range(norb):
        for q in range(p + 1, norb):
            theta = params[idx]
            phi = params[idx + 1]
            kpq = theta * np.exp(1j * phi)
            K[p, q] = kpq
            K[q, p] = -np.conj(kpq)
            idx += 2
    from scipy.linalg import expm

    return expm(K)


def _ov_givens_params_from_unitary(
    mat: np.ndarray, nocc: int, norb: int
) -> np.ndarray:
    """Extract OV Givens parameters from a unitary that is close to identity
    in the OO and VV blocks.

    Returns flat array of length 2 * nocc * nvirt.
    Layout: for each pair (i, a) with i ∈ occ, a ∈ virt, two entries [θ, φ].
    """
    from scipy.linalg import logm

    nvirt = norb - nocc
    K = logm(mat)
    K = 0.5 * (K - K.conj().T)

    params = []
    for i in range(nocc):
        for a in range(nocc, norb):
            kia = K[i, a]
            theta = abs(kia)
            phi = np.angle(kia) if abs(kia) > 1e-14 else 0.0
            params.extend([theta, phi])
    return np.array(params, dtype=np.float64)


def _unitary_from_ov_givens_params(
    params: np.ndarray, nocc: int, norb: int
) -> np.ndarray:
    """Reconstruct a unitary from OV-only Givens parameters."""
    nvirt = norb - nocc
    K = np.zeros((norb, norb), dtype=np.complex128)
    idx = 0
    for i in range(nocc):
        for a in range(nocc, norb):
            theta = params[idx]
            phi = params[idx + 1]
            kia = theta * np.exp(1j * phi)
            K[i, a] = kia
            K[a, i] = -np.conj(kia)
            idx += 2
    from scipy.linalg import expm

    return expm(K)


# ---------------------------------------------------------------------------
# Parameterization dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class IGCR2Parameterization:
    """Irreducible single-layer GCR-2 parameterization.

    No runtime gauge maps.  The flat parameter vector directly encodes
    the irreducible operators.
    """

    norb: int
    nocc: int

    @property
    def nvirt(self) -> int:
        return self.norb - self.nocc

    # --- sizes ---

    @property
    def n_left_params(self) -> int:
        """2 * C(norb, 2) = norb*(norb-1)."""
        return self.norb * (self.norb - 1)

    @property
    def n_double_params(self) -> int:
        """norb - 1  (orbital 0 is the reference)."""
        return self.norb - 1

    @property
    def n_pair_params(self) -> int:
        """C(norb, 2) = norb*(norb-1)/2."""
        return self.norb * (self.norb - 1) // 2

    @property
    def n_right_params(self) -> int:
        """2 * nocc * nvirt."""
        return 2 * self.nocc * self.nvirt

    @property
    def n_params(self) -> int:
        return (
            self.n_left_params
            + self.n_double_params
            + self.n_pair_params
            + self.n_right_params
        )

    # --- unpack ---

    def unpack(self, params: np.ndarray) -> dict:
        """Unpack flat vector into named arrays.

        Returns dict with keys:
            left_rotation:  np.ndarray (norb, norb) unitary
            double_params:  np.ndarray (norb - 1,)
            pair_params:    np.ndarray (norb, norb) symmetric
            right_rotation: np.ndarray (norb, norb) unitary
        """
        params = np.asarray(params, dtype=np.float64)
        if params.shape != (self.n_params,):
            raise ValueError(
                f"Expected {self.n_params} params, got {params.shape}"
            )

        idx = 0
        norb = self.norb

        # Left rotation
        n = self.n_left_params
        left = _unitary_from_full_givens_params(params[idx : idx + n], norb)
        idx += n

        # Double-occupancy params
        n = self.n_double_params
        double_params = params[idx : idx + n].copy()
        idx += n

        # Pair params → symmetric matrix
        n = self.n_pair_params
        raw_pairs = params[idx : idx + n]
        pair_mat = np.zeros((norb, norb), dtype=np.float64)
        k = 0
        for p in range(norb):
            for q in range(p + 1, norb):
                pair_mat[p, q] = raw_pairs[k]
                pair_mat[q, p] = raw_pairs[k]
                k += 1
        idx += n

        # Right rotation
        n = self.n_right_params
        right = _unitary_from_ov_givens_params(
            params[idx : idx + n], self.nocc, norb
        )

        return {
            "left_rotation": left,
            "double_params": double_params,
            "pair_params": pair_mat,
            "right_rotation": right,
        }

    # --- pack ---

    def pack(
        self,
        *,
        left_rotation: np.ndarray,
        double_params: np.ndarray,
        pair_params: np.ndarray,
        right_rotation: np.ndarray,
    ) -> np.ndarray:
        """Pack named arrays into a flat parameter vector."""
        out = np.zeros(self.n_params, dtype=np.float64)
        idx = 0

        n = self.n_left_params
        out[idx : idx + n] = _full_givens_params_from_unitary(left_rotation)
        idx += n

        n = self.n_double_params
        out[idx : idx + n] = np.asarray(double_params, dtype=np.float64)
        idx += n

        n = self.n_pair_params
        norb = self.norb
        k = 0
        for p in range(norb):
            for q in range(p + 1, norb):
                out[idx + k] = pair_params[p, q]
                k += 1
        idx += n

        n = self.n_right_params
        out[idx : idx + n] = _ov_givens_params_from_unitary(
            right_rotation, self.nocc, self.norb
        )

        return out

    # --- apply ---

    def apply(
        self,
        params: np.ndarray,
        vec: np.ndarray,
        nelec: tuple[int, int],
    ) -> np.ndarray:
        """Apply iGCR-2 with given flat parameter vector to a statevector."""
        d = self.unpack(params)
        return apply_igcr2(
            vec,
            double_params=d["double_params"],
            pair_params=d["pair_params"],
            norb=self.norb,
            nelec=nelec,
            left_orbital_rotation=d["left_rotation"],
            right_orbital_rotation=d["right_rotation"],
            copy=True,
        )

    def params_to_vec(
        self,
        reference_vec: np.ndarray,
        nelec: tuple[int, int],
    ) -> Callable[[np.ndarray], np.ndarray]:
        """Return a closure params → statevector for use in optimizers."""
        ref = np.asarray(reference_vec, dtype=np.complex128)

        def func(params: np.ndarray) -> np.ndarray:
            return self.apply(params, ref, nelec)

        return func


# ---------------------------------------------------------------------------
# Seed conversion from redundant UCJ
# ---------------------------------------------------------------------------

def igcr2_params_from_ucj(
    ucj_ansatz,  # UCJAnsatz with n_layers=1
    nocc: int,
    parameterization: IGCR2Parameterization | None = None,
) -> np.ndarray:
    """Convert a single-layer spin-restricted UCJ ansatz to iGCR-2 parameters.

    The mapping is:
        U_L = U_F · U,    U_R = U†
        J_i  from algebraic reduction of J

    Parameters
    ----------
    ucj_ansatz : UCJAnsatz (single layer, spin-restricted or spin-balanced)
    nocc : number of occupied orbitals
    parameterization : optional pre-built IGCR2Parameterization

    Returns
    -------
    flat parameter vector for IGCR2Parameterization
    """
    if ucj_ansatz.n_layers != 1:
        raise ValueError("Only single-layer UCJ can be mapped to iGCR-2")

    layer = ucj_ansatz.layers[0]
    norb = ucj_ansatz.norb

    if parameterization is None:
        parameterization = IGCR2Parameterization(norb=norb, nocc=nocc)

    # --- Orbital rotations ---
    U = np.asarray(layer.orbital_rotation, dtype=np.complex128)

    # Left = U_F · U
    if ucj_ansatz.final_orbital_rotation is not None:
        U_F = np.asarray(ucj_ansatz.final_orbital_rotation, dtype=np.complex128)
        M_L = U_F @ U
    else:
        M_L = U.copy()

    # Right = U†
    M_R = U.conj().T

    # --- Diagonal Jastrow reduction ---
    # Extract the redundant diagonal parameters
    diag = layer.diagonal

    # Handle both spin-restricted and spin-balanced
    if hasattr(diag, "double_params"):
        # SpinRestrictedSpec: already has b_p and c_{pq}
        b = np.asarray(diag.double_params, dtype=np.float64)  # shape (norb,)
        c = np.asarray(diag.pair_params, dtype=np.float64)    # shape (norb, norb)
    elif hasattr(diag, "same_spin_params"):
        # SpinBalancedSpec: derive spin-restricted form
        # J_restricted_pq = (J^{αα}_{pq} + J^{αβ}_{pq}) / 2
        same = np.asarray(diag.same_spin_params, dtype=np.float64)
        mixed = np.asarray(diag.mixed_spin_params, dtype=np.float64)
        # b_p = diagonal of mixed (on-site αβ)
        b = np.diag(mixed).copy()
        # c_{pq} = same[p,q] + mixed[p,q] for p != q  (approximation for spin-restricted)
        c = np.zeros((norb, norb), dtype=np.float64)
        for p in range(norb):
            for q in range(p + 1, norb):
                c[p, q] = same[p, q] + mixed[p, q]
                c[q, p] = c[p, q]
    else:
        raise TypeError(f"Unrecognized diagonal spec type: {type(diag)}")

    # Remove fixed-N_e redundancy: β_p = b_p - b_0 for p = 1..norb-1
    beta = b[1:] - b[0]

    # Pair params: just read off upper triangle of c
    gamma = c.copy()

    return parameterization.pack(
        left_rotation=M_L,
        double_params=beta,
        pair_params=gamma,
        right_rotation=M_R,
    )