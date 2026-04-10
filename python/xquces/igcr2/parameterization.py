"""Irreducible single-layer GCR-2 parameterization (spin-restricted, 121 params).

Flat parameter layout:
    [left_rotation | beta | gamma | right_rotation]

    left_rotation:  norb*(norb-1)  — (Re, Im) of off-diagonal anti-Hermitian κ_{pq}
    beta:           norb - 1       — D_p for p=1..norb-1  (D_0 eliminated)
    gamma:          C(norb, 2)     — γ_{pq} for N_p N_q,  0 ≤ p < q
    right_rotation: 2·nocc·nvirt   — (Re, Im) of OV anti-Hermitian Z_{ai}

Total = norb*(norb-1) + (norb-1) + C(norb,2) + 2·nocc·nvirt
      = 56 + 7 + 28 + 30 = 121   for norb=8, nocc=5

Forward map (every LM iteration):
    1.  Build K_L from (Re, Im) params → U_L = expm(K_L)         [no singularity]
    2.  Build double_params / pair_params from β, γ               [trivial array fill]
    3.  Build K_R from (Re, Im) OV params → U_R = expm(K_R)
    4.  Apply  U_R → exp(iJ) → U_L   via existing Rust kernel

No gauge maps, no SVD, no coupling.  The V @ x machinery is not needed here
because we explicitly parameterize the irreducible generators.

Seed conversion from spin-balanced UCJ:
    γ_{pq} = (J^{αα}_{pq} + J^{αβ}_{pq}) / 2       [correct projection]
    β_p    = J^{αβ}_{pp} - J^{αβ}_{00}               [D_0 eliminated]
    U_L    = logm(U_F · U), zero diagonal, read off   [discard diagonal phases]
    U_R    = logm(U†), extract OV block

The seed is approximate (spin-balanced → spin-restricted loses (s-m)/2 · (S-M)
and discards diagonal orbital phases), but is on the correct energy scale.
The optimizer recovers the rest.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import scipy.linalg

from xquces.gates import apply_gcr_spin_restricted
from xquces.gcr.model import gcr_from_ucj_ansatz
from xquces.ucj.model import UCJAnsatz


# ---------------------------------------------------------------------------
# (Re, Im) anti-Hermitian charts — no polar singularity
# ---------------------------------------------------------------------------

def _left_unitary_from_params(params: np.ndarray, norb: int) -> np.ndarray:
    """Off-diagonal anti-Hermitian (Re, Im) → unitary.  κ_{pp} = 0."""
    K = np.zeros((norb, norb), dtype=np.complex128)
    idx = 0
    for p in range(norb):
        for q in range(p + 1, norb):
            z = params[idx] + 1j * params[idx + 1]
            K[p, q] = z
            K[q, p] = -np.conj(z)
            idx += 2
    return scipy.linalg.expm(K)


def _left_params_from_unitary(U: np.ndarray, norb: int) -> np.ndarray:
    """Unitary → off-diagonal anti-Hermitian (Re, Im).  Discards diagonal phases."""
    K = scipy.linalg.logm(U)
    K = 0.5 * (K - K.conj().T)          # project to anti-Hermitian
    np.fill_diagonal(K, 0.0)            # gauge fix: κ_{pp} = 0
    params = np.empty(norb * (norb - 1), dtype=np.float64)
    idx = 0
    for p in range(norb):
        for q in range(p + 1, norb):
            params[idx] = K[p, q].real
            params[idx + 1] = K[p, q].imag
            idx += 2
    return params


def _ov_unitary_from_params(params: np.ndarray, nocc: int, norb: int) -> np.ndarray:
    """OV anti-Hermitian (Re, Im) → unitary."""
    K = np.zeros((norb, norb), dtype=np.complex128)
    idx = 0
    for i in range(nocc):
        for a in range(nocc, norb):
            z = params[idx] + 1j * params[idx + 1]
            K[i, a] = z
            K[a, i] = -np.conj(z)
            idx += 2
    return scipy.linalg.expm(K)


def _ov_params_from_unitary(U: np.ndarray, nocc: int, norb: int) -> np.ndarray:
    """Unitary → OV anti-Hermitian (Re, Im).  Discards OO/VV blocks."""
    K = scipy.linalg.logm(U)
    K = 0.5 * (K - K.conj().T)
    params = np.empty(2 * nocc * (norb - nocc), dtype=np.float64)
    idx = 0
    for i in range(nocc):
        for a in range(nocc, norb):
            params[idx] = K[i, a].real
            params[idx + 1] = K[i, a].imag
            idx += 2
    return params


# ---------------------------------------------------------------------------
# Parameterization
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class IGCR2Parameterization:
    """Irreducible single-layer spin-restricted GCR-2."""

    norb: int
    nocc: int

    @property
    def nvirt(self) -> int:
        return self.norb - self.nocc

    # --- sizes ---

    @property
    def n_left_params(self) -> int:
        return self.norb * (self.norb - 1)

    @property
    def n_beta_params(self) -> int:
        return self.norb - 1

    @property
    def n_gamma_params(self) -> int:
        return self.norb * (self.norb - 1) // 2

    @property
    def n_right_params(self) -> int:
        return 2 * self.nocc * self.nvirt

    @property
    def n_params(self) -> int:
        return (
            self.n_left_params
            + self.n_beta_params
            + self.n_gamma_params
            + self.n_right_params
        )

    # --- unpack flat vector → operators ---

    def unpack(self, params: np.ndarray) -> dict:
        """Flat vector → {left_rotation, double_params, pair_params, right_rotation}.

        double_params has shape (norb,) with double_params[0] = 0.
        pair_params has shape (norb, norb), symmetric.
        """
        params = np.asarray(params, dtype=np.float64)
        if params.shape != (self.n_params,):
            raise ValueError(f"Expected {self.n_params} params, got {params.shape}")

        norb = self.norb
        idx = 0

        # Left rotation
        n = self.n_left_params
        left = _left_unitary_from_params(params[idx : idx + n], norb)
        idx += n

        # Beta → double_params (length norb, first entry 0)
        n = self.n_beta_params
        beta = params[idx : idx + n]
        double_params = np.zeros(norb, dtype=np.float64)
        double_params[1:] = beta
        idx += n

        # Gamma → pair_params (symmetric matrix)
        n = self.n_gamma_params
        raw = params[idx : idx + n]
        pair_params = np.zeros((norb, norb), dtype=np.float64)
        k = 0
        for p in range(norb):
            for q in range(p + 1, norb):
                pair_params[p, q] = raw[k]
                pair_params[q, p] = raw[k]
                k += 1
        idx += n

        # Right rotation
        n = self.n_right_params
        right = _ov_unitary_from_params(params[idx : idx + n], self.nocc, norb)

        return {
            "left_rotation": left,
            "double_params": double_params,
            "pair_params": pair_params,
            "right_rotation": right,
        }

    # --- pack operators → flat vector ---

    def pack(
        self,
        *,
        left_rotation: np.ndarray,
        double_params: np.ndarray,
        pair_params: np.ndarray,
        right_rotation: np.ndarray,
    ) -> np.ndarray:
        norb = self.norb
        out = np.zeros(self.n_params, dtype=np.float64)
        idx = 0

        n = self.n_left_params
        out[idx : idx + n] = _left_params_from_unitary(left_rotation, norb)
        idx += n

        n = self.n_beta_params
        out[idx : idx + n] = np.asarray(double_params, dtype=np.float64)[1:]
        idx += n

        n = self.n_gamma_params
        k = 0
        for p in range(norb):
            for q in range(p + 1, norb):
                out[idx + k] = pair_params[p, q]
                k += 1
        idx += n

        n = self.n_right_params
        out[idx : idx + n] = _ov_params_from_unitary(right_rotation, self.nocc, norb)

        return out

    # --- apply (forward map) ---

    def apply(
        self,
        params: np.ndarray,
        vec: np.ndarray,
        nelec: tuple[int, int],
    ) -> np.ndarray:
        d = self.unpack(params)
        return apply_gcr_spin_restricted(
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
        ref = np.asarray(reference_vec, dtype=np.complex128)

        def func(params: np.ndarray) -> np.ndarray:
            return self.apply(params, ref, nelec)

        return func


# ---------------------------------------------------------------------------
# Seed conversion from spin-balanced UCJ
# ---------------------------------------------------------------------------

def igcr2_params_from_ucj(
    ucj_ansatz: UCJAnsatz,
    nocc: int,
    parameterization: IGCR2Parameterization | None = None,
) -> np.ndarray:
    """Convert single-layer spin-balanced UCJ → 121-param iGCR-2.

    Orbital rotations:
        M_L = U_F · U      (left, diagonal phases discarded)
        M_R = U†            (right, OO/VV blocks discarded)

    Diagonal projection (spin-balanced → spin-restricted):
        γ_{pq} = (J^{αα}_{pq} + J^{αβ}_{pq}) / 2    for p < q
        β_p    = J^{αβ}_{pp} - J^{αβ}_{00}            for p ≥ 1

    The factor of 1/2 is correct because N_p N_q = S_{pq} + M_{pq},
    so γ multiplies BOTH same-spin and mixed-spin equally.  The best
    approximation to s·S + m·M is ((s+m)/2)·(S+M).

    The seed is approximate (residual error ~ (s-m)/2 · (S-M) plus
    discarded diagonal phases).  The optimizer recovers the rest.
    """
    if ucj_ansatz.n_layers != 1:
        raise ValueError("Only single-layer UCJ → iGCR-2")

    layer = ucj_ansatz.layers[0]
    norb = ucj_ansatz.norb

    if parameterization is None:
        parameterization = IGCR2Parameterization(norb=norb, nocc=nocc)

    # --- Orbital rotations ---
    U = np.asarray(layer.orbital_rotation, dtype=np.complex128)
    if ucj_ansatz.final_orbital_rotation is not None:
        U_F = np.asarray(ucj_ansatz.final_orbital_rotation, dtype=np.complex128)
        M_L = U_F @ U
    else:
        M_L = U.copy()
    M_R = U.conj().T

    # --- Diagonal Jastrow ---
    diag = layer.diagonal

    if hasattr(diag, "double_params"):
        # SpinRestrictedSpec: already in the right form
        b = np.asarray(diag.double_params, dtype=np.float64)
        c = np.asarray(diag.pair_params, dtype=np.float64)
    elif hasattr(diag, "same_spin_params"):
        # SpinBalancedSpec: project to spin-restricted
        same = np.asarray(diag.same_spin_params, dtype=np.float64)
        mixed = np.asarray(diag.mixed_spin_params, dtype=np.float64)

        # D_p from mixed-spin diagonal
        b = np.diag(mixed).copy()

        # γ_{pq} = (s + m) / 2  ← THIS IS THE FIX (was s + m before)
        c = np.zeros((norb, norb), dtype=np.float64)
        for p in range(norb):
            for q in range(p + 1, norb):
                c[p, q] = (same[p, q] + mixed[p, q]) / 2.0
                c[q, p] = c[p, q]
    else:
        raise TypeError(f"Unknown diagonal spec: {type(diag)}")

    # Irreducible: β_p = b_p - b_0,  γ_{pq} unchanged
    double_params = np.zeros(norb, dtype=np.float64)
    double_params[1:] = b[1:] - b[0]

    return parameterization.pack(
        left_rotation=M_L,
        double_params=double_params,
        pair_params=c,
        right_rotation=M_R,
    )