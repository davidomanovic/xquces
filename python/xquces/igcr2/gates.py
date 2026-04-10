"""Gate-level application of the irreducible GCR-2 ansatz.

The iGCR-2 state is:
    |ψ⟩ = U_L(κ) · exp[i J_i(β, γ)] · U_R(Z) · |Φ₀⟩

All three pieces are applied as direct Rust-backed gates with no runtime
gauge maps or redundant-to-reduced conversions.
"""

from __future__ import annotations

import numpy as np

from xquces._lib import apply_igcr2_diag_in_place
from xquces.basis import flatten_state, occ_rows, reshape_state
from xquces.orbitals import apply_orbital_rotation


def apply_igcr2_diagonal(
    vec: np.ndarray,
    *,
    double_params: np.ndarray,
    pair_params: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    copy: bool = True,
) -> np.ndarray:
    """Apply the irreducible diagonal correlator exp[i J_i(β, γ)].

    Parameters
    ----------
    vec : statevector, shape (dim,)
    double_params : β_p for p = 1, ..., norb-1.  Shape (norb - 1,).
        Orbital 0 is the reference (D_0 eliminated).
    pair_params : γ_{pq} for 0 ≤ p < q.  Shape (norb, norb), symmetric.
        Only the upper triangle is read.
    norb, nelec : system size
    copy : if True, operate on a copy
    """
    arr = np.array(vec, dtype=np.complex128, copy=copy)

    double_params = np.asarray(double_params, dtype=np.float64)
    pair_params = np.asarray(pair_params, dtype=np.float64)

    if double_params.shape != (norb - 1,):
        raise ValueError(f"double_params must have shape ({norb - 1},), got {double_params.shape}")
    if pair_params.shape != (norb, norb):
        raise ValueError(f"pair_params must have shape ({norb}, {norb})")

    # Pre-exponentiate
    double_exp = np.exp(1j * double_params)

    pair_exp = np.ones((norb, norb), dtype=np.complex128)
    for p in range(norb):
        for q in range(p + 1, norb):
            pair_exp[p, q] = np.exp(1j * pair_params[p, q])

    state2 = reshape_state(arr, norb, nelec)
    occ_a = occ_rows(norb, nelec[0])
    occ_b = occ_rows(norb, nelec[1])

    apply_igcr2_diag_in_place(
        state2,
        double_exp,
        pair_exp,
        norb,
        occ_a,
        occ_b,
    )

    return flatten_state(state2)


def apply_igcr2(
    vec: np.ndarray,
    *,
    double_params: np.ndarray,
    pair_params: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    left_orbital_rotation: np.ndarray | None = None,
    right_orbital_rotation: np.ndarray | None = None,
    copy: bool = True,
) -> np.ndarray:
    """Apply the full iGCR-2 ansatz: U_L · exp[iJ_i] · U_R · |vec⟩.

    Parameters
    ----------
    double_params : β_p, shape (norb - 1,)
    pair_params : γ_{pq}, shape (norb, norb), symmetric upper tri
    left_orbital_rotation : U_L unitary, shape (norb, norb) or None
    right_orbital_rotation : U_R unitary, shape (norb, norb) or None
    """
    arr = np.array(vec, dtype=np.complex128, copy=copy)

    # Step 1: right OV rotation
    if right_orbital_rotation is not None:
        arr = apply_orbital_rotation(
            arr,
            right_orbital_rotation,
            norb=norb,
            nelec=nelec,
            copy=False,
        )

    # Step 2: irreducible diagonal correlator
    arr = apply_igcr2_diagonal(
        arr,
        double_params=double_params,
        pair_params=pair_params,
        norb=norb,
        nelec=nelec,
        copy=False,
    )

    # Step 3: left orbital rotation
    if left_orbital_rotation is not None:
        arr = apply_orbital_rotation(
            arr,
            left_orbital_rotation,
            norb=norb,
            nelec=nelec,
            copy=False,
        )

    return arr