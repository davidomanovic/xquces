from __future__ import annotations

import numpy as np

from xquces._lib import apply_igcr2_spin_restricted_in_place_num_rep
from xquces._lib import apply_ucj_spin_balanced_in_place_num_rep
from xquces._lib import apply_ucj_spin_restricted_in_place_num_rep
from xquces.basis import flatten_state, occ_indicator_rows, occ_rows, reshape_state
from xquces.orbitals import apply_orbital_rotation


def apply_ucj_spin_restricted(
    vec: np.ndarray,
    double_params: np.ndarray,
    pair_params: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    time: float = 1.0,
    orbital_rotation: np.ndarray | None = None,
    copy: bool = True,
) -> np.ndarray:
    arr = np.array(vec, dtype=np.complex128, copy=copy)
    if orbital_rotation is not None:
        arr = apply_orbital_rotation(
            arr,
            orbital_rotation.conj().T,
            norb=norb,
            nelec=nelec,
            copy=False,
        )

    double_params = np.asarray(double_params, dtype=np.float64)
    pair_params = np.asarray(pair_params, dtype=np.float64)

    if double_params.shape != (norb,):
        raise ValueError("double_params must have shape (norb,)")
    if pair_params.shape != (norb, norb):
        raise ValueError("pair_params must have shape (norb, norb)")
    if not np.allclose(pair_params, pair_params.T):
        raise ValueError("pair_params must be symmetric")

    pair_upper = np.zeros((norb, norb), dtype=np.complex128)
    for p in range(norb):
        for q in range(p + 1, norb):
            pair_upper[p, q] = np.exp(1j * time * pair_params[p, q])

    double_exp = np.exp(1j * time * double_params)

    state2 = reshape_state(arr, norb, nelec)
    occ_a = occ_rows(norb, nelec[0])
    occ_b = occ_rows(norb, nelec[1])

    apply_ucj_spin_restricted_in_place_num_rep(
        state2,
        double_exp,
        pair_upper,
        norb,
        occ_a,
        occ_b,
    )

    out = flatten_state(state2)
    if orbital_rotation is not None:
        out = apply_orbital_rotation(
            out,
            orbital_rotation,
            norb=norb,
            nelec=nelec,
            copy=False,
        )
    return out


def apply_ucj_spin_balanced(
    vec: np.ndarray,
    same_spin_params: np.ndarray,
    mixed_spin_params: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    time: float = 1.0,
    orbital_rotation: np.ndarray | None = None,
    copy: bool = True,
) -> np.ndarray:
    arr = np.array(vec, dtype=np.complex128, copy=copy)
    if orbital_rotation is not None:
        arr = apply_orbital_rotation(
            arr,
            orbital_rotation.conj().T,
            norb=norb,
            nelec=nelec,
            copy=False,
        )

    same_spin_params = np.asarray(same_spin_params, dtype=np.float64)
    mixed_spin_params = np.asarray(mixed_spin_params, dtype=np.float64)

    if same_spin_params.shape != (norb, norb):
        raise ValueError("same_spin_params must have shape (norb, norb)")
    if mixed_spin_params.shape != (norb, norb):
        raise ValueError("mixed_spin_params must have shape (norb, norb)")
    if not np.allclose(same_spin_params, same_spin_params.T):
        raise ValueError("same_spin_params must be symmetric")
    if not np.allclose(mixed_spin_params, mixed_spin_params.T):
        raise ValueError("mixed_spin_params must be symmetric")

    same_pair_exp = np.ones((norb, norb), dtype=np.complex128)
    mixed_pair_exp = np.ones((norb, norb), dtype=np.complex128)
    for p in range(norb):
        for q in range(p + 1, norb):
            same_pair_exp[p, q] = np.exp(1j * time * same_spin_params[p, q])
            mixed_pair_exp[p, q] = np.exp(1j * time * mixed_spin_params[p, q])

    same_diag_exp = np.exp(0.5j * time * np.diag(same_spin_params))
    mixed_diag_exp = np.exp(1j * time * np.diag(mixed_spin_params))

    state2 = reshape_state(arr, norb, nelec)
    occ_a = occ_rows(norb, nelec[0])
    occ_b = occ_rows(norb, nelec[1])

    apply_ucj_spin_balanced_in_place_num_rep(
        state2,
        same_pair_exp,
        mixed_pair_exp,
        same_diag_exp,
        mixed_diag_exp,
        norb,
        occ_a,
        occ_b,
    )

    out = flatten_state(state2)
    if orbital_rotation is not None:
        out = apply_orbital_rotation(
            out,
            orbital_rotation,
            norb=norb,
            nelec=nelec,
            copy=False,
        )
    return out


def apply_igcr2_spin_restricted(
    vec: np.ndarray,
    pair_params: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    time: float = 1.0,
    left_orbital_rotation: np.ndarray | None = None,
    right_orbital_rotation: np.ndarray | None = None,
    copy: bool = True,
) -> np.ndarray:
    arr = np.array(vec, dtype=np.complex128, copy=copy)
    if right_orbital_rotation is not None:
        arr = apply_orbital_rotation(
            arr, right_orbital_rotation, norb=norb, nelec=nelec, copy=False
        )

    pair_params = np.asarray(pair_params, dtype=np.float64) * time

    state2 = reshape_state(arr, norb, nelec)
    occ_a = occ_indicator_rows(norb, nelec[0])
    occ_b = occ_indicator_rows(norb, nelec[1])
    apply_igcr2_spin_restricted_in_place_num_rep(
        state2, pair_params, norb, occ_a, occ_b
    )
    arr = flatten_state(state2)

    if left_orbital_rotation is not None:
        arr = apply_orbital_rotation(
            arr, left_orbital_rotation, norb=norb, nelec=nelec, copy=False
        )
    return arr


def apply_gcr_spin_restricted(
    vec: np.ndarray,
    double_params: np.ndarray,
    pair_params: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    time: float = 1.0,
    left_orbital_rotation: np.ndarray | None = None,
    right_orbital_rotation: np.ndarray | None = None,
    copy: bool = True,
) -> np.ndarray:
    arr = np.array(vec, dtype=np.complex128, copy=copy)
    if right_orbital_rotation is not None:
        arr = apply_orbital_rotation(
            arr,
            right_orbital_rotation,
            norb=norb,
            nelec=nelec,
            copy=False,
        )
    arr = apply_ucj_spin_restricted(
        arr,
        double_params=double_params,
        pair_params=pair_params,
        norb=norb,
        nelec=nelec,
        time=time,
        orbital_rotation=None,
        copy=False,
    )
    if left_orbital_rotation is not None:
        arr = apply_orbital_rotation(
            arr,
            left_orbital_rotation,
            norb=norb,
            nelec=nelec,
            copy=False,
        )
    return arr


def apply_gcr_spin_balanced(
    vec: np.ndarray,
    same_spin_params: np.ndarray,
    mixed_spin_params: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    time: float = 1.0,
    left_orbital_rotation: np.ndarray | None = None,
    right_orbital_rotation: np.ndarray | None = None,
    copy: bool = True,
) -> np.ndarray:
    arr = np.array(vec, dtype=np.complex128, copy=copy)
    if right_orbital_rotation is not None:
        arr = apply_orbital_rotation(
            arr,
            right_orbital_rotation,
            norb=norb,
            nelec=nelec,
            copy=False,
        )
    arr = apply_ucj_spin_balanced(
        arr,
        same_spin_params=same_spin_params,
        mixed_spin_params=mixed_spin_params,
        norb=norb,
        nelec=nelec,
        time=time,
        orbital_rotation=None,
        copy=False,
    )
    if left_orbital_rotation is not None:
        arr = apply_orbital_rotation(
            arr,
            left_orbital_rotation,
            norb=norb,
            nelec=nelec,
            copy=False,
        )
    return arr
