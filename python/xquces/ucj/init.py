from __future__ import annotations

import numpy as np

from xquces.orbitals import ov_params_from_t1
from xquces.ucj.params import ansatz_from_parameters_spin_restricted, n_layer_params_spin_restricted


def pair_params_from_t2(t2: np.ndarray) -> np.ndarray:
    t2 = np.asarray(t2, dtype=np.float64)
    if t2.ndim != 4:
        raise ValueError("t2 must have shape (nocc, nocc, nvirt, nvirt)")
    nocc1, nocc2, nvirt1, nvirt2 = t2.shape
    if nocc1 != nocc2 or nvirt1 != nvirt2:
        raise ValueError("t2 must have shape (nocc, nocc, nvirt, nvirt)")
    nocc = nocc1
    nvirt = nvirt1
    norb = nocc + nvirt
    pair = np.zeros((norb, norb), dtype=np.float64)

    for i in range(nocc):
        for j in range(i + 1, nocc):
            v = np.linalg.norm(t2[i, j])
            pair[i, j] = v
            pair[j, i] = v

    for a in range(nvirt):
        for b in range(a + 1, nvirt):
            v = np.linalg.norm(t2[:, :, a, b])
            pair[nocc + a, nocc + b] = v
            pair[nocc + b, nocc + a] = v

    for i in range(nocc):
        for a in range(nvirt):
            v = np.linalg.norm(t2[i, :, a, :])
            pair[i, nocc + a] = v
            pair[nocc + a, i] = v

    np.fill_diagonal(pair, 0.0)
    mx = np.max(np.abs(pair))
    if mx > 0:
        pair /= mx
    return pair


def default_ov_kick(nocc: int, nvirt: int, magnitude: float = 0.1) -> np.ndarray:
    out = np.zeros((2 * nocc * nvirt,), dtype=np.float64)
    if nocc == 0 or nvirt == 0 or magnitude == 0.0:
        return out
    out[0] = magnitude
    return out


def ucj_seed_parameters(
    t2: np.ndarray,
    *,
    t1: np.ndarray | None = None,
    n_layers: int = 1,
    pair_scale: float = 1.0,
    ov_scale: float = 1.0,
    ov_kick: float = 0.1,
) -> np.ndarray:
    t2 = np.asarray(t2, dtype=np.float64)
    if t2.ndim != 4:
        raise ValueError("t2 must have shape (nocc, nocc, nvirt, nvirt)")
    nocc, _, nvirt, _ = t2.shape
    norb = nocc + nvirt

    pair = pair_scale * pair_params_from_t2(t2)
    pair_vals = np.asarray(
        [pair[a, b] for a in range(norb) for b in range(a + 1, norb)],
        dtype=np.float64,
    )

    if t1 is None:
        ov_params = default_ov_kick(nocc, nvirt, magnitude=ov_kick)
    else:
        t1 = np.asarray(t1, dtype=np.complex128)
        ov_params = ov_scale * ov_params_from_t1(t1)
        if np.max(np.abs(ov_params)) < 1e-12:
            ov_params = default_ov_kick(nocc, nvirt, magnitude=ov_kick)

    npl = n_layer_params_spin_restricted(norb, nocc)
    x0 = np.zeros(n_layers * npl, dtype=np.float64)

    for ell in range(n_layers):
        i = ell * npl
        x0[i : i + norb] = 0.0
        i += norb
        x0[i : i + len(pair_vals)] = pair_vals if ell == 0 else 0.0
        i += len(pair_vals)
        x0[i : i + len(ov_params)] = ov_params if ell == 0 else 0.0

    return x0


def ucj_from_t_amplitudes(
    t2: np.ndarray,
    *,
    t1: np.ndarray | None = None,
    n_layers: int = 1,
    pair_scale: float = 1.0,
    ov_scale: float = 1.0,
    ov_kick: float = 0.1,
):
    t2 = np.asarray(t2, dtype=np.float64)
    nocc = t2.shape[0]
    norb = t2.shape[0] + t2.shape[2]
    x0 = ucj_seed_parameters(
        t2,
        t1=t1,
        n_layers=n_layers,
        pair_scale=pair_scale,
        ov_scale=ov_scale,
        ov_kick=ov_kick,
    )
    return ansatz_from_parameters_spin_restricted(
        x0,
        norb=norb,
        nocc=nocc,
        n_layers=n_layers,
    )