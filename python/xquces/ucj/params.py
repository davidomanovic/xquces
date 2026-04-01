from __future__ import annotations

import numpy as np

from xquces.orbitals import orbital_rotation_from_ov_params
from xquces.ucj.model import SpinRestrictedSpec, UCJAnsatz, UCJLayer


def n_pair_params(norb: int) -> int:
    return norb * (norb - 1) // 2


def n_ov_params(norb: int, nocc: int) -> int:
    return 2 * nocc * (norb - nocc)


def n_layer_params_spin_restricted(norb: int, nocc: int) -> int:
    return norb + n_pair_params(norb) + n_ov_params(norb, nocc)


def _pair_indices(norb: int) -> list[tuple[int, int]]:
    return [(p, q) for p in range(norb) for q in range(p + 1, norb)]


def ansatz_from_parameters_spin_restricted(
    params: np.ndarray,
    *,
    norb: int,
    nocc: int,
    n_layers: int,
) -> UCJAnsatz:
    params = np.asarray(params, dtype=np.float64)
    npl = n_layer_params_spin_restricted(norb, nocc)
    expected = n_layers * npl
    if params.shape != (expected,):
        raise ValueError(f"expected parameter vector of shape {(expected,)}")
    pairs = _pair_indices(norb)
    layers = []
    idx = 0
    for _ in range(n_layers):
        d = params[idx : idx + norb].copy()
        idx += norb
        p = np.zeros((norb, norb), dtype=np.float64)
        vals = params[idx : idx + len(pairs)]
        idx += len(pairs)
        for val, (a, b) in zip(vals, pairs):
            p[a, b] = val
            p[b, a] = val
        ov = params[idx : idx + n_ov_params(norb, nocc)]
        idx += n_ov_params(norb, nocc)
        u = orbital_rotation_from_ov_params(ov, norb=norb, nocc=nocc, gauge_fix=True)
        layers.append(
            UCJLayer(
                diagonal=SpinRestrictedSpec(double_params=d, pair_params=p),
                orbital_rotation=u,
            )
        )
    return UCJAnsatz(layers=tuple(layers))


def parameters_from_ansatz_spin_restricted(
    ansatz: UCJAnsatz,
    *,
    nocc: int,
) -> np.ndarray:
    norb = ansatz.norb
    pairs = _pair_indices(norb)
    chunks = []
    for layer in ansatz.layers:
        if not isinstance(layer.diagonal, SpinRestrictedSpec):
            raise TypeError("expected spin-restricted layers")
        chunks.append(np.asarray(layer.diagonal.double_params, dtype=np.float64))
        chunks.append(np.asarray([layer.diagonal.pair_params[a, b] for a, b in pairs], dtype=np.float64))
        chunks.append(np.zeros(n_ov_params(norb, nocc), dtype=np.float64))
    return np.concatenate(chunks)