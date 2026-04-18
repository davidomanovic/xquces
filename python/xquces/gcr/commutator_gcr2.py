from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from functools import cache
from typing import Callable

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import expm_multiply

from xquces.basis import flatten_state, occ_indicator_rows, occ_rows, reshape_state
from xquces.gcr.igcr2 import IGCR2Ansatz, IGCR2SpinRestrictedParameterization
from xquces.orbitals import apply_orbital_rotation
from xquces.ucj.model import UCJAnsatz

try:
    from xquces._lib import (
        apply_gcr2_pairhop_middle_cached_in_place_num_rep,
        apply_gcr2_pairhop_middle_in_place_num_rep,
    )
except ImportError:  # pragma: no cover - only used before rebuilding the Rust extension
    apply_gcr2_pairhop_middle_cached_in_place_num_rep = None
    apply_gcr2_pairhop_middle_in_place_num_rep = None


def _validate_pairs(
    pairs: list[tuple[int, int]] | None,
    norb: int,
) -> tuple[tuple[int, int], ...]:
    if pairs is None:
        return tuple(itertools.combinations(range(norb), 2))
    out = []
    seen = set()
    for p, q in pairs:
        p = int(p)
        q = int(q)
        if not (0 <= p < q < norb):
            raise ValueError("pairs must satisfy 0 <= p < q < norb")
        if (p, q) in seen:
            raise ValueError("pairs must not contain duplicates")
        seen.add((p, q))
        out.append((p, q))
    return tuple(out)


def _replace_orbital(
    occ: tuple[int, ...],
    old: int,
    new: int,
) -> tuple[tuple[int, ...], int] | None:
    if old not in occ or new in occ:
        return None
    pos_old = occ.index(old)
    sign = -1 if pos_old % 2 else 1
    after_annihilate = list(occ)
    after_annihilate.pop(pos_old)
    insert_at = int(np.searchsorted(after_annihilate, new))
    sign *= -1 if insert_at % 2 else 1
    after_annihilate.insert(insert_at, new)
    return tuple(after_annihilate), sign


@cache
def _sector_occ_tuples(norb: int, nocc: int) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(x) for x in row) for row in occ_rows(norb, nocc))


@cache
def _diag2_features(
    norb: int,
    nelec: tuple[int, int],
    pairs: tuple[tuple[int, int], ...],
) -> np.ndarray:
    occ_a = occ_indicator_rows(norb, nelec[0]).astype(np.float64, copy=False)
    occ_b = occ_indicator_rows(norb, nelec[1]).astype(np.float64, copy=False)
    counts = (
        occ_a[:, None, :] + occ_b[None, :, :]
    ).reshape(occ_a.shape[0] * occ_b.shape[0], norb)
    features = np.empty((counts.shape[0], len(pairs)), dtype=np.float64)
    for col, (p, q) in enumerate(pairs):
        features[:, col] = counts[:, p] * counts[:, q]
    return features


@cache
def _pair_hop_matrices(
    norb: int,
    nelec: tuple[int, int],
    pairs: tuple[tuple[int, int], ...],
) -> tuple[sparse.csr_matrix, ...]:
    occ_a = _sector_occ_tuples(norb, nelec[0])
    occ_b = _sector_occ_tuples(norb, nelec[1])
    map_a = {occ: idx for idx, occ in enumerate(occ_a)}
    map_b = {occ: idx for idx, occ in enumerate(occ_b)}
    dim_a = len(occ_a)
    dim_b = len(occ_b)
    dim = dim_a * dim_b
    matrices = []

    for p, q in pairs:
        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []

        for ia, alpha in enumerate(occ_a):
            alpha_q_to_p = _replace_orbital(alpha, q, p)
            alpha_p_to_q = _replace_orbital(alpha, p, q)
            for ib, beta in enumerate(occ_b):
                source = ia * dim_b + ib

                beta_q_to_p = _replace_orbital(beta, q, p)
                if alpha_q_to_p is not None and beta_q_to_p is not None:
                    new_alpha, sign_a = alpha_q_to_p
                    new_beta, sign_b = beta_q_to_p
                    target = map_a[new_alpha] * dim_b + map_b[new_beta]
                    rows.append(target)
                    cols.append(source)
                    data.append(float(sign_a * sign_b))

                beta_p_to_q = _replace_orbital(beta, p, q)
                if alpha_p_to_q is not None and beta_p_to_q is not None:
                    new_alpha, sign_a = alpha_p_to_q
                    new_beta, sign_b = beta_p_to_q
                    target = map_a[new_alpha] * dim_b + map_b[new_beta]
                    rows.append(target)
                    cols.append(source)
                    data.append(float(-sign_a * sign_b))

        matrices.append(
            sparse.csr_matrix((data, (rows, cols)), shape=(dim, dim), dtype=np.complex128)
        )

    return tuple(matrices)


@cache
def _pair_array(pairs: tuple[tuple[int, int], ...]) -> np.ndarray:
    return np.asarray(pairs, dtype=np.uintp)


@cache
def _pair_hop_transition_arrays(
    norb: int,
    nelec: tuple[int, int],
    pairs: tuple[tuple[int, int], ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sources = []
    targets = []
    pair_indices = []
    signs = []
    for k, mat in enumerate(_pair_hop_matrices(norb, nelec, pairs)):
        coo = mat.tocoo()
        sources.append(np.asarray(coo.col, dtype=np.uintp))
        targets.append(np.asarray(coo.row, dtype=np.uintp))
        pair_indices.append(np.full(coo.nnz, k, dtype=np.uintp))
        signs.append(np.asarray(coo.data.real, dtype=np.float64))
    if not sources:
        empty_idx = np.zeros(0, dtype=np.uintp)
        empty_sign = np.zeros(0, dtype=np.float64)
        return empty_idx, empty_idx, empty_idx, empty_sign
    return (
        np.concatenate(sources),
        np.concatenate(targets),
        np.concatenate(pair_indices),
        np.concatenate(signs),
    )


def gcr2_pairhop_middle_generator(
    pair_params: np.ndarray,
    pair_hop_params: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    pairs: tuple[tuple[int, int], ...],
) -> sparse.csr_matrix:
    pair_params = np.asarray(pair_params, dtype=np.float64)
    pair_hop_params = np.asarray(pair_hop_params, dtype=np.float64)
    if pair_params.shape != (len(pairs),):
        raise ValueError("pair_params has the wrong shape")
    if pair_hop_params.shape != (len(pairs),):
        raise ValueError("pair_hop_params has the wrong shape")

    features = _diag2_features(norb, nelec, pairs)
    diagonal = 1j * (features @ pair_params)
    generator = sparse.diags(diagonal, format="csr", dtype=np.complex128)
    for coeff, mat in zip(pair_hop_params, _pair_hop_matrices(norb, nelec, pairs)):
        if coeff:
            generator = generator + float(coeff) * mat
    return generator.tocsr()


@dataclass(frozen=True)
class GCR2PairHopAnsatz:
    """Single-layer GCR-2 with a non-commuting singlet pair-hop middle sector.

    The middle exponential is

        exp(i D_2 + sum_pq c_pq (P_p^dag P_q - P_q^dag P_p)),

    surrounded by the same left/right orbital rotations used by the iGCR2
    parameterization.  This is intentionally modest: it tests whether the useful
    part of a second UCJ layer can be captured by a commutator-like sector
    without introducing higher-body diagonal GCR-3/GCR-4 terms.
    """

    pair_params: np.ndarray
    pair_hop_params: np.ndarray
    left: np.ndarray
    right: np.ndarray
    norb: int
    nocc: int
    pairs: tuple[tuple[int, int], ...]
    use_rust: bool = True
    rust_max_dim: int | None = 512
    rust_taylor_tol: float = 1e-13
    rust_taylor_max_terms: int = 48

    def middle_generator(self, nelec: tuple[int, int]) -> sparse.csr_matrix:
        return gcr2_pairhop_middle_generator(
            self.pair_params,
            self.pair_hop_params,
            self.norb,
            nelec,
            self.pairs,
        )

    def apply(self, vec: np.ndarray, nelec: tuple[int, int], copy: bool = True) -> np.ndarray:
        out = apply_orbital_rotation(
            vec,
            self.right,
            self.norb,
            nelec,
            copy=copy,
        )
        dim = _diag2_features(self.norb, nelec, self.pairs).shape[0]
        use_rust_allowed = self.use_rust and (
            self.rust_max_dim is None or dim <= self.rust_max_dim
        )
        use_rust_middle = (
            use_rust_allowed
            and apply_gcr2_pairhop_middle_cached_in_place_num_rep is not None
        )
        if use_rust_middle:
            out2 = reshape_state(out, self.norb, nelec)
            source, target, pair_index, sign = _pair_hop_transition_arrays(
                self.norb, nelec, self.pairs
            )
            apply_gcr2_pairhop_middle_cached_in_place_num_rep(
                out2,
                self.pair_params,
                self.pair_hop_params,
                _diag2_features(self.norb, nelec, self.pairs),
                source,
                target,
                pair_index,
                sign,
                float(self.rust_taylor_tol),
                int(self.rust_taylor_max_terms),
            )
            out = flatten_state(out2)
        elif (
            use_rust_allowed
            and apply_gcr2_pairhop_middle_in_place_num_rep is not None
        ):
            out2 = reshape_state(out, self.norb, nelec)
            apply_gcr2_pairhop_middle_in_place_num_rep(
                out2,
                self.pair_params,
                self.pair_hop_params,
                self.norb,
                occ_indicator_rows(self.norb, nelec[0]),
                occ_indicator_rows(self.norb, nelec[1]),
                _pair_array(self.pairs),
                float(self.rust_taylor_tol),
                int(self.rust_taylor_max_terms),
            )
            out = flatten_state(out2)
        else:
            out = expm_multiply(self.middle_generator(nelec), out)
        return apply_orbital_rotation(
            out,
            self.left,
            self.norb,
            nelec,
            copy=False,
        )


@dataclass(frozen=True)
class GCR2PairHopParameterization:
    norb: int
    nocc: int
    pairs: list[tuple[int, int]] | None = None
    base_parameterization: IGCR2SpinRestrictedParameterization | None = None
    left_right_ov_relative_scale: float | None = 1.0
    real_right_orbital_chart: bool = False
    _pairs: tuple[tuple[int, int], ...] = field(init=False, repr=False)

    def __post_init__(self):
        object.__setattr__(self, "_pairs", _validate_pairs(self.pairs, self.norb))
        if self.base_parameterization is not None:
            if self.base_parameterization.norb != self.norb:
                raise ValueError("base_parameterization.norb does not match")
            if self.base_parameterization.nocc != self.nocc:
                raise ValueError("base_parameterization.nocc does not match")

    @property
    def pair_indices(self) -> tuple[tuple[int, int], ...]:
        return self._pairs

    @property
    def _base(self) -> IGCR2SpinRestrictedParameterization:
        if self.base_parameterization is not None:
            return self.base_parameterization
        return IGCR2SpinRestrictedParameterization(
            self.norb,
            self.nocc,
            interaction_pairs=list(self._pairs),
            real_right_orbital_chart=self.real_right_orbital_chart,
            left_right_ov_relative_scale=self.left_right_ov_relative_scale,
        )

    @property
    def n_left_orbital_rotation_params(self) -> int:
        return self._base.n_left_orbital_rotation_params

    @property
    def n_pair_params(self) -> int:
        return len(self._pairs)

    @property
    def n_pair_hop_params(self) -> int:
        return len(self._pairs)

    @property
    def n_right_orbital_rotation_params(self) -> int:
        return self._base.n_right_orbital_rotation_params

    @property
    def n_params(self) -> int:
        return (
            self.n_left_orbital_rotation_params
            + self.n_pair_params
            + self.n_pair_hop_params
            + self.n_right_orbital_rotation_params
        )

    def _split(self, params: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        params = np.asarray(params, dtype=np.float64)
        if params.shape != (self.n_params,):
            raise ValueError(f"Expected {(self.n_params,)}, got {params.shape}.")
        idx = 0
        n = self.n_left_orbital_rotation_params
        left = params[idx : idx + n]
        idx += n
        n = self.n_pair_params
        pair = params[idx : idx + n]
        idx += n
        n = self.n_pair_hop_params
        pair_hop = params[idx : idx + n]
        idx += n
        right = params[idx:]
        return left, pair, pair_hop, right

    def _base_params_from_split(
        self,
        left: np.ndarray,
        pair: np.ndarray,
        right: np.ndarray,
    ) -> np.ndarray:
        return np.concatenate([left, pair, right]).astype(np.float64, copy=False)

    def ansatz_from_parameters(self, params: np.ndarray) -> GCR2PairHopAnsatz:
        left, pair, pair_hop, right = self._split(params)
        base_ansatz = self._base.ansatz_from_parameters(
            self._base_params_from_split(left, pair, right)
        )
        if not isinstance(base_ansatz, IGCR2Ansatz):
            raise TypeError("base parameterization returned an unexpected ansatz")
        pair_matrix = np.asarray(base_ansatz.diagonal.pair, dtype=np.float64)
        pair_values = np.asarray(
            [pair_matrix[p, q] for p, q in self._pairs],
            dtype=np.float64,
        )
        return GCR2PairHopAnsatz(
            pair_params=pair_values,
            pair_hop_params=np.asarray(pair_hop, dtype=np.float64),
            left=np.asarray(base_ansatz.left, dtype=np.complex128),
            right=np.asarray(base_ansatz.right, dtype=np.complex128),
            norb=self.norb,
            nocc=self.nocc,
            pairs=self._pairs,
        )

    def parameters_from_igcr2(
        self,
        params: np.ndarray,
        parameterization: IGCR2SpinRestrictedParameterization | None = None,
    ) -> np.ndarray:
        parameterization = self._base if parameterization is None else parameterization
        if parameterization.norb != self.norb or parameterization.nocc != self.nocc:
            raise ValueError("IGCR2 parameterization shape does not match")
        ansatz = parameterization.ansatz_from_parameters(params)
        base_params = self._base.parameters_from_ansatz(ansatz)
        left = base_params[: self.n_left_orbital_rotation_params]
        pair_start = self.n_left_orbital_rotation_params
        pair_stop = pair_start + self.n_pair_params
        pair = base_params[pair_start:pair_stop]
        right = base_params[pair_stop:]
        return np.concatenate(
            [
                left,
                pair,
                np.zeros(self.n_pair_hop_params, dtype=np.float64),
                right,
            ]
        )

    def parameters_from_ucj_ansatz(self, ansatz: UCJAnsatz) -> np.ndarray:
        base_params = self._base.parameters_from_ucj_ansatz(ansatz)
        left = base_params[: self.n_left_orbital_rotation_params]
        pair_start = self.n_left_orbital_rotation_params
        pair_stop = pair_start + self.n_pair_params
        pair = base_params[pair_start:pair_stop]
        right = base_params[pair_stop:]
        return np.concatenate(
            [
                left,
                pair,
                np.zeros(self.n_pair_hop_params, dtype=np.float64),
                right,
            ]
        )

    def params_to_vec(
        self,
        reference_vec: np.ndarray,
        nelec: tuple[int, int],
    ) -> Callable[[np.ndarray], np.ndarray]:
        reference_vec = np.asarray(reference_vec, dtype=np.complex128)

        def func(params: np.ndarray) -> np.ndarray:
            return self.ansatz_from_parameters(params).apply(
                reference_vec,
                nelec=nelec,
                copy=True,
            )

        return func
