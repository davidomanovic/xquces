from __future__ import annotations

from dataclasses import dataclass, field
from functools import cache
from typing import Callable

import numpy as np
import scipy.linalg

from xquces.basis import occ_indicator_rows
from xquces.gcr.commutator_gcr2 import _diag2_features, _validate_pairs
from xquces.gcr.igcr2 import IGCR2Ansatz, IGCR2SpinRestrictedParameterization
from xquces.orbitals import apply_orbital_rotation
from xquces.ucj.model import UCJAnsatz


def _phase_diag2(
    vec: np.ndarray,
    pair_params: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    pairs: tuple[tuple[int, int], ...],
) -> None:
    pair_params = np.asarray(pair_params, dtype=np.float64)
    if pair_params.shape != (len(pairs),):
        raise ValueError("pair_params has the wrong shape")
    phases = _diag2_features(norb, nelec, pairs) @ pair_params
    vec *= np.exp(1j * phases)


def _validate_triples(
    triples: list[tuple[int, int, int]] | None,
    norb: int,
    pairs: tuple[tuple[int, int], ...],
) -> tuple[tuple[int, int, int], ...]:
    pair_set = set(pairs)
    if triples is None:
        out = []
        for p, q in pairs:
            for r in range(norb):
                if r == p or r == q:
                    continue
                out.append((r, p, q))
        return tuple(out)
    out = []
    seen = set()
    for r, p, q in triples:
        r = int(r)
        p = int(p)
        q = int(q)
        if not (0 <= r < norb):
            raise ValueError("spectator index out of bounds")
        if not (0 <= p < q < norb):
            raise ValueError("pair indices must satisfy 0 <= p < q < norb")
        if r == p or r == q:
            raise ValueError("spectator index must be distinct from the target pair")
        if (p, q) not in pair_set:
            raise ValueError("target pair is not present in pair list")
        triple = (r, p, q)
        if triple in seen:
            raise ValueError("triples must not contain duplicates")
        seen.add(triple)
        out.append(triple)
    return tuple(out)


def _helmert_basis(m: int) -> np.ndarray:
    if m <= 1:
        return np.zeros((m, 0), dtype=np.float64)
    out = np.zeros((m, m - 1), dtype=np.float64)
    for k in range(1, m):
        norm = np.sqrt(k * (k + 1))
        out[:k, k - 1] = 1.0 / norm
        out[k, k - 1] = -float(k) / norm
    return out


@cache
def _spectator_sector_indices(
    norb: int,
    nelec: tuple[int, int],
    spectator: int,
) -> tuple[np.ndarray, np.ndarray]:
    occ_a = occ_indicator_rows(norb, nelec[0])
    occ_b = occ_indicator_rows(norb, nelec[1])
    rows0 = np.flatnonzero(occ_a[:, spectator] == 0).astype(np.uintp, copy=False)
    rows1 = np.flatnonzero(occ_a[:, spectator] == 1).astype(np.uintp, copy=False)
    cols0 = np.flatnonzero(occ_b[:, spectator] == 0).astype(np.uintp, copy=False)
    cols1 = np.flatnonzero(occ_b[:, spectator] == 1).astype(np.uintp, copy=False)
    dim_b = occ_b.shape[0]

    def make_indices(rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        if rows.size == 0 or cols.size == 0:
            return np.zeros(0, dtype=np.uintp)
        return (rows[:, None] * dim_b + cols[None, :]).reshape(-1).astype(np.uintp, copy=False)

    return make_indices(rows0, cols0), make_indices(rows1, cols1)


def _pair_rotation_unitary(norb: int, p: int, q: int, theta: float) -> np.ndarray:
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    out = np.eye(norb, dtype=np.complex128)
    out[p, p] = c
    out[q, q] = c
    out[p, q] = -s
    out[q, p] = s
    return out


def _apply_spectator_controlled_rotation(
    vec: np.ndarray,
    theta: float,
    spectator: int,
    p: int,
    q: int,
    norb: int,
    nelec: tuple[int, int],
    copy: bool = True,
) -> np.ndarray:
    out = np.array(vec, dtype=np.complex128, copy=copy)
    if abs(theta) <= 1e-15:
        return out
    sector00, sector22 = _spectator_sector_indices(norb, nelec, spectator)
    if sector00.size == 0 and sector22.size == 0:
        return out
    base = np.array(out, dtype=np.complex128, copy=True)
    if sector00.size:
        out[sector00] = 0.0
    if sector22.size:
        out[sector22] = 0.0
    if sector00.size:
        tmp = np.zeros_like(base)
        tmp[sector00] = base[sector00]
        out += apply_orbital_rotation(
            tmp,
            _pair_rotation_unitary(norb, p, q, -theta),
            norb,
            nelec,
            copy=False,
        )
    if sector22.size:
        tmp = np.zeros_like(base)
        tmp[sector22] = base[sector22]
        out += apply_orbital_rotation(
            tmp,
            _pair_rotation_unitary(norb, p, q, theta),
            norb,
            nelec,
            copy=False,
        )
    return out


def _principal_antihermitian_log(u: np.ndarray) -> np.ndarray:
    kappa = scipy.linalg.logm(np.asarray(u, dtype=np.complex128))
    return 0.5 * (kappa - kappa.conj().T)


def _dominant_real_component(z: complex) -> float:
    return float(np.real(z) if abs(np.real(z)) >= abs(np.imag(z)) else np.imag(z))


@dataclass(frozen=True)
class GCR2SpectatorOrbitalAnsatz:
    pair_params: np.ndarray
    spectator_params: np.ndarray
    left: np.ndarray
    right: np.ndarray
    norb: int
    nocc: int
    pairs: tuple[tuple[int, int], ...]
    triples: tuple[tuple[int, int, int], ...]

    def apply(self, vec: np.ndarray, nelec: tuple[int, int], copy: bool = True) -> np.ndarray:
        out = apply_orbital_rotation(
            vec,
            self.right,
            self.norb,
            nelec,
            copy=copy,
        )
        _phase_diag2(out, 0.5 * self.pair_params, self.norb, nelec, self.pairs)
        for theta, (r, p, q) in zip(self.spectator_params, self.triples):
            out = _apply_spectator_controlled_rotation(
                out,
                float(theta),
                r,
                p,
                q,
                self.norb,
                nelec,
                copy=False,
            )
        _phase_diag2(out, 0.5 * self.pair_params, self.norb, nelec, self.pairs)
        return apply_orbital_rotation(
            out,
            self.left,
            self.norb,
            nelec,
            copy=False,
        )


@dataclass(frozen=True)
class GCR2SpectatorOrbitalParameterization:
    norb: int
    nocc: int
    pairs: list[tuple[int, int]] | None = None
    triples: list[tuple[int, int, int]] | None = None
    base_parameterization: IGCR2SpinRestrictedParameterization | None = None
    left_right_ov_relative_scale: float | None = 1.0
    real_right_orbital_chart: bool = False
    _pairs: tuple[tuple[int, int], ...] = field(init=False, repr=False)
    _triples: tuple[tuple[int, int, int], ...] = field(init=False, repr=False)
    _spectator_transform: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        if self.base_parameterization is not None:
            if self.base_parameterization.norb != self.norb:
                raise ValueError("base_parameterization.norb does not match")
            if self.base_parameterization.nocc != self.nocc:
                raise ValueError("base_parameterization.nocc does not match")
            base_pairs = tuple(self.base_parameterization.pair_indices)
            if self.pairs is None:
                object.__setattr__(self, "_pairs", base_pairs)
            else:
                pairs = _validate_pairs(self.pairs, self.norb)
                if pairs != base_pairs:
                    raise ValueError("pairs do not match base_parameterization")
                object.__setattr__(self, "_pairs", pairs)
        else:
            object.__setattr__(self, "_pairs", _validate_pairs(self.pairs, self.norb))
        triples = _validate_triples(self.triples, self.norb, self._pairs)
        object.__setattr__(self, "_triples", triples)

        transform_blocks = []
        cursor = 0
        for p, q in self._pairs:
            group = [idx for idx, triple in enumerate(triples) if triple[1] == p and triple[2] == q]
            basis = _helmert_basis(len(group))
            if basis.shape[1] == 0:
                continue
            block = np.zeros((len(triples), basis.shape[1]), dtype=np.float64)
            block[np.asarray(group, dtype=np.int64), :] = basis
            transform_blocks.append(block)
            cursor += basis.shape[1]
        transform = (
            np.hstack(transform_blocks)
            if transform_blocks
            else np.zeros((len(triples), 0), dtype=np.float64)
        )
        object.__setattr__(self, "_spectator_transform", transform)

    @property
    def pair_indices(self) -> tuple[tuple[int, int], ...]:
        return self._pairs

    @property
    def triple_indices(self) -> tuple[tuple[int, int, int], ...]:
        return self._triples

    @property
    def spectator_transform(self) -> np.ndarray:
        return self._spectator_transform

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
    def n_full_spectator_terms(self) -> int:
        return len(self._triples)

    @property
    def n_spectator_params(self) -> int:
        return self._spectator_transform.shape[1]

    @property
    def n_right_orbital_rotation_params(self) -> int:
        return self._base.n_right_orbital_rotation_params

    @property
    def n_params(self) -> int:
        return (
            self.n_left_orbital_rotation_params
            + self.n_pair_params
            + self.n_spectator_params
            + self.n_right_orbital_rotation_params
        )

    def full_spectator_params_from_reduced(self, params: np.ndarray) -> np.ndarray:
        params = np.asarray(params, dtype=np.float64)
        if params.shape != (self.n_spectator_params,):
            raise ValueError(f"Expected {(self.n_spectator_params,)}, got {params.shape}.")
        return self._spectator_transform @ params

    def reduced_spectator_params_from_full(self, params: np.ndarray) -> np.ndarray:
        params = np.asarray(params, dtype=np.float64)
        if params.shape != (self.n_full_spectator_terms,):
            raise ValueError(f"Expected {(self.n_full_spectator_terms,)}, got {params.shape}.")
        return self._spectator_transform.T @ params

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
        n = self.n_spectator_params
        spectator = params[idx : idx + n]
        idx += n
        right = params[idx:]
        return left, pair, spectator, right

    def _base_params_from_split(
        self,
        left: np.ndarray,
        pair: np.ndarray,
        right: np.ndarray,
    ) -> np.ndarray:
        return np.concatenate([left, pair, right]).astype(np.float64, copy=False)

    def heuristic_spectator_params_from_ansatz(
        self,
        ansatz: IGCR2Ansatz,
        spectator_scale: float = 0.01,
    ) -> np.ndarray:
        if not isinstance(ansatz, IGCR2Ansatz):
            raise TypeError(type(ansatz).__name__)
        if spectator_scale <= 0 or self.n_spectator_params == 0:
            return np.zeros(self.n_spectator_params, dtype=np.float64)
        pair = np.asarray(ansatz.diagonal.pair, dtype=np.float64)
        kappa_left = _principal_antihermitian_log(ansatz.left)
        kappa_right = _principal_antihermitian_log(ansatz.right)
        kappa = 0.5 * (kappa_left + kappa_right)
        raw_full = np.zeros(self.n_full_spectator_terms, dtype=np.float64)
        by_pair: dict[tuple[int, int], list[int]] = {}
        for idx, (r, p, q) in enumerate(self._triples):
            by_pair.setdefault((p, q), []).append(idx)
            delta = float(pair[r, p] - pair[r, q])
            amp = _dominant_real_component(kappa[p, q])
            raw_full[idx] = delta * amp
        for indices in by_pair.values():
            values = raw_full[indices]
            raw_full[indices] = values - float(np.mean(values))
        raw = self.reduced_spectator_params_from_full(raw_full)
        max_abs = float(np.max(np.abs(raw))) if raw.size else 0.0
        if max_abs <= 1e-14:
            return np.zeros(self.n_spectator_params, dtype=np.float64)
        return spectator_scale * raw / max_abs

    def ansatz_from_parameters(self, params: np.ndarray) -> GCR2SpectatorOrbitalAnsatz:
        left, pair, spectator, right = self._split(params)
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
        spectator_full = self.full_spectator_params_from_reduced(np.asarray(spectator, dtype=np.float64))
        return GCR2SpectatorOrbitalAnsatz(
            pair_params=pair_values,
            spectator_params=spectator_full,
            left=np.asarray(base_ansatz.left, dtype=np.complex128),
            right=np.asarray(base_ansatz.right, dtype=np.complex128),
            norb=self.norb,
            nocc=self.nocc,
            pairs=self._pairs,
            triples=self._triples,
        )

    def parameters_from_igcr2(
        self,
        params: np.ndarray,
        parameterization: IGCR2SpinRestrictedParameterization | None = None,
        initialize_spectator: bool = True,
        spectator_scale: float = 0.01,
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
        spectator = (
            self.heuristic_spectator_params_from_ansatz(ansatz, spectator_scale)
            if initialize_spectator
            else np.zeros(self.n_spectator_params, dtype=np.float64)
        )
        return np.concatenate([left, pair, spectator, right])

    def parameters_from_ucj_ansatz(
        self,
        ansatz: UCJAnsatz,
        initialize_spectator: bool = True,
        spectator_scale: float = 0.01,
    ) -> np.ndarray:
        base_params = self._base.parameters_from_ucj_ansatz(ansatz)
        base_ansatz = self._base.ansatz_from_parameters(base_params)
        left = base_params[: self.n_left_orbital_rotation_params]
        pair_start = self.n_left_orbital_rotation_params
        pair_stop = pair_start + self.n_pair_params
        pair = base_params[pair_start:pair_stop]
        right = base_params[pair_stop:]
        spectator = (
            self.heuristic_spectator_params_from_ansatz(base_ansatz, spectator_scale)
            if initialize_spectator
            else np.zeros(self.n_spectator_params, dtype=np.float64)
        )
        return np.concatenate([left, pair, spectator, right])

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
