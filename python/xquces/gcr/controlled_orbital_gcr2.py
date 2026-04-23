from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from functools import cache
from typing import Callable

import numpy as np
import scipy.linalg

from xquces.basis import occ_indicator_rows
from xquces.gcr.commutator_gcr2 import _diag2_features, _validate_pairs
from xquces.gcr.igcr2 import IGCR2Ansatz, IGCR2SpinRestrictedParameterization, IGCR2SpinRestrictedSpec
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


# ---------------------------------------------------------------------------
# Two-spectator (C_2) extension
# ---------------------------------------------------------------------------


@cache
def _two_spectator_transform_basis(m: int) -> np.ndarray:
    """Orthonormal null-space basis (columns) of the K_m vertex-edge incidence matrix.

    Returns array of shape (C(m,2), C(m-1,2)).  Each column is a basis vector for
    the gauge-fixed subspace satisfying Σ_{s≠r} ξ_{rs} = 0 for all r.
    """
    n_edges = m * (m - 1) // 2
    n_free = (m - 1) * (m - 2) // 2
    if m <= 2:
        return np.zeros((n_edges, 0), dtype=np.float64)
    edges = list(itertools.combinations(range(m), 2))
    # vertex-edge incidence matrix of K_m
    A = np.zeros((m, n_edges), dtype=np.float64)
    for j, (r, s) in enumerate(edges):
        A[r, j] = 1.0
        A[s, j] = 1.0
    _, s_vals, Vt = np.linalg.svd(A, full_matrices=True)
    # rank(K_m incidence) = m - 1; null space occupies the last n_free rows of Vt
    basis = np.ascontiguousarray(Vt[m - 1:].T, dtype=np.float64)
    assert basis.shape == (n_edges, n_free), f"Expected ({n_edges},{n_free}), got {basis.shape}"
    return basis


def _validate_quadruples(
    quadruples: list[tuple[int, int, int, int]] | None,
    norb: int,
    pairs: tuple[tuple[int, int], ...],
) -> tuple[tuple[int, int, int, int], ...]:
    """Validate or generate the list of (r, s, p, q) two-spectator index quadruples.

    Convention: r < s, r,s ∉ {p,q}.  The default ordering is: for each (p,q) in
    ``pairs`` order, enumerate (r,s) in itertools.combinations order over the
    remaining spectator indices.
    """
    pair_set = set(pairs)
    if quadruples is None:
        out = []
        for p, q in pairs:
            spectators = [x for x in range(norb) if x != p and x != q]
            for r, s in itertools.combinations(spectators, 2):
                out.append((r, s, p, q))
        return tuple(out)
    out = []
    seen: set[tuple[int, int, int, int]] = set()
    for tup in quadruples:
        r, s, p, q = int(tup[0]), int(tup[1]), int(tup[2]), int(tup[3])
        if not (0 <= r < s < norb):
            raise ValueError(f"spectator indices must satisfy 0 <= r < s < norb, got r={r}, s={s}")
        if not (0 <= p < q < norb):
            raise ValueError(f"pair indices must satisfy 0 <= p < q < norb, got p={p}, q={q}")
        if r == p or r == q or s == p or s == q:
            raise ValueError("spectator indices must be distinct from pair indices")
        if (p, q) not in pair_set:
            raise ValueError(f"pair ({p},{q}) is not in the pairs list")
        key = (r, s, p, q)
        if key in seen:
            raise ValueError("quadruples must not contain duplicates")
        seen.add(key)
        out.append(key)
    return tuple(out)


def project_two_spectator_gauge(
    xi: np.ndarray,
    norb: int,
    pairs: tuple[tuple[int, int], ...],
    quadruples: tuple[tuple[int, int, int, int], ...],
) -> np.ndarray:
    """Project two-spectator parameters onto the gauge-fixed subspace.

    Enforces: for each (p,q) and each r ∉ {p,q},  Σ_{s ∉ {p,q,r}} ξ_{rs,pq} = 0.
    The projection is idempotent (applying it twice yields the same result).
    """
    xi = np.asarray(xi, dtype=np.float64).copy()
    m = norb - 2
    if m < 2:
        return xi
    basis = _two_spectator_transform_basis(m)   # (n_edges, n_free)
    proj = basis @ basis.T                       # (n_edges, n_edges) orthogonal projector
    for p, q in pairs:
        group = [idx for idx, quad in enumerate(quadruples) if quad[2] == p and quad[3] == q]
        if not group:
            continue
        g = np.asarray(group, dtype=np.intp)
        xi[g] = proj @ xi[g]
    return xi


@cache
def _two_spectator_sector_indices(
    norb: int,
    nelec: tuple[int, int],
    r: int,
    s: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """FCI state indices for each joint occupation sector of orbitals r and s.

    Returns (sec_00, sec_02, sec_20, sec_22) where sec_{nr}{ns} contains FCI
    indices where N_r = nr and N_s = ns, with N = n_α + n_β ∈ {0, 1, 2}.
    Only sectors where N_r, N_s ∈ {0, 2} are active under Ñ_r Ñ_s (Ñ = N − 1).
    """
    occ_a = occ_indicator_rows(norb, nelec[0])
    occ_b = occ_indicator_rows(norb, nelec[1])
    dim_a, dim_b = occ_a.shape[0], occ_b.shape[0]
    Nr = occ_a[:, r][:, None].astype(np.int8) + occ_b[:, r][None, :].astype(np.int8)
    Ns = occ_a[:, s][:, None].astype(np.int8) + occ_b[:, s][None, :].astype(np.int8)
    flat = np.arange(dim_a * dim_b, dtype=np.intp).reshape(dim_a, dim_b)

    def _get(nr: int, ns: int) -> np.ndarray:
        return flat[(Nr == nr) & (Ns == ns)].ravel().astype(np.uintp, copy=False)

    return _get(0, 0), _get(0, 2), _get(2, 0), _get(2, 2)


def _apply_two_spectator_rotation(
    vec: np.ndarray,
    xi: float,
    r: int,
    s: int,
    p: int,
    q: int,
    norb: int,
    nelec: tuple[int, int],
    copy: bool = True,
) -> np.ndarray:
    """Apply exp(ξ Ñ_r Ñ_s A_{pq}) where Ñ_r = N_r − 1.

    Controlled-controlled Givens rotation:
      Ñ_r Ñ_s = +1  when (N_r, N_s) ∈ {(0,0), (2,2)}  → Givens by +ξ
      Ñ_r Ñ_s = −1  when (N_r, N_s) ∈ {(0,2), (2,0)}  → Givens by −ξ
      Ñ_r Ñ_s =  0  when N_r = 1 or N_s = 1            → identity
    """
    out = np.array(vec, dtype=np.complex128, copy=copy)
    if abs(xi) <= 1e-15:
        return out
    sec00, sec02, sec20, sec22 = _two_spectator_sector_indices(norb, nelec, r, s)
    if sec00.size == 0 and sec02.size == 0 and sec20.size == 0 and sec22.size == 0:
        return out
    base = np.array(out, copy=True)
    for sec in (sec00, sec02, sec20, sec22):
        if sec.size:
            out[sec] = 0.0
    # (N_r=0,N_s=0) and (N_r=2,N_s=2): Ñ_r Ñ_s = +1 → rotate by +ξ
    for sec in (sec00, sec22):
        if sec.size:
            tmp = np.zeros_like(base)
            tmp[sec] = base[sec]
            out += apply_orbital_rotation(
                tmp, _pair_rotation_unitary(norb, p, q, xi), norb, nelec, copy=False
            )
    # (N_r=0,N_s=2) and (N_r=2,N_s=0): Ñ_r Ñ_s = −1 → rotate by −ξ
    for sec in (sec02, sec20):
        if sec.size:
            tmp = np.zeros_like(base)
            tmp[sec] = base[sec]
            out += apply_orbital_rotation(
                tmp, _pair_rotation_unitary(norb, p, q, -xi), norb, nelec, copy=False
            )
    return out


@dataclass(frozen=True)
class GCR2TwoSpectatorOrbitalAnsatz:
    """Full ansatz |ψ⟩ = U_L e^{iD/2} C_2(Ξ) C_1(Θ) e^{iD/2} U_R |Φ_0⟩.

    C_1 is the one-spectator block (innermost), C_2 is the two-spectator correction.
    """

    pair_params: np.ndarray
    spectator_params: np.ndarray       # full C_1 parameters, one per triple
    two_spec_params: np.ndarray        # full C_2 parameters, one per quadruple
    left: np.ndarray
    right: np.ndarray
    norb: int
    nocc: int
    pairs: tuple[tuple[int, int], ...]
    triples: tuple[tuple[int, int, int], ...]
    quadruples: tuple[tuple[int, int, int, int], ...]

    def apply(self, vec: np.ndarray, nelec: tuple[int, int], copy: bool = True) -> np.ndarray:
        out = apply_orbital_rotation(vec, self.right, self.norb, nelec, copy=copy)
        _phase_diag2(out, 0.5 * self.pair_params, self.norb, nelec, self.pairs)
        # C_1(Θ): one-spectator block (innermost)
        for theta, (r, p, q) in zip(self.spectator_params, self.triples):
            out = _apply_spectator_controlled_rotation(
                out, float(theta), r, p, q, self.norb, nelec, copy=False
            )
        # C_2(Ξ): two-spectator block (outermost correction)
        for xi_val, (r, s, p, q) in zip(self.two_spec_params, self.quadruples):
            out = _apply_two_spectator_rotation(
                out, float(xi_val), r, s, p, q, self.norb, nelec, copy=False
            )
        _phase_diag2(out, 0.5 * self.pair_params, self.norb, nelec, self.pairs)
        return apply_orbital_rotation(out, self.left, self.norb, nelec, copy=False)


def _expected_two_spec_param_count(norb: int) -> dict[str, int]:
    """Return raw/gauge-fixed two-spectator parameter counts for a given norb."""
    m = norb - 2
    n_pairs = norb * (norb - 1) // 2
    n_raw = m * (m - 1) // 2 * n_pairs            # C(m,2) × C(norb,2)
    n_gauge = (m - 1) * (m - 2) // 2 * n_pairs    # C(m-1,2) × C(norb,2)
    return {"raw": n_raw, "gauge_fixed": n_gauge, "n_pairs": n_pairs, "m": m}


@dataclass(frozen=True)
class GCR2TwoSpectatorOrbitalParameterization:
    """Parameterization for the full two-spectator GCR-2 ansatz.

    Extends :class:`GCR2SpectatorOrbitalParameterization` by adding a second
    spectator block C_2(Ξ) applied outside C_1(Θ).  Parameters are laid out as::

        [left_orbital | pair | spectator_C1 (reduced) | two_spec_C2 (reduced) | right_orbital]

    The two-spectator parameters live in the null space of the vertex-edge
    incidence matrix of K_{norb-2} (one copy per orbital pair (p,q)), which
    enforces Σ_{s ∉ {p,q,r}} ξ_{rs,pq} = 0 for every r.
    """

    norb: int
    nocc: int
    pairs: list[tuple[int, int]] | None = None
    triples: list[tuple[int, int, int]] | None = None
    quadruples: list[tuple[int, int, int, int]] | None = None
    base_parameterization: IGCR2SpinRestrictedParameterization | None = None
    left_right_ov_relative_scale: float | None = 1.0
    real_right_orbital_chart: bool = False
    _pairs: tuple[tuple[int, int], ...] = field(init=False, repr=False)
    _triples: tuple[tuple[int, int, int], ...] = field(init=False, repr=False)
    _quadruples: tuple[tuple[int, int, int, int], ...] = field(init=False, repr=False)
    _spectator_transform: np.ndarray = field(init=False, repr=False)
    _two_spec_transform: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        # --- pairs ---
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

        # --- triples (C_1) ---
        triples = _validate_triples(self.triples, self.norb, self._pairs)
        object.__setattr__(self, "_triples", triples)

        # --- quadruples (C_2) ---
        quadruples = _validate_quadruples(self.quadruples, self.norb, self._pairs)
        object.__setattr__(self, "_quadruples", quadruples)

        # --- one-spectator transform (Helmert basis per (p,q) group) ---
        spec_blocks = []
        for p, q in self._pairs:
            group = [i for i, t in enumerate(triples) if t[1] == p and t[2] == q]
            basis = _helmert_basis(len(group))
            if basis.shape[1] == 0:
                continue
            block = np.zeros((len(triples), basis.shape[1]), dtype=np.float64)
            block[np.asarray(group, dtype=np.int64), :] = basis
            spec_blocks.append(block)
        spec_transform = (
            np.hstack(spec_blocks) if spec_blocks
            else np.zeros((len(triples), 0), dtype=np.float64)
        )
        object.__setattr__(self, "_spectator_transform", spec_transform)

        # --- two-spectator transform (null-space basis per (p,q) group) ---
        m = self.norb - 2
        two_blocks = []
        for p, q in self._pairs:
            group = [i for i, quad in enumerate(quadruples) if quad[2] == p and quad[3] == q]
            if not group:
                continue
            basis = _two_spectator_transform_basis(m)   # (C(m,2), C(m-1,2))
            if basis.shape[1] == 0:
                continue
            block = np.zeros((len(quadruples), basis.shape[1]), dtype=np.float64)
            block[np.asarray(group, dtype=np.int64), :] = basis
            two_blocks.append(block)
        two_transform = (
            np.hstack(two_blocks) if two_blocks
            else np.zeros((len(quadruples), 0), dtype=np.float64)
        )
        object.__setattr__(self, "_two_spec_transform", two_transform)

    # ------------------------------------------------------------------
    # Index accessors
    # ------------------------------------------------------------------

    @property
    def pair_indices(self) -> tuple[tuple[int, int], ...]:
        return self._pairs

    @property
    def triple_indices(self) -> tuple[tuple[int, int, int], ...]:
        return self._triples

    @property
    def quadruple_indices(self) -> tuple[tuple[int, int, int, int], ...]:
        return self._quadruples

    @property
    def spectator_transform(self) -> np.ndarray:
        return self._spectator_transform

    @property
    def two_spec_transform(self) -> np.ndarray:
        return self._two_spec_transform

    # ------------------------------------------------------------------
    # Base parameterization
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Parameter counts
    # ------------------------------------------------------------------

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
    def n_full_two_spec_terms(self) -> int:
        return len(self._quadruples)

    @property
    def n_two_spec_params(self) -> int:
        return self._two_spec_transform.shape[1]

    @property
    def n_right_orbital_rotation_params(self) -> int:
        return self._base.n_right_orbital_rotation_params

    @property
    def n_params(self) -> int:
        return (
            self.n_left_orbital_rotation_params
            + self.n_pair_params
            + self.n_spectator_params
            + self.n_two_spec_params
            + self.n_right_orbital_rotation_params
        )

    # ------------------------------------------------------------------
    # Transforms between reduced and full parameter vectors
    # ------------------------------------------------------------------

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

    def full_two_spec_params_from_reduced(self, params: np.ndarray) -> np.ndarray:
        params = np.asarray(params, dtype=np.float64)
        if params.shape != (self.n_two_spec_params,):
            raise ValueError(f"Expected {(self.n_two_spec_params,)}, got {params.shape}.")
        return self._two_spec_transform @ params

    def reduced_two_spec_params_from_full(self, params: np.ndarray) -> np.ndarray:
        params = np.asarray(params, dtype=np.float64)
        if params.shape != (self.n_full_two_spec_terms,):
            raise ValueError(f"Expected {(self.n_full_two_spec_terms,)}, got {params.shape}.")
        return self._two_spec_transform.T @ params

    # ------------------------------------------------------------------
    # Internal split/merge helpers
    # ------------------------------------------------------------------

    def _split(
        self, params: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        params = np.asarray(params, dtype=np.float64)
        if params.shape != (self.n_params,):
            raise ValueError(f"Expected {(self.n_params,)}, got {params.shape}.")
        idx = 0
        n = self.n_left_orbital_rotation_params
        left = params[idx: idx + n]; idx += n
        n = self.n_pair_params
        pair = params[idx: idx + n]; idx += n
        n = self.n_spectator_params
        spectator = params[idx: idx + n]; idx += n
        n = self.n_two_spec_params
        two_spec = params[idx: idx + n]; idx += n
        right = params[idx:]
        return left, pair, spectator, two_spec, right

    def _base_params_from_split(
        self, left: np.ndarray, pair: np.ndarray, right: np.ndarray
    ) -> np.ndarray:
        return np.concatenate([left, pair, right]).astype(np.float64, copy=False)

    # ------------------------------------------------------------------
    # Ansatz construction
    # ------------------------------------------------------------------

    def ansatz_from_parameters(self, params: np.ndarray) -> GCR2TwoSpectatorOrbitalAnsatz:
        left, pair, spectator, two_spec, right = self._split(params)
        base_ansatz = self._base.ansatz_from_parameters(
            self._base_params_from_split(left, pair, right)
        )
        if not isinstance(base_ansatz, IGCR2Ansatz):
            raise TypeError("base parameterization returned an unexpected ansatz type")
        pair_matrix = np.asarray(base_ansatz.diagonal.pair, dtype=np.float64)
        pair_values = np.asarray(
            [pair_matrix[p, q] for p, q in self._pairs], dtype=np.float64
        )
        spectator_full = self._spectator_transform @ np.asarray(spectator, dtype=np.float64)
        two_spec_full = self._two_spec_transform @ np.asarray(two_spec, dtype=np.float64)
        return GCR2TwoSpectatorOrbitalAnsatz(
            pair_params=pair_values,
            spectator_params=spectator_full,
            two_spec_params=two_spec_full,
            left=np.asarray(base_ansatz.left, dtype=np.complex128),
            right=np.asarray(base_ansatz.right, dtype=np.complex128),
            norb=self.norb,
            nocc=self.nocc,
            pairs=self._pairs,
            triples=self._triples,
            quadruples=self._quadruples,
        )

    # ------------------------------------------------------------------
    # Inverse: parameters from ansatz (round-trip)
    # ------------------------------------------------------------------

    def parameters_from_ansatz(self, ansatz: GCR2TwoSpectatorOrbitalAnsatz) -> np.ndarray:
        """Extract reduced parameters from an ansatz (inverse of ansatz_from_parameters)."""
        if ansatz.norb != self.norb:
            raise ValueError("ansatz norb does not match parameterization")
        if ansatz.pairs != self._pairs:
            raise ValueError("ansatz pairs do not match parameterization")
        if ansatz.triples != self._triples:
            raise ValueError("ansatz triples do not match parameterization")
        if ansatz.quadruples != self._quadruples:
            raise ValueError("ansatz quadruples do not match parameterization")
        pair_matrix = np.zeros((self.norb, self.norb), dtype=np.float64)
        for (p, q), val in zip(self._pairs, ansatz.pair_params):
            pair_matrix[p, q] = float(val)
            pair_matrix[q, p] = float(val)
        igcr2_ansatz = IGCR2Ansatz(
            diagonal=IGCR2SpinRestrictedSpec(pair=pair_matrix),
            left=np.asarray(ansatz.left, dtype=np.complex128),
            right=np.asarray(ansatz.right, dtype=np.complex128),
            nocc=self.nocc,
        )
        base_params = self._base.parameters_from_ansatz(igcr2_ansatz)
        n_left = self.n_left_orbital_rotation_params
        n_pair = self.n_pair_params
        left = base_params[:n_left]
        pair = base_params[n_left: n_left + n_pair]
        right = base_params[n_left + n_pair:]
        spectator_reduced = self._spectator_transform.T @ np.asarray(
            ansatz.spectator_params, dtype=np.float64
        )
        two_spec_reduced = self._two_spec_transform.T @ np.asarray(
            ansatz.two_spec_params, dtype=np.float64
        )
        return np.concatenate([left, pair, spectator_reduced, two_spec_reduced, right])

    # ------------------------------------------------------------------
    # Initialization from lower-level parameterizations
    # ------------------------------------------------------------------

    def parameters_from_igcr2(
        self,
        params: np.ndarray,
        parameterization: IGCR2SpinRestrictedParameterization | None = None,
    ) -> np.ndarray:
        """Lift IGCR2 params to two-spectator params (C_1 = C_2 = 0)."""
        parameterization = self._base if parameterization is None else parameterization
        if parameterization.norb != self.norb or parameterization.nocc != self.nocc:
            raise ValueError("IGCR2 parameterization shape does not match")
        ansatz = parameterization.ansatz_from_parameters(params)
        base_params = self._base.parameters_from_ansatz(ansatz)
        n_left = self.n_left_orbital_rotation_params
        n_pair = self.n_pair_params
        left = base_params[:n_left]
        pair = base_params[n_left: n_left + n_pair]
        right = base_params[n_left + n_pair:]
        return np.concatenate([
            left,
            pair,
            np.zeros(self.n_spectator_params, dtype=np.float64),
            np.zeros(self.n_two_spec_params, dtype=np.float64),
            right,
        ])

    def parameters_from_one_spectator(
        self,
        params: np.ndarray,
        one_spec_parameterization: GCR2SpectatorOrbitalParameterization | None = None,
    ) -> np.ndarray:
        """Lift one-spectator params to two-spectator params (pads ξ with zeros)."""
        if one_spec_parameterization is None:
            one_spec_parameterization = GCR2SpectatorOrbitalParameterization(
                norb=self.norb,
                nocc=self.nocc,
                pairs=list(self._pairs) if self._pairs else None,
                base_parameterization=self.base_parameterization,
                left_right_ov_relative_scale=self.left_right_ov_relative_scale,
                real_right_orbital_chart=self.real_right_orbital_chart,
            )
        if one_spec_parameterization.norb != self.norb:
            raise ValueError("one_spec_parameterization.norb does not match")
        one_left, one_pair, one_spec, one_right = one_spec_parameterization._split(params)
        # Map C_1 spectator params: assume same triples structure, else zero-pad
        if self.n_spectator_params == one_spec_parameterization.n_spectator_params:
            spectator_out = np.asarray(one_spec, dtype=np.float64)
        else:
            spectator_out = np.zeros(self.n_spectator_params, dtype=np.float64)
        return np.concatenate([
            one_left,
            one_pair,
            spectator_out,
            np.zeros(self.n_two_spec_params, dtype=np.float64),
            one_right,
        ])

    def parameters_from_ucj_ansatz(
        self,
        ansatz: UCJAnsatz,
        initialize_spectator: bool = True,
        spectator_scale: float = 0.01,
    ) -> np.ndarray:
        """Build two-spectator params from a UCJ ansatz (ξ = 0, C_1 optionally seeded)."""
        base_params = self._base.parameters_from_ucj_ansatz(ansatz)
        base_ansatz = self._base.ansatz_from_parameters(base_params)
        n_left = self.n_left_orbital_rotation_params
        n_pair = self.n_pair_params
        left = base_params[:n_left]
        pair = base_params[n_left: n_left + n_pair]
        right = base_params[n_left + n_pair:]
        if initialize_spectator and self.n_spectator_params > 0:
            spectator = _heuristic_spectator_params(
                base_ansatz, self._triples, self._spectator_transform, spectator_scale
            )
        else:
            spectator = np.zeros(self.n_spectator_params, dtype=np.float64)
        return np.concatenate([
            left, pair, spectator, np.zeros(self.n_two_spec_params, dtype=np.float64), right
        ])

    # ------------------------------------------------------------------
    # One-step Newton initialization for ξ
    # ------------------------------------------------------------------

    def initialize_two_spec_newton(
        self,
        params: np.ndarray,
        energy_func: Callable[[np.ndarray], float],
        fd_step: float = 1e-4,
        damping: float = 1e-3,
    ) -> np.ndarray:
        """One-step Newton update for the two-spectator parameters.

        With the C_1/orbital parameters frozen at ``params``, computes
        ξ_init = −g_k / (h_k + λ)  per reduced parameter k, where g_k and h_k
        are the first- and second-order finite-difference approximations of the
        energy gradient and Hessian along ξ_k, and λ = damping × max|h|.

        Returns a new full parameter vector with the ξ block replaced.
        """
        if self.n_two_spec_params == 0:
            return np.asarray(params, dtype=np.float64).copy()
        left, pair, spectator, _, right = self._split(params)
        base = np.concatenate([left, pair, spectator, np.zeros(self.n_two_spec_params), right])
        n_two = self.n_two_spec_params
        two_spec_start = self.n_left_orbital_rotation_params + self.n_pair_params + self.n_spectator_params

        e0 = energy_func(base)
        g = np.zeros(n_two, dtype=np.float64)
        h = np.zeros(n_two, dtype=np.float64)
        for k in range(n_two):
            x_plus = base.copy(); x_plus[two_spec_start + k] += fd_step
            x_minus = base.copy(); x_minus[two_spec_start + k] -= fd_step
            e_plus = energy_func(x_plus)
            e_minus = energy_func(x_minus)
            g[k] = (e_plus - e_minus) / (2.0 * fd_step)
            h[k] = (e_plus - 2.0 * e0 + e_minus) / (fd_step ** 2)

        lam = damping * float(np.max(np.abs(h))) if np.any(h != 0) else damping
        xi_reduced = -g / (h + lam)

        # Project onto gauge subspace
        xi_full = self.full_two_spec_params_from_reduced(xi_reduced)
        xi_full = project_two_spectator_gauge(
            xi_full, self.norb, self._pairs, self._quadruples
        )
        xi_reduced = self.reduced_two_spec_params_from_full(xi_full)

        out = base.copy()
        out[two_spec_start: two_spec_start + n_two] = xi_reduced
        return out

    # ------------------------------------------------------------------
    # Functional interface
    # ------------------------------------------------------------------

    def params_to_vec(
        self,
        reference_vec: np.ndarray,
        nelec: tuple[int, int],
    ) -> Callable[[np.ndarray], np.ndarray]:
        reference_vec = np.asarray(reference_vec, dtype=np.complex128)

        def func(params: np.ndarray) -> np.ndarray:
            return self.ansatz_from_parameters(params).apply(
                reference_vec, nelec=nelec, copy=True
            )

        return func


def _heuristic_spectator_params(
    ansatz: IGCR2Ansatz,
    triples: tuple[tuple[int, int, int], ...],
    spectator_transform: np.ndarray,
    spectator_scale: float,
) -> np.ndarray:
    """Heuristic seed for C_1 spectator params from an IGCR2 ansatz."""
    if spectator_scale <= 0 or spectator_transform.shape[1] == 0:
        return np.zeros(spectator_transform.shape[1], dtype=np.float64)
    pair = np.asarray(ansatz.diagonal.pair, dtype=np.float64)
    kappa_left = _principal_antihermitian_log(ansatz.left)
    kappa_right = _principal_antihermitian_log(ansatz.right)
    kappa = 0.5 * (kappa_left + kappa_right)
    raw_full = np.zeros(len(triples), dtype=np.float64)
    by_pair: dict[tuple[int, int], list[int]] = {}
    for idx, (r, p, q) in enumerate(triples):
        by_pair.setdefault((p, q), []).append(idx)
        delta = float(pair[r, p] - pair[r, q])
        amp = _dominant_real_component(kappa[p, q])
        raw_full[idx] = delta * amp
    for indices in by_pair.values():
        vals = raw_full[indices]
        raw_full[indices] = vals - float(np.mean(vals))
    raw = spectator_transform.T @ raw_full
    max_abs = float(np.max(np.abs(raw))) if raw.size else 0.0
    if max_abs <= 1e-14:
        return np.zeros(spectator_transform.shape[1], dtype=np.float64)
    return spectator_scale * raw / max_abs
