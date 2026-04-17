from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import scipy.linalg
import scipy.optimize

from xquces.gates import apply_gcr_spin_balanced, apply_gcr_spin_restricted, apply_igcr2_spin_restricted
from xquces.gcr.model import GCRAnsatz, gcr_from_ucj_ansatz
from xquces.ucj.init import UCJBalancedDFSeed, UCJRestrictedProjectedDFSeed
from xquces.ucj.model import SpinBalancedSpec, SpinRestrictedSpec, UCJAnsatz
from xquces.ucj.parameterization import ov_final_unitary


def _symmetric_matrix_from_values(values, norb, pairs):
    out = np.zeros((norb, norb), dtype=np.float64)
    if pairs:
        rows, cols = zip(*pairs)
        vals = np.asarray(values, dtype=np.float64)
        out[rows, cols] = vals
        out[cols, rows] = vals
    return out


def _validate_pairs(pairs, norb, allow_diagonal=False):
    if pairs is None:
        if allow_diagonal:
            return list(itertools.combinations_with_replacement(range(norb), 2))
        return list(itertools.combinations(range(norb), 2))
    out = []
    seen = set()
    for p, q in pairs:
        if not (0 <= p < norb and 0 <= q < norb):
            raise ValueError("interaction pair index out of bounds")
        if p > q:
            raise ValueError("interaction pairs must be upper triangular")
        if not allow_diagonal and p == q:
            raise ValueError("diagonal interaction pairs are not allowed here")
        if (p, q) in seen:
            raise ValueError("interaction pairs must not contain duplicates")
        seen.add((p, q))
        out.append((p, q))
    return out


def _assert_square_matrix(a: np.ndarray, name: str) -> None:
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"{name} must be a square matrix")


def _diag_unitary(phases: np.ndarray) -> np.ndarray:
    return np.diag(np.exp(1j * np.asarray(phases, dtype=np.float64)))


def _orbital_relabeling_unitary(old_for_new: np.ndarray, phases: np.ndarray | None = None) -> np.ndarray:
    old_for_new = np.asarray(old_for_new, dtype=np.int64)
    if old_for_new.ndim != 1:
        raise ValueError("old_for_new must be a one-dimensional permutation")
    norb = old_for_new.shape[0]
    if sorted(old_for_new.tolist()) != list(range(norb)):
        raise ValueError("old_for_new must be a permutation of orbital indices")
    if phases is None:
        phases_arr = np.ones(norb, dtype=np.complex128)
    else:
        phases_arr = np.asarray(phases, dtype=np.complex128)
        if phases_arr.shape != (norb,):
            raise ValueError("phases must have shape (norb,)")
        bad = np.abs(phases_arr) <= 1e-14
        phases_arr = np.where(bad, 1.0 + 0.0j, phases_arr / np.abs(phases_arr))
    relabel = np.zeros((norb, norb), dtype=np.complex128)
    relabel[old_for_new, np.arange(norb)] = phases_arr
    return relabel


def orbital_relabeling_from_overlap(overlap: np.ndarray, nocc: int | None = None, block_diagonal: bool = True) -> tuple[np.ndarray, np.ndarray]:
    overlap = np.asarray(overlap, dtype=np.complex128)
    _assert_square_matrix(overlap, "overlap")
    norb = overlap.shape[0]
    if nocc is None:
        blocks = [(0, norb)]
    else:
        if not (0 <= nocc <= norb):
            raise ValueError("nocc must satisfy 0 <= nocc <= norb")
        blocks = [(0, nocc), (nocc, norb)] if block_diagonal else [(0, norb)]
    old_for_new = np.empty(norb, dtype=np.int64)
    phases = np.ones(norb, dtype=np.complex128)
    for start, stop in blocks:
        if start == stop:
            continue
        sub = overlap[start:stop, start:stop]
        old_rows, new_cols = scipy.optimize.linear_sum_assignment(-np.abs(sub))
        for old_local, new_local in zip(old_rows, new_cols):
            old_idx = start + int(old_local)
            new_idx = start + int(new_local)
            old_for_new[new_idx] = old_idx
            val = overlap[old_idx, new_idx]
            if abs(val) > 1e-14:
                phases[new_idx] = val / abs(val)
    return old_for_new, phases


def _zero_diag_antihermitian_from_parameters(params: np.ndarray, norb: int, pairs: list[tuple[int, int]] | None = None) -> np.ndarray:
    params = np.asarray(params, dtype=np.float64)
    if pairs is None:
        pairs = list(itertools.combinations(range(norb), 2))
    expected = 2 * len(pairs)
    if params.shape != (expected,):
        raise ValueError(f"Expected {(expected,)}, got {params.shape}.")
    out = np.zeros((norb, norb), dtype=np.complex128)
    idx = 0
    for p, q in pairs:
        z = params[idx] + 1j * params[idx + 1]
        idx += 2
        out[p, q] = z
        out[q, p] = -np.conjugate(z)
    return out


def _parameters_from_zero_diag_antihermitian(kappa: np.ndarray, pairs: list[tuple[int, int]] | None = None) -> np.ndarray:
    kappa = np.asarray(kappa, dtype=np.complex128)
    _assert_square_matrix(kappa, "kappa")
    if not np.allclose(kappa.conj().T, -kappa, atol=1e-10):
        raise ValueError("kappa must be antihermitian")
    norb = kappa.shape[0]
    if pairs is None:
        pairs = list(itertools.combinations(range(norb), 2))
    out = np.zeros(2 * len(pairs), dtype=np.float64)
    idx = 0
    for p, q in pairs:
        z = kappa[p, q]
        out[idx] = float(np.real(z))
        out[idx + 1] = float(np.imag(z))
        idx += 2
    return out


def _principal_antihermitian_log(u: np.ndarray) -> np.ndarray:
    kappa = scipy.linalg.logm(u)
    kappa = np.asarray(kappa, dtype=np.complex128)
    return 0.5 * (kappa - kappa.conj().T)


def _left_phase_fix_initial_guesses(u: np.ndarray) -> list[np.ndarray]:
    norb = u.shape[0]
    guesses = [np.zeros(norb, dtype=np.float64)]
    diag = np.diag(u)
    safe_diag = np.where(np.abs(diag) > 1e-14, diag, 1.0 + 0.0j)
    guesses.append(-np.angle(safe_diag))
    col_phases = np.zeros(norb, dtype=np.float64)
    for j in range(norb):
        col = u[:, j]
        idx = int(np.argmax(np.abs(col)))
        val = col[idx]
        if abs(val) > 1e-14:
            col_phases[j] = -np.angle(val)
    guesses.append(col_phases)
    return guesses


def _left_parameters_and_right_phase_from_unitary(u: np.ndarray, pairs: list[tuple[int, int]] | None = None) -> tuple[np.ndarray, np.ndarray]:
    u = np.asarray(u, dtype=np.complex128)
    _assert_square_matrix(u, "u")
    norb = u.shape[0]
    if not np.allclose(u.conj().T @ u, np.eye(norb), atol=1e-10):
        raise ValueError("u must be unitary")
    def residual(column_phase: np.ndarray) -> np.ndarray:
        shifted = u @ _diag_unitary(column_phase)
        kappa = _principal_antihermitian_log(shifted)
        return np.imag(np.diag(kappa))
    best_phase = None
    best_norm = np.inf
    for guess in _left_phase_fix_initial_guesses(u):
        result = scipy.optimize.root(residual, guess, method="hybr", options={"xtol": 1e-11, "maxfev": 2000})
        value = np.linalg.norm(residual(result.x))
        if value < best_norm:
            best_norm = value
            best_phase = np.asarray(result.x, dtype=np.float64)
        if value < 1e-11:
            break
    if best_phase is None or best_norm > 1e-8:
        raise ValueError("could not phase-fix left unitary into zero-diagonal chart")
    shifted = u @ _diag_unitary(best_phase)
    kappa = _principal_antihermitian_log(shifted)
    np.fill_diagonal(kappa, 0.0)
    if pairs is not None:
        allowed = np.zeros(kappa.shape, dtype=bool)
        for p, q in pairs:
            allowed[p, q] = True
            allowed[q, p] = True
        projected = np.linalg.norm(kappa[~allowed]) > 1e-7
        kappa[~allowed] = 0.0
    else:
        projected = False
    if not projected and np.linalg.norm(scipy.linalg.expm(kappa) - shifted) > 1e-7:
        raise ValueError("phase-fixed left unitary is outside the principal zero-diagonal chart")
    return _parameters_from_zero_diag_antihermitian(kappa, pairs=pairs), -best_phase


def _left_unitary_from_parameters(params: np.ndarray, norb: int, pairs: list[tuple[int, int]] | None = None) -> np.ndarray:
    kappa = _zero_diag_antihermitian_from_parameters(params, norb, pairs=pairs)
    return np.asarray(scipy.linalg.expm(kappa), dtype=np.complex128)


def _left_parameters_from_unitary(u: np.ndarray) -> np.ndarray:
    params, _ = _left_parameters_and_right_phase_from_unitary(u)
    return params


@dataclass(frozen=True)
class IGCR2LeftUnitaryChart:
    def n_params(self, norb: int) -> int:
        return norb * (norb - 1)
    def unitary_from_parameters(self, params: np.ndarray, norb: int) -> np.ndarray:
        return _left_unitary_from_parameters(params, norb)
    def parameters_from_unitary(self, u: np.ndarray) -> np.ndarray:
        return _left_parameters_from_unitary(u)
    def parameters_and_right_phase_from_unitary(self, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return _left_parameters_and_right_phase_from_unitary(u)


@dataclass(frozen=True)
class IGCR2BlockDiagLeftUnitaryChart:
    nocc: int
    nvirt: int
    def __post_init__(self):
        if self.nocc < 0 or self.nvirt < 0:
            raise ValueError("nocc and nvirt must be nonnegative")
    @property
    def norb(self) -> int:
        return self.nocc + self.nvirt
    def n_params(self, norb: int | None = None) -> int:
        if norb is not None and norb != self.norb:
            raise ValueError(f"norb={norb} does not match chart norb={self.norb}")
        return self.nocc * (self.nocc - 1) + self.nvirt * (self.nvirt - 1)
    def unitary_from_parameters(self, params: np.ndarray, norb: int | None = None) -> np.ndarray:
        if norb is not None and norb != self.norb:
            raise ValueError(f"norb={norb} does not match chart norb={self.norb}")
        params = np.asarray(params, dtype=np.float64)
        n_oo = self.nocc * (self.nocc - 1)
        n_vv = self.nvirt * (self.nvirt - 1)
        u = np.eye(self.norb, dtype=np.complex128)
        if self.nocc >= 1:
            kappa_oo = _zero_diag_antihermitian_from_parameters(params[:n_oo], self.nocc)
            u[: self.nocc, : self.nocc] = np.asarray(scipy.linalg.expm(kappa_oo), dtype=np.complex128)
        if self.nvirt >= 1:
            kappa_vv = _zero_diag_antihermitian_from_parameters(params[n_oo : n_oo + n_vv], self.nvirt)
            u[self.nocc :, self.nocc :] = np.asarray(scipy.linalg.expm(kappa_vv), dtype=np.complex128)
        return u
    def parameters_and_right_phase_from_unitary(self, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        u = np.asarray(u, dtype=np.complex128)
        if u.shape != (self.norb, self.norb):
            raise ValueError(f"Expected shape {(self.norb, self.norb)}, got {u.shape}.")
        params_parts = []
        phase_parts = []
        for start, size in [(0, self.nocc), (self.nocc, self.nvirt)]:
            if size == 0:
                continue
            block = u[start : start + size, start : start + size]
            if size == 1:
                val = block[0, 0]
                u_block = np.array([[val / abs(val)]], dtype=np.complex128) if abs(val) > 1e-14 else np.eye(1, dtype=np.complex128)
            else:
                u_block, _ = scipy.linalg.polar(block, side="right")
            p, ph = _left_parameters_and_right_phase_from_unitary(u_block)
            params_parts.append(p)
            phase_parts.append(ph)
        params = np.concatenate(params_parts) if params_parts else np.zeros(0, dtype=np.float64)
        right_phase = np.concatenate(phase_parts) if phase_parts else np.zeros(self.norb, dtype=np.float64)
        return params, right_phase
    def parameters_from_unitary(self, u: np.ndarray) -> np.ndarray:
        params, _ = self.parameters_and_right_phase_from_unitary(u)
        return params


@dataclass(frozen=True)
class IGCR2ReferenceOVUnitaryChart:
    nocc: int
    nvirt: int
    def __post_init__(self):
        if self.nocc < 0 or self.nvirt < 0:
            raise ValueError("nocc and nvirt must be nonnegative")
    @property
    def norb(self) -> int:
        return self.nocc + self.nvirt
    def n_params(self, norb: int | None = None) -> int:
        if norb is not None and norb != self.norb:
            raise ValueError("norb does not match chart dimensions")
        return 2 * self.nocc * self.nvirt
    def unitary_from_parameters(self, params: np.ndarray, norb: int | None = None) -> np.ndarray:
        if norb is not None and norb != self.norb:
            raise ValueError("norb does not match chart dimensions")
        return ov_final_unitary(np.asarray(params, dtype=np.float64), self.norb, self.nocc)
    def parameters_from_unitary(self, u: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=np.complex128)
        if u.shape != (self.norb, self.norb):
            raise ValueError("u has wrong shape")
        return exact_reference_ov_params_from_unitary(u, self.nocc)


@dataclass(frozen=True)
class IGCR2RealReferenceOVUnitaryChart:
    nocc: int
    nvirt: int
    def __post_init__(self):
        if self.nocc < 0 or self.nvirt < 0:
            raise ValueError("nocc and nvirt must be nonnegative")
    @property
    def norb(self) -> int:
        return self.nocc + self.nvirt
    def n_params(self, norb: int | None = None) -> int:
        if norb is not None and norb != self.norb:
            raise ValueError("norb does not match chart dimensions")
        return self.nocc * self.nvirt
    def unitary_from_parameters(self, params: np.ndarray, norb: int | None = None) -> np.ndarray:
        if norb is not None and norb != self.norb:
            raise ValueError("norb does not match chart dimensions")
        params = np.asarray(params, dtype=np.float64)
        expected = self.nocc * self.nvirt
        if params.shape != (expected,):
            raise ValueError(f"Expected {(expected,)}, got {params.shape}.")
        full = np.concatenate([params, np.zeros(expected, dtype=np.float64)])
        return ov_final_unitary(full, self.norb, self.nocc)
    def parameters_from_unitary(self, u: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=np.complex128)
        if u.shape != (self.norb, self.norb):
            raise ValueError("u has wrong shape")
        n = self.nocc * self.nvirt
        return exact_reference_ov_params_from_unitary(u, self.nocc)[:n]


def exact_reference_ov_params_from_unitary(u, nocc):
    u = np.asarray(u, dtype=np.complex128)
    norb = u.shape[0]
    nvirt = norb - nocc
    if nocc == 0 or nvirt == 0:
        return np.zeros(0, dtype=np.float64)
    f = u[:, :nocc]
    a = f[:nocc, :]
    x, _ = scipy.linalg.polar(a, side="right")
    fp = f @ x.conj().T
    c = fp[nocc:, :]
    u_left, s, vh = np.linalg.svd(c, full_matrices=False)
    angles = np.arcsin(np.clip(s, -1.0, 1.0))
    z = u_left @ np.diag(angles) @ vh
    return np.concatenate([z.real.ravel(), z.imag.ravel()])


def exact_reference_ov_unitary(u, nocc):
    params = exact_reference_ov_params_from_unitary(u, nocc)
    return ov_final_unitary(params, u.shape[0], nocc)


def _right_unitary_from_left_and_final(left: np.ndarray, final: np.ndarray, nocc: int) -> np.ndarray:
    del nocc
    return np.asarray(left, dtype=np.complex128).conj().T @ final


def _final_unitary_from_left_and_right(left: np.ndarray, right: np.ndarray, nocc: int) -> np.ndarray:
    return exact_reference_ov_unitary(np.asarray(left, dtype=np.complex128) @ right, nocc)


def _left_right_ov_parameter_indices(norb: int, nocc: int, right_start: int):
    nvirt = norb - nocc
    if nocc == 0 or nvirt == 0:
        return []
    pair_to_idx = {pair: k for k, pair in enumerate(itertools.combinations(range(norb), 2))}
    n_right_complex = nocc * nvirt
    out = []
    for a in range(nvirt):
        q = nocc + a
        for i in range(nocc):
            k = pair_to_idx[(i, q)]
            out.append((2 * k, right_start + a * nocc + i))
            out.append((2 * k + 1, right_start + n_right_complex + a * nocc + i))
    return out


def _left_right_ov_adapted_to_native(params: np.ndarray, norb: int, nocc: int, right_start: int, relative_scale: float | None) -> np.ndarray:
    out = np.array(params, copy=True, dtype=np.float64)
    if relative_scale is None:
        return out
    scale = float(relative_scale)
    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    for left_idx, right_idx in _left_right_ov_parameter_indices(norb, nocc, right_start):
        co_rotating = out[left_idx]
        relative = out[right_idx]
        out[left_idx] = inv_sqrt2 * (co_rotating + scale * relative)
        out[right_idx] = inv_sqrt2 * (co_rotating - scale * relative)
    return out


def _native_to_left_right_ov_adapted(params: np.ndarray, norb: int, nocc: int, right_start: int, relative_scale: float | None) -> np.ndarray:
    out = np.array(params, copy=True, dtype=np.float64)
    if relative_scale is None:
        return out
    scale = float(relative_scale)
    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    for left_idx, right_idx in _left_right_ov_parameter_indices(norb, nocc, right_start):
        left = out[left_idx]
        right = out[right_idx]
        out[left_idx] = inv_sqrt2 * (left + right)
        out[right_idx] = inv_sqrt2 * (left - right) / scale
    return out


def _n_total_from_nocc(nocc: int) -> int:
    return 2 * int(nocc)


def _restricted_irreducible_pair_matrix(double_params: np.ndarray, pair_params: np.ndarray) -> np.ndarray:
    b = np.asarray(double_params, dtype=np.float64)
    pair = np.asarray(pair_params, dtype=np.float64)
    shift = 0.5 * (b[:, None] + b[None, :])
    out = np.array(pair, copy=True, dtype=np.float64)
    mask = ~np.eye(pair.shape[0], dtype=bool)
    out[mask] -= shift[mask]
    np.fill_diagonal(out, 0.0)
    return out


def _restricted_left_phase_vector(double_params: np.ndarray, nocc: int) -> np.ndarray:
    return 0.5 * (_n_total_from_nocc(nocc) - 1) * np.asarray(double_params, dtype=np.float64)


def _balanced_irreducible_pair_matrices(same_spin_params: np.ndarray, mixed_spin_params: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    same = np.asarray(same_spin_params, dtype=np.float64)
    mixed = np.asarray(mixed_spin_params, dtype=np.float64)
    b = np.diag(mixed)
    shift = 0.5 * (b[:, None] + b[None, :])
    same_red = np.array(same, copy=True, dtype=np.float64)
    mixed_red = np.array(mixed, copy=True, dtype=np.float64)
    mask = ~np.eye(same.shape[0], dtype=bool)
    same_red[mask] -= shift[mask]
    mixed_red[mask] -= shift[mask]
    np.fill_diagonal(same_red, 0.0)
    np.fill_diagonal(mixed_red, 0.0)
    return same_red, mixed_red


def _balanced_left_phase_vector(same_spin_params: np.ndarray, mixed_spin_params: np.ndarray, nocc: int) -> np.ndarray:
    same_diag = np.diag(np.asarray(same_spin_params, dtype=np.float64))
    mixed_diag = np.diag(np.asarray(mixed_spin_params, dtype=np.float64))
    return 0.5 * same_diag + 0.5 * (_n_total_from_nocc(nocc) - 1) * mixed_diag


@dataclass(frozen=True)
class IGCR2SpinRestrictedSpec:
    pair: np.ndarray
    @property
    def norb(self):
        return self.pair.shape[0]
    def full_double(self):
        return np.zeros(self.norb, dtype=np.float64)
    def to_standard(self):
        pair = np.array(self.pair, copy=True, dtype=np.float64)
        np.fill_diagonal(pair, 0.0)
        return SpinRestrictedSpec(double_params=self.full_double(), pair_params=pair)


@dataclass(frozen=True)
class IGCR2SpinBalancedSpec:
    same_diag: np.ndarray
    same: np.ndarray
    mixed: np.ndarray
    double: np.ndarray
    @property
    def norb(self):
        return int(np.asarray(self.double, dtype=np.float64).shape[0])
    def full_double(self):
        double = np.asarray(self.double, dtype=np.float64)
        if double.shape != (self.norb,):
            raise ValueError("double has inconsistent shape")
        return double
    def to_standard(self):
        same = np.array(self.same, copy=True, dtype=np.float64)
        mixed = np.array(self.mixed, copy=True, dtype=np.float64)
        np.fill_diagonal(same, np.asarray(self.same_diag, dtype=np.float64))
        np.fill_diagonal(mixed, self.full_double())
        return SpinBalancedSpec(same_spin_params=same, mixed_spin_params=mixed)


def reduce_spin_restricted(diag: SpinRestrictedSpec):
    pair = np.asarray(diag.pair_params, dtype=np.float64).copy()
    b = np.asarray(diag.double_params, dtype=np.float64)
    shift = 0.5 * (b[:, None] + b[None, :])
    mask = ~np.eye(pair.shape[0], dtype=bool)
    pair[mask] -= shift[mask]
    np.fill_diagonal(pair, 0.0)
    return IGCR2SpinRestrictedSpec(pair=pair)


def reduce_spin_balanced(diag: SpinBalancedSpec):
    same = np.asarray(diag.same_spin_params, dtype=np.float64).copy()
    mixed = np.asarray(diag.mixed_spin_params, dtype=np.float64).copy()
    same_diag = np.diag(same).copy()
    double = np.diag(mixed).copy()
    np.fill_diagonal(same, 0.0)
    np.fill_diagonal(mixed, 0.0)
    return IGCR2SpinBalancedSpec(same_diag=same_diag, same=same, mixed=mixed, double=double)


@dataclass(frozen=True)
class IGCR2Ansatz:
    diagonal: IGCR2SpinRestrictedSpec | IGCR2SpinBalancedSpec
    left: np.ndarray
    right: np.ndarray
    nocc: int
    @property
    def norb(self):
        return self.diagonal.norb
    @property
    def is_spin_restricted(self):
        return isinstance(self.diagonal, IGCR2SpinRestrictedSpec)
    @property
    def is_spin_balanced(self):
        return isinstance(self.diagonal, IGCR2SpinBalancedSpec)
    def apply(self, vec, nelec, copy=True):
        if self.is_spin_restricted:
            return apply_igcr2_spin_restricted(vec, self.diagonal.pair, self.norb, nelec, left_orbital_rotation=self.left, right_orbital_rotation=self.right, copy=copy)
        d = self.diagonal.to_standard()
        return apply_gcr_spin_balanced(vec, d.same_spin_params, d.mixed_spin_params, self.norb, nelec, left_orbital_rotation=self.left, right_orbital_rotation=self.right, copy=copy)
    @classmethod
    def from_gcr_ansatz(cls, ansatz: GCRAnsatz, nocc: int):
        right_ov = exact_reference_ov_unitary(ansatz.right_orbital_rotation, nocc)
        if ansatz.is_spin_restricted:
            diag = reduce_spin_restricted(ansatz.diagonal)
            b = np.asarray(ansatz.diagonal.double_params, dtype=np.float64)
            phase_vec = _restricted_left_phase_vector(b, nocc)
            left = np.asarray(ansatz.left_orbital_rotation, dtype=np.complex128) @ _diag_unitary(phase_vec)
        else:
            diag = reduce_spin_balanced(ansatz.diagonal)
            left = np.asarray(ansatz.left_orbital_rotation, dtype=np.complex128)
        return cls(diagonal=diag, left=left, right=np.asarray(right_ov, dtype=np.complex128), nocc=nocc)
    @classmethod
    def from_ucj(cls, ucj: UCJAnsatz, nocc: int):
        gcr = gcr_from_ucj_ansatz(ucj)
        return cls.from_gcr_ansatz(gcr, nocc=nocc)
    @classmethod
    def from_ucj_ansatz(cls, ansatz: UCJAnsatz, nocc: int):
        return cls.from_ucj(ansatz, nocc=nocc)
    @classmethod
    def from_t_balanced(cls, t2, **kwargs):
        ucj = UCJBalancedDFSeed(t2=t2, **kwargs).build_ansatz()
        return cls.from_ucj(ucj, nocc=t2.shape[0])
    @classmethod
    def from_t_restricted(cls, t2, **kwargs):
        ucj = UCJRestrictedProjectedDFSeed(t2=t2, **kwargs).build_ansatz()
        return cls.from_ucj(ucj, nocc=t2.shape[0])


def relabel_igcr2_ansatz_orbitals(ansatz: IGCR2Ansatz, old_for_new: np.ndarray, phases: np.ndarray | None = None) -> IGCR2Ansatz:
    if ansatz.norb != len(old_for_new):
        raise ValueError("orbital permutation length must match ansatz.norb")
    relabel = _orbital_relabeling_unitary(old_for_new, phases)
    old_for_new = np.asarray(old_for_new, dtype=np.int64)
    if ansatz.is_spin_restricted:
        pair = ansatz.diagonal.pair[np.ix_(old_for_new, old_for_new)]
        diagonal = IGCR2SpinRestrictedSpec(pair=pair)
    else:
        d = ansatz.diagonal.to_standard()
        diag = SpinBalancedSpec(same_spin_params=d.same_spin_params[np.ix_(old_for_new, old_for_new)], mixed_spin_params=d.mixed_spin_params[np.ix_(old_for_new, old_for_new)])
        diagonal = reduce_spin_balanced(diag)
    return IGCR2Ansatz(diagonal=diagonal, left=relabel.conj().T @ ansatz.left @ relabel, right=relabel.conj().T @ ansatz.right @ relabel, nocc=ansatz.nocc)


@dataclass(frozen=True)
class IGCR2SpinRestrictedParameterization:
    norb: int
    nocc: int
    interaction_pairs: list[tuple[int, int]] | None = None
    left_orbital_chart: object = field(default_factory=IGCR2LeftUnitaryChart)
    right_orbital_chart_override: object | None = None
    real_right_orbital_chart: bool = False
    left_right_ov_relative_scale: float | None = 1.0
    def __post_init__(self):
        if not (0 <= self.nocc <= self.norb):
            raise ValueError("nocc must satisfy 0 <= nocc <= norb")
        _validate_pairs(self.interaction_pairs, self.norb, allow_diagonal=False)
        if self.left_right_ov_relative_scale is not None and (not np.isfinite(float(self.left_right_ov_relative_scale)) or self.left_right_ov_relative_scale <= 0):
            raise ValueError("left_right_ov_relative_scale must be positive or None")
    @property
    def pair_indices(self):
        return _validate_pairs(self.interaction_pairs, self.norb, allow_diagonal=False)
    @property
    def right_orbital_chart(self):
        if self.right_orbital_chart_override is not None:
            return self.right_orbital_chart_override
        if self.real_right_orbital_chart:
            return IGCR2RealReferenceOVUnitaryChart(self.nocc, self.norb - self.nocc)
        return IGCR2ReferenceOVUnitaryChart(self.nocc, self.norb - self.nocc)
    @property
    def _left_orbital_chart(self):
        return self.left_orbital_chart
    @property
    def n_left_orbital_rotation_params(self):
        return self._left_orbital_chart.n_params(self.norb)
    @property
    def n_double_params(self):
        return 0
    @property
    def n_pair_params(self):
        return len(self.pair_indices)
    @property
    def n_right_orbital_rotation_params(self):
        return self.right_orbital_chart.n_params(self.norb)
    @property
    def _right_orbital_rotation_start(self):
        return self.n_left_orbital_rotation_params + self.n_pair_params
    @property
    def _left_right_ov_transform_scale(self):
        return None
    def _native_parameters_from_public(self, params: np.ndarray) -> np.ndarray:
        return _left_right_ov_adapted_to_native(params, self.norb, self.nocc, self._right_orbital_rotation_start, self._left_right_ov_transform_scale)
    def _public_parameters_from_native(self, params: np.ndarray) -> np.ndarray:
        return _native_to_left_right_ov_adapted(params, self.norb, self.nocc, self._right_orbital_rotation_start, self._left_right_ov_transform_scale)
    @property
    def n_params(self):
        return self.n_left_orbital_rotation_params + self.n_pair_params + self.n_right_orbital_rotation_params
    def ansatz_from_parameters(self, params: np.ndarray):
        params = np.asarray(params, dtype=np.float64)
        if params.shape != (self.n_params,):
            raise ValueError(f"Expected {(self.n_params,)}, got {params.shape}.")
        params = self._native_parameters_from_public(params)
        idx = 0
        n = self.n_left_orbital_rotation_params
        left = self._left_orbital_chart.unitary_from_parameters(params[idx:idx + n], self.norb)
        idx += n
        n = self.n_pair_params
        pair = _symmetric_matrix_from_values(params[idx:idx + n], self.norb, self.pair_indices)
        idx += n
        n = self.n_right_orbital_rotation_params
        final = self.right_orbital_chart.unitary_from_parameters(params[idx:idx + n], self.norb)
        right = _right_unitary_from_left_and_final(left, final, self.nocc)
        return IGCR2Ansatz(diagonal=IGCR2SpinRestrictedSpec(pair=pair), left=left, right=right, nocc=self.nocc)
    def parameters_from_ansatz(self, ansatz: IGCR2Ansatz):
        if ansatz.norb != self.norb:
            raise ValueError("ansatz norb does not match parameterization")
        if not ansatz.is_spin_restricted:
            raise TypeError("expected a spin-restricted ansatz")
        left_chart = self._left_orbital_chart
        if hasattr(left_chart, "parameters_and_right_phase_from_unitary"):
            left_params, right_phase = left_chart.parameters_and_right_phase_from_unitary(np.asarray(ansatz.left, dtype=np.complex128))
        else:
            left_params = left_chart.parameters_from_unitary(np.asarray(ansatz.left, dtype=np.complex128))
            right_phase = np.zeros(self.norb, dtype=np.float64)
        pair_eff = ansatz.diagonal.pair
        right_eff = _diag_unitary(right_phase) @ np.asarray(ansatz.right, dtype=np.complex128)
        out = np.zeros(self.n_params, dtype=np.float64)
        idx = 0
        n = self.n_left_orbital_rotation_params
        out[idx:idx + n] = left_params
        idx += n
        n = self.n_pair_params
        out[idx:idx + n] = np.asarray([pair_eff[p, q] for p, q in self.pair_indices], dtype=np.float64)
        idx += n
        n = self.n_right_orbital_rotation_params
        left_param_unitary = self._left_orbital_chart.unitary_from_parameters(left_params, self.norb)
        final_eff = _final_unitary_from_left_and_right(left_param_unitary, right_eff, self.nocc)
        out[idx:idx + n] = self.right_orbital_chart.parameters_from_unitary(final_eff)
        return self._public_parameters_from_native(out)
    def parameters_from_ucj_ansatz(self, ansatz: UCJAnsatz):
        return self.parameters_from_ansatz(IGCR2Ansatz.from_ucj_ansatz(ansatz, self.nocc))
    def transfer_parameters_from(self, previous_parameters: np.ndarray, previous_parameterization: "IGCR2SpinRestrictedParameterization | None" = None, old_for_new: np.ndarray | None = None, phases: np.ndarray | None = None, orbital_overlap: np.ndarray | None = None, block_diagonal: bool = True) -> np.ndarray:
        if previous_parameterization is None:
            previous_parameterization = self
        ansatz = previous_parameterization.ansatz_from_parameters(previous_parameters)
        if ansatz.nocc != self.nocc:
            raise ValueError("previous ansatz nocc does not match this parameterization")
        if orbital_overlap is not None:
            if old_for_new is not None or phases is not None:
                raise ValueError("Pass either orbital_overlap or explicit relabeling, not both.")
            old_for_new, phases = orbital_relabeling_from_overlap(orbital_overlap, nocc=self.nocc, block_diagonal=block_diagonal)
        if old_for_new is not None:
            ansatz = relabel_igcr2_ansatz_orbitals(ansatz, old_for_new, phases)
        return self.parameters_from_ansatz(ansatz)
    def params_to_vec(self, reference_vec: np.ndarray, nelec: tuple[int, int]) -> Callable[[np.ndarray], np.ndarray]:
        reference_vec = np.asarray(reference_vec, dtype=np.complex128)
        def func(params: np.ndarray) -> np.ndarray:
            return self.ansatz_from_parameters(params).apply(reference_vec, nelec=nelec, copy=True)
        return func


@dataclass(frozen=True)
class IGCR2SpinBalancedParameterization:
    norb: int
    nocc: int
    same_spin_interaction_pairs: list[tuple[int, int]] | None = None
    mixed_spin_interaction_pairs: list[tuple[int, int]] | None = None
    left_orbital_chart: object = field(default_factory=IGCR2LeftUnitaryChart)
    left_right_ov_relative_scale: float | None = 3.0
    def __post_init__(self):
        if not (0 <= self.nocc <= self.norb):
            raise ValueError("nocc must satisfy 0 <= nocc <= norb")
        _validate_pairs(self.same_spin_interaction_pairs, self.norb, allow_diagonal=False)
        _validate_pairs(self.mixed_spin_interaction_pairs, self.norb, allow_diagonal=False)
        if self.left_right_ov_relative_scale is not None and (not np.isfinite(float(self.left_right_ov_relative_scale)) or self.left_right_ov_relative_scale <= 0):
            raise ValueError("left_right_ov_relative_scale must be positive or None")
    @property
    def same_spin_indices(self):
        return _validate_pairs(self.same_spin_interaction_pairs, self.norb, allow_diagonal=False)
    @property
    def mixed_spin_indices(self):
        return _validate_pairs(self.mixed_spin_interaction_pairs, self.norb, allow_diagonal=False)
    @property
    def right_orbital_chart(self):
        return IGCR2ReferenceOVUnitaryChart(self.nocc, self.norb - self.nocc)
    @property
    def _left_orbital_chart(self):
        return self.left_orbital_chart
    @property
    def n_left_orbital_rotation_params(self):
        return self._left_orbital_chart.n_params(self.norb)
    @property
    def n_same_diag_params(self):
        return self.norb
    @property
    def n_double_params(self):
        return self.norb
    @property
    def n_same_spin_params(self):
        return len(self.same_spin_indices)
    @property
    def n_mixed_spin_params(self):
        return len(self.mixed_spin_indices)
    @property
    def n_right_orbital_rotation_params(self):
        return self.right_orbital_chart.n_params(self.norb)
    @property
    def _right_orbital_rotation_start(self):
        return self.n_left_orbital_rotation_params + self.n_same_diag_params + self.n_double_params + self.n_same_spin_params + self.n_mixed_spin_params
    @property
    def _left_right_ov_transform_scale(self):
        return None
    def _native_parameters_from_public(self, params: np.ndarray) -> np.ndarray:
        return _left_right_ov_adapted_to_native(params, self.norb, self.nocc, self._right_orbital_rotation_start, self._left_right_ov_transform_scale)
    def _public_parameters_from_native(self, params: np.ndarray) -> np.ndarray:
        return _native_to_left_right_ov_adapted(params, self.norb, self.nocc, self._right_orbital_rotation_start, self._left_right_ov_transform_scale)
    @property
    def n_params(self):
        return self.n_left_orbital_rotation_params + self.n_same_diag_params + self.n_double_params + self.n_same_spin_params + self.n_mixed_spin_params + self.n_right_orbital_rotation_params
    def ansatz_from_parameters(self, params: np.ndarray):
        params = np.asarray(params, dtype=np.float64)
        if params.shape != (self.n_params,):
            raise ValueError(f"Expected {(self.n_params,)}, got {params.shape}.")
        params = self._native_parameters_from_public(params)
        idx = 0
        n = self.n_left_orbital_rotation_params
        left = self._left_orbital_chart.unitary_from_parameters(params[idx:idx + n], self.norb)
        idx += n
        same_diag = np.asarray(params[idx:idx + self.n_same_diag_params], dtype=np.float64)
        idx += self.n_same_diag_params
        double = np.asarray(params[idx:idx + self.n_double_params], dtype=np.float64)
        idx += self.n_double_params
        same = _symmetric_matrix_from_values(np.asarray(params[idx:idx + self.n_same_spin_params], dtype=np.float64), self.norb, self.same_spin_indices)
        idx += self.n_same_spin_params
        mixed = _symmetric_matrix_from_values(np.asarray(params[idx:idx + self.n_mixed_spin_params], dtype=np.float64), self.norb, self.mixed_spin_indices)
        idx += self.n_mixed_spin_params
        n = self.n_right_orbital_rotation_params
        final = self.right_orbital_chart.unitary_from_parameters(params[idx:idx + n], self.norb)
        right = _right_unitary_from_left_and_final(left, final, self.nocc)
        return IGCR2Ansatz(diagonal=IGCR2SpinBalancedSpec(same_diag=same_diag, same=same, mixed=mixed, double=double), left=left, right=right, nocc=self.nocc)
    def parameters_from_ansatz(self, ansatz: IGCR2Ansatz):
        if ansatz.norb != self.norb:
            raise ValueError("ansatz norb does not match parameterization")
        if not ansatz.is_spin_balanced:
            raise TypeError("expected a spin-balanced ansatz")
        d = ansatz.diagonal.to_standard()
        same_mat = np.asarray(d.same_spin_params, dtype=np.float64).copy()
        mixed_mat = np.asarray(d.mixed_spin_params, dtype=np.float64).copy()
        same_diag = np.diag(same_mat).copy()
        mixed_double = np.diag(mixed_mat).copy()
        np.fill_diagonal(same_mat, 0.0)
        np.fill_diagonal(mixed_mat, 0.0)
        same_full = np.asarray([same_mat[p, q] for p, q in self.same_spin_indices], dtype=np.float64)
        mixed_full = np.asarray([mixed_mat[p, q] for p, q in self.mixed_spin_indices], dtype=np.float64)
        left_chart = self._left_orbital_chart
        if hasattr(left_chart, "parameters_and_right_phase_from_unitary"):
            left_params, right_phase = left_chart.parameters_and_right_phase_from_unitary(np.asarray(ansatz.left, dtype=np.complex128))
        else:
            left_params = left_chart.parameters_from_unitary(np.asarray(ansatz.left, dtype=np.complex128))
            right_phase = np.zeros(self.norb, dtype=np.float64)
        right_eff = _diag_unitary(right_phase) @ np.asarray(ansatz.right, dtype=np.complex128)
        out = np.zeros(self.n_params, dtype=np.float64)
        idx = 0
        out[idx:idx + self.n_left_orbital_rotation_params] = left_params
        idx += self.n_left_orbital_rotation_params
        out[idx:idx + self.n_same_diag_params] = same_diag
        idx += self.n_same_diag_params
        out[idx:idx + self.n_double_params] = mixed_double
        idx += self.n_double_params
        out[idx:idx + self.n_same_spin_params] = same_full
        idx += self.n_same_spin_params
        out[idx:idx + self.n_mixed_spin_params] = mixed_full
        idx += self.n_mixed_spin_params
        n = self.n_right_orbital_rotation_params
        left_param_unitary = self._left_orbital_chart.unitary_from_parameters(left_params, self.norb)
        final_eff = _final_unitary_from_left_and_right(left_param_unitary, right_eff, self.nocc)
        out[idx:idx + n] = self.right_orbital_chart.parameters_from_unitary(final_eff)
        return self._public_parameters_from_native(out)
    def parameters_from_ucj_ansatz(self, ansatz: UCJAnsatz):
        return self.parameters_from_ansatz(IGCR2Ansatz.from_ucj_ansatz(ansatz, self.nocc))
    def transfer_parameters_from(self, previous_parameters: np.ndarray, previous_parameterization: "IGCR2SpinBalancedParameterization | None" = None, old_for_new: np.ndarray | None = None, phases: np.ndarray | None = None, orbital_overlap: np.ndarray | None = None, block_diagonal: bool = True) -> np.ndarray:
        if previous_parameterization is None:
            previous_parameterization = self
        ansatz = previous_parameterization.ansatz_from_parameters(previous_parameters)
        if ansatz.nocc != self.nocc:
            raise ValueError("previous ansatz nocc does not match this parameterization")
        if orbital_overlap is not None:
            if old_for_new is not None or phases is not None:
                raise ValueError("Pass either orbital_overlap or explicit relabeling, not both.")
            old_for_new, phases = orbital_relabeling_from_overlap(orbital_overlap, nocc=self.nocc, block_diagonal=block_diagonal)
        if old_for_new is not None:
            ansatz = relabel_igcr2_ansatz_orbitals(ansatz, old_for_new, phases)
        return self.parameters_from_ansatz(ansatz)
    def params_to_vec(self, reference_vec: np.ndarray, nelec: tuple[int, int]) -> Callable[[np.ndarray], np.ndarray]:
        reference_vec = np.asarray(reference_vec, dtype=np.complex128)
        def func(params: np.ndarray) -> np.ndarray:
            return self.ansatz_from_parameters(params).apply(reference_vec, nelec=nelec, copy=True)
        return func
