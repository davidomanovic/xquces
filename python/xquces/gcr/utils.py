from __future__ import annotations

import itertools

import numpy as np
import scipy.linalg
import scipy.optimize

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


def _orbital_relabeling_unitary(
    old_for_new: np.ndarray, phases: np.ndarray | None = None
) -> np.ndarray:
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


def orbital_relabeling_from_overlap(
    overlap: np.ndarray, nocc: int | None = None, block_diagonal: bool = True
) -> tuple[np.ndarray, np.ndarray]:
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


def orbital_transport_unitary_from_overlap(
    overlap: np.ndarray,
    nocc: int | None = None,
    block_diagonal: bool = False,
) -> np.ndarray:
    overlap = np.asarray(overlap, dtype=np.complex128)
    _assert_square_matrix(overlap, "overlap")
    if block_diagonal:
        norb = overlap.shape[0]
        if nocc is None:
            raise ValueError("nocc is required for block-diagonal orbital transport")
        if not (0 <= nocc <= norb):
            raise ValueError("nocc must satisfy 0 <= nocc <= norb")
        out = np.zeros_like(overlap)
        for start, stop in ((0, nocc), (nocc, norb)):
            if start == stop:
                continue
            sub = overlap[start:stop, start:stop]
            u, _, vh = np.linalg.svd(sub)
            out[start:stop, start:stop] = u @ vh
        return out
    u, _, vh = np.linalg.svd(overlap)
    return u @ vh


def _zero_diag_antihermitian_from_parameters(
    params: np.ndarray, norb: int, pairs: list[tuple[int, int]] | None = None
) -> np.ndarray:
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


def _parameters_from_zero_diag_antihermitian(
    kappa: np.ndarray, pairs: list[tuple[int, int]] | None = None
) -> np.ndarray:
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


def _wrap_phases(phases: np.ndarray) -> np.ndarray:
    return (np.asarray(phases, dtype=np.float64) + np.pi) % (2.0 * np.pi) - np.pi


def _left_phase_fix_initial_guesses(u: np.ndarray) -> list[np.ndarray]:
    norb = u.shape[0]
    bases = [np.zeros(norb, dtype=np.float64)]
    diag = np.diag(u)
    safe_diag = np.where(np.abs(diag) > 1e-14, diag, 1.0 + 0.0j)
    bases.append(-np.angle(safe_diag))
    col_phases = np.zeros(norb, dtype=np.float64)
    for j in range(norb):
        col = u[:, j]
        idx = int(np.argmax(np.abs(col)))
        val = col[idx]
        if abs(val) > 1e-14:
            col_phases[j] = -np.angle(val)
    bases.append(col_phases)

    det_shift = -np.angle(np.linalg.det(u)) / max(norb, 1)
    shifts = (0.0, det_shift, 0.5 * np.pi, -0.5 * np.pi, np.pi)
    guesses = []
    seen = set()
    for base in bases:
        for shift in shifts:
            guess = _wrap_phases(base + shift)
            key = tuple(np.round(guess, decimals=12))
            if key not in seen:
                seen.add(key)
                guesses.append(guess)
    return guesses


def _left_parameters_and_right_phase_from_unitary(
    u: np.ndarray, pairs: list[tuple[int, int]] | None = None
) -> tuple[np.ndarray, np.ndarray]:
    u = np.asarray(u, dtype=np.complex128)
    _assert_square_matrix(u, "u")
    norb = u.shape[0]
    if not np.allclose(u.conj().T @ u, np.eye(norb), atol=1e-10):
        raise ValueError("u must be unitary")

    def residual(column_phase: np.ndarray) -> np.ndarray:
        shifted = u @ _diag_unitary(_wrap_phases(column_phase))
        kappa = _principal_antihermitian_log(shifted)
        return np.imag(np.diag(kappa))

    best_phase = None
    best_norm = np.inf
    guesses = _left_phase_fix_initial_guesses(u)

    def update_best(phase: np.ndarray) -> float:
        nonlocal best_phase, best_norm
        phase = _wrap_phases(phase)
        value = float(np.linalg.norm(residual(phase)))
        if value < best_norm:
            best_norm = value
            best_phase = phase
        return value

    for guess in guesses:
        result = scipy.optimize.root(
            residual, guess, method="hybr", options={"xtol": 1e-11, "maxfev": 2000}
        )
        value = update_best(result.x)
        if value < 1e-11:
            break
    if best_norm > 1e-8:
        for guess in guesses:
            result = scipy.optimize.least_squares(
                residual,
                guess,
                bounds=(-np.pi, np.pi),
                xtol=1e-12,
                ftol=1e-12,
                gtol=1e-12,
                max_nfev=4000,
            )
            value = update_best(result.x)
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
        raise ValueError(
            "phase-fixed left unitary is outside the principal zero-diagonal chart"
        )
    return _parameters_from_zero_diag_antihermitian(kappa, pairs=pairs), -best_phase


def _left_unitary_from_parameters(
    params: np.ndarray, norb: int, pairs: list[tuple[int, int]] | None = None
) -> np.ndarray:
    kappa = _zero_diag_antihermitian_from_parameters(params, norb, pairs=pairs)
    return np.asarray(scipy.linalg.expm(kappa), dtype=np.complex128)


def _left_parameters_from_unitary(u: np.ndarray) -> np.ndarray:
    params, _ = _left_parameters_and_right_phase_from_unitary(u)
    return params

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


def _right_unitary_from_left_and_final(
    left: np.ndarray, final: np.ndarray, nocc: int
) -> np.ndarray:
    del nocc
    return np.asarray(left, dtype=np.complex128).conj().T @ final


def _final_unitary_from_left_and_right(
    left: np.ndarray,
    right: np.ndarray,
    nocc: int,
    *,
    project_reference_ov: bool = True,
) -> np.ndarray:
    final = np.asarray(left, dtype=np.complex128) @ np.asarray(
        right, dtype=np.complex128
    )
    if not project_reference_ov:
        return final
    return exact_reference_ov_unitary(final, nocc)


def _left_right_ov_parameter_indices(norb: int, nocc: int, right_start: int):
    nvirt = norb - nocc
    if nocc == 0 or nvirt == 0:
        return []
    pair_to_idx = {
        pair: k for k, pair in enumerate(itertools.combinations(range(norb), 2))
    }
    n_right_complex = nocc * nvirt
    out = []
    for a in range(nvirt):
        q = nocc + a
        for i in range(nocc):
            k = pair_to_idx[(i, q)]
            out.append((2 * k, right_start + a * nocc + i))
            out.append((2 * k + 1, right_start + n_right_complex + a * nocc + i))
    return out


def _left_right_ov_adapted_to_native(
    params: np.ndarray,
    norb: int,
    nocc: int,
    right_start: int,
    relative_scale: float | None,
) -> np.ndarray:
    out = np.array(params, copy=True, dtype=np.float64)
    if relative_scale is None:
        return out
    scale = float(relative_scale)
    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    for left_idx, right_idx in _left_right_ov_parameter_indices(
        norb, nocc, right_start
    ):
        co_rotating = out[left_idx]
        relative = out[right_idx]
        out[left_idx] = inv_sqrt2 * (co_rotating + scale * relative)
        out[right_idx] = inv_sqrt2 * (co_rotating - scale * relative)
    return out


def _native_to_left_right_ov_adapted(
    params: np.ndarray,
    norb: int,
    nocc: int,
    right_start: int,
    relative_scale: float | None,
) -> np.ndarray:
    out = np.array(params, copy=True, dtype=np.float64)
    if relative_scale is None:
        return out
    scale = float(relative_scale)
    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    for left_idx, right_idx in _left_right_ov_parameter_indices(
        norb, nocc, right_start
    ):
        left = out[left_idx]
        right = out[right_idx]
        out[left_idx] = inv_sqrt2 * (left + right)
        out[right_idx] = inv_sqrt2 * (left - right) / scale
    return out


def _n_total_from_nocc(nocc: int) -> int:
    return 2 * int(nocc)


def _restricted_irreducible_pair_matrix(
    double_params: np.ndarray, pair_params: np.ndarray
) -> np.ndarray:
    b = np.asarray(double_params, dtype=np.float64)
    pair = np.asarray(pair_params, dtype=np.float64)
    shift = 0.5 * (b[:, None] + b[None, :])
    out = np.array(pair, copy=True, dtype=np.float64)
    mask = ~np.eye(pair.shape[0], dtype=bool)
    out[mask] -= shift[mask]
    np.fill_diagonal(out, 0.0)
    return out


def _restricted_left_phase_vector(double_params: np.ndarray, nocc: int) -> np.ndarray:
    return (
        0.5
        * (_n_total_from_nocc(nocc) - 1)
        * np.asarray(double_params, dtype=np.float64)
    )


def _balanced_irreducible_pair_matrices(
    same_spin_params: np.ndarray, mixed_spin_params: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
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


def _balanced_left_phase_vector(
    same_spin_params: np.ndarray, mixed_spin_params: np.ndarray, nocc: int
) -> np.ndarray:
    same_diag = np.diag(np.asarray(same_spin_params, dtype=np.float64))
    mixed_diag = np.diag(np.asarray(mixed_spin_params, dtype=np.float64))
    return 0.5 * same_diag + 0.5 * (_n_total_from_nocc(nocc) - 1) * mixed_diag

def _default_tau_indices(norb: int) -> list[tuple[int, int]]:
    return [(p, q) for p in range(norb) for q in range(norb) if p != q]


def _default_triple_indices(norb: int) -> list[tuple[int, int, int]]:
    return list(itertools.combinations(range(norb), 3))


def _default_pair_indices(norb: int) -> list[tuple[int, int]]:
    return list(itertools.combinations(range(norb), 2))


def _validate_ordered_pairs(
    pairs: list[tuple[int, int]] | None,
    norb: int,
) -> list[tuple[int, int]]:
    if pairs is None:
        return _default_tau_indices(norb)
    out = []
    seen = set()
    for p, q in pairs:
        if not (0 <= p < norb and 0 <= q < norb):
            raise ValueError("ordered-pair index out of bounds")
        if p == q:
            raise ValueError("ordered-pair diagonal entries are not allowed")
        if (p, q) in seen:
            raise ValueError("ordered pairs must not contain duplicates")
        seen.add((p, q))
        out.append((p, q))
    return out


def _validate_triples(
    triples: list[tuple[int, int, int]] | None,
    norb: int,
) -> list[tuple[int, int, int]]:
    if triples is None:
        return _default_triple_indices(norb)
    out = []
    seen = set()
    for p, q, r in triples:
        if not (0 <= p < q < r < norb):
            raise ValueError("triple indices must satisfy 0 <= p < q < r < norb")
        if (p, q, r) in seen:
            raise ValueError("triple indices must not contain duplicates")
        seen.add((p, q, r))
        out.append((p, q, r))
    return out


def _ordered_matrix_from_values(
    values: np.ndarray,
    norb: int,
    pairs: list[tuple[int, int]],
) -> np.ndarray:
    out = np.zeros((norb, norb), dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (len(pairs),):
        raise ValueError(f"Expected {(len(pairs),)}, got {values.shape}.")
    for value, (p, q) in zip(values, pairs):
        out[p, q] = value
    np.fill_diagonal(out, 0.0)
    return out


def _values_from_ordered_matrix(
    mat: np.ndarray,
    pairs: list[tuple[int, int]],
) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.float64)
    return np.asarray([mat[p, q] for p, q in pairs], dtype=np.float64)

def _default_eta_indices(norb: int) -> list[tuple[int, int]]:
    return list(itertools.combinations(range(norb), 2))


def _default_rho_indices(norb: int) -> list[tuple[int, int, int]]:
    out = []
    for p in range(norb):
        for q in range(norb):
            if q == p:
                continue
            for r in range(q + 1, norb):
                if r == p:
                    continue
                out.append((p, q, r))
    return out


def _default_sigma_indices(norb: int) -> list[tuple[int, int, int, int]]:
    return list(itertools.combinations(range(norb), 4))


def _validate_rho_indices(
    triples: list[tuple[int, int, int]] | None,
    norb: int,
) -> list[tuple[int, int, int]]:
    if triples is None:
        return _default_rho_indices(norb)
    out = []
    seen = set()
    for p, q, r in triples:
        if not (0 <= p < norb and 0 <= q < norb and 0 <= r < norb):
            raise ValueError("rho indices out of bounds")
        if p == q or p == r or q == r:
            raise ValueError("rho indices must be distinct")
        if q >= r:
            raise ValueError("rho indices must satisfy q < r")
        if (p, q, r) in seen:
            raise ValueError("rho indices must not contain duplicates")
        seen.add((p, q, r))
        out.append((p, q, r))
    return out


def _validate_sigma_indices(
    quads: list[tuple[int, int, int, int]] | None,
    norb: int,
) -> list[tuple[int, int, int, int]]:
    if quads is None:
        return _default_sigma_indices(norb)
    out = []
    seen = set()
    for p, q, r, s in quads:
        if not (0 <= p < q < r < s < norb):
            raise ValueError("sigma indices must satisfy 0 <= p < q < r < s < norb")
        if (p, q, r, s) in seen:
            raise ValueError("sigma indices must not contain duplicates")
        seen.add((p, q, r, s))
        out.append((p, q, r, s))
    return out

__all__ = [
    "_assert_square_matrix",
    "_balanced_irreducible_pair_matrices",
    "_balanced_left_phase_vector",
    "_default_eta_indices",
    "_default_pair_indices",
    "_default_rho_indices",
    "_default_sigma_indices",
    "_default_tau_indices",
    "_default_triple_indices",
    "_diag_unitary",
    "_final_unitary_from_left_and_right",
    "_left_right_ov_adapted_to_native",
    "_native_to_left_right_ov_adapted",
    "_n_total_from_nocc",
    "_orbital_relabeling_unitary",
    "_ordered_matrix_from_values",
    "_parameters_from_zero_diag_antihermitian",
    "_restricted_irreducible_pair_matrix",
    "_restricted_left_phase_vector",
    "_right_unitary_from_left_and_final",
    "_symmetric_matrix_from_values",
    "_validate_ordered_pairs",
    "_validate_pairs",
    "_validate_rho_indices",
    "_validate_sigma_indices",
    "_validate_triples",
    "_values_from_ordered_matrix",
    "_zero_diag_antihermitian_from_parameters",
    "exact_reference_ov_params_from_unitary",
    "exact_reference_ov_unitary",
    "orbital_relabeling_from_overlap",
    "orbital_transport_unitary_from_overlap",
]
