from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from typing import Callable

import numpy as np

from xquces.basis import sector_shape
from xquces.gcr.product_pair_uccd import (
    _pair_uccd_ov_pairs,
    pair_uccd_parameters_from_t2,
)
from xquces.states import _doci_spatial_basis, _doci_subspace_indices


@cache
def pair_givens_indices(norb: int) -> tuple[tuple[int, int], ...]:
    return tuple((p, q) for p in range(norb) for q in range(p + 1, norb))


pair_jastrow_indices = pair_givens_indices


def _validate_closed_pair_count(norb: int, nocc: int) -> tuple[int, int]:
    norb = int(norb)
    nocc = int(nocc)
    if norb < 0:
        raise ValueError("norb must be non-negative")
    if nocc < 0 or nocc > norb:
        raise ValueError("nocc must satisfy 0 <= nocc <= norb")
    return norb, nocc


def _as_real_params(params: np.ndarray, expected: int, name: str) -> np.ndarray:
    out = np.asarray(params, dtype=np.float64)
    if out.shape != (expected,):
        raise ValueError(f"{name}: expected {(expected,)}, got {out.shape}.")
    return out


def _hf_pair_register_state(norb: int, nocc: int) -> np.ndarray:
    out = np.zeros(1 << norb, dtype=np.complex128)
    bitmask = 0
    for p in range(nocc):
        bitmask |= 1 << p
    out[bitmask] = 1.0
    return out


def _bitmask_from_occ(occ: tuple[int, ...]) -> int:
    bitmask = 0
    for p in occ:
        bitmask |= 1 << int(p)
    return bitmask


def embed_pair_register_state(
    norb: int,
    nocc: int,
    pair_state: np.ndarray,
) -> np.ndarray:
    """Embed a logical pair-register state into the spin-orbital DOCI subspace."""

    norb, nocc = _validate_closed_pair_count(norb, nocc)
    arr = np.asarray(pair_state, dtype=np.complex128).reshape(-1)
    if arr.shape != (1 << norb,):
        raise ValueError(f"Expected {(1 << norb,)}, got {arr.shape}.")
    nelec = (nocc, nocc)
    dim_a, dim_b = sector_shape(norb, nelec)
    out = np.zeros(dim_a * dim_b, dtype=np.complex128)
    indices = _doci_subspace_indices(norb, nelec)
    for k, occ in enumerate(_doci_spatial_basis(norb, nocc)):
        out[int(indices[k])] = arr[_bitmask_from_occ(occ)]
    return out


def embed_pair_register_jacobian(
    norb: int,
    nocc: int,
    pair_jacobian: np.ndarray,
) -> np.ndarray:
    norb, nocc = _validate_closed_pair_count(norb, nocc)
    jac = np.asarray(pair_jacobian, dtype=np.complex128)
    if jac.ndim != 2 or jac.shape[0] != (1 << norb):
        raise ValueError("pair_jacobian has incompatible shape")
    nelec = (nocc, nocc)
    dim_a, dim_b = sector_shape(norb, nelec)
    out = np.zeros((dim_a * dim_b, jac.shape[1]), dtype=np.complex128)
    indices = _doci_subspace_indices(norb, nelec)
    for k, occ in enumerate(_doci_spatial_basis(norb, nocc)):
        out[int(indices[k])] = jac[_bitmask_from_occ(occ)]
    return out


def project_spin_orbital_vector_to_pair_register(
    norb: int,
    nocc: int,
    vec: np.ndarray,
) -> np.ndarray:
    norb, nocc = _validate_closed_pair_count(norb, nocc)
    arr = np.asarray(vec, dtype=np.complex128).reshape(-1)
    nelec = (nocc, nocc)
    dim_a, dim_b = sector_shape(norb, nelec)
    if arr.shape != (dim_a * dim_b,):
        raise ValueError(f"Expected {(dim_a * dim_b,)}, got {arr.shape}.")
    out = np.zeros(1 << norb, dtype=np.complex128)
    indices = _doci_subspace_indices(norb, nelec)
    for k, occ in enumerate(_doci_spatial_basis(norb, nocc)):
        out[_bitmask_from_occ(occ)] = arr[int(indices[k])]
    return out


def _apply_pair_register_givens_in_place(
    vec: np.ndarray,
    p: int,
    q: int,
    theta: float,
) -> None:
    if theta == 0.0:
        return
    bit_p = 1 << p
    bit_q = 1 << q
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    for mask in range(vec.size):
        if mask & bit_p and not mask & bit_q:
            target = mask ^ bit_p ^ bit_q
            x = vec[mask]
            y = vec[target]
            vec[mask] = c * x - s * y
            vec[target] = s * x + c * y


def _pair_register_givens_generator_apply(
    vec: np.ndarray,
    p: int,
    q: int,
) -> np.ndarray:
    out = np.zeros_like(vec)
    bit_p = 1 << p
    bit_q = 1 << q
    for mask in range(vec.size):
        if mask & bit_p and not mask & bit_q:
            target = mask ^ bit_p ^ bit_q
            out[target] += vec[mask]
            out[mask] -= vec[target]
    return out


def _apply_pair_jastrow_phase_in_place(
    vec: np.ndarray,
    params: np.ndarray,
    indices: tuple[tuple[int, int], ...],
    *,
    time: float,
) -> None:
    for theta, (p, q) in zip(time * params, indices):
        if theta == 0.0:
            continue
        bit_p = 1 << p
        bit_q = 1 << q
        phase = np.exp(1j * float(theta))
        for mask in range(vec.size):
            if mask & bit_p and mask & bit_q:
                vec[mask] *= phase


def _pair_jastrow_generator_apply(
    vec: np.ndarray,
    p: int,
    q: int,
    *,
    time: float,
) -> np.ndarray:
    out = np.zeros_like(vec)
    bit_p = 1 << p
    bit_q = 1 << q
    for mask in range(vec.size):
        if mask & bit_p and mask & bit_q:
            out[mask] = 1j * time * vec[mask]
    return out


def _apply_pair_gcr2_op_in_place(vec: np.ndarray, op: tuple) -> None:
    kind = op[0]
    if kind == "givens":
        _, p, q, theta, time, *_ = op
        _apply_pair_register_givens_in_place(vec, p, q, time * theta)
        return
    if kind == "jastrow":
        _, p, q, theta, time, *_ = op
        _apply_pair_jastrow_phase_in_place(
            vec,
            np.asarray([theta], dtype=np.float64),
            ((p, q),),
            time=time,
        )
        return
    raise ValueError(f"Unknown operation kind {kind!r}")


def _pair_gcr2_ops(
    norb: int,
    nocc: int,
    reference_params: np.ndarray,
    left_params: np.ndarray,
    jastrow_params: np.ndarray,
    right_params: np.ndarray,
    *,
    time: float,
) -> list[tuple]:
    ops: list[tuple] = []
    n_ref = len(_pair_uccd_ov_pairs(norb, nocc))
    n_pair = len(pair_givens_indices(norb))
    for k, (theta, (i, a)) in enumerate(
        zip(reference_params, _pair_uccd_ov_pairs(norb, nocc))
    ):
        col = k
        ops.append(("givens", i, a, float(theta), time, col))
    for k, (theta, (p, q)) in enumerate(zip(right_params, pair_givens_indices(norb))):
        col = n_ref + 2 * n_pair + k
        ops.append(("givens", p, q, float(theta), time, col))
    for k, (theta, (p, q)) in enumerate(
        zip(jastrow_params, pair_jastrow_indices(norb))
    ):
        col = n_ref + n_pair + k
        ops.append(("jastrow", p, q, float(theta), time, col))
    for k, (theta, (p, q)) in enumerate(zip(left_params, pair_givens_indices(norb))):
        col = n_ref + k
        ops.append(("givens", p, q, float(theta), time, col))
    return ops


@dataclass(frozen=True)
class PairGCR2SpinRestrictedSpec:
    jastrow_params: np.ndarray

    def __post_init__(self):
        params = np.asarray(self.jastrow_params, dtype=np.float64)
        object.__setattr__(self, "jastrow_params", params)


@dataclass(frozen=True)
class PairGCR2Ansatz:
    norb: int
    nocc: int
    reference_params: np.ndarray
    left_params: np.ndarray
    diagonal: PairGCR2SpinRestrictedSpec
    right_params: np.ndarray

    def __post_init__(self):
        norb, nocc = _validate_closed_pair_count(self.norb, self.nocc)
        n_ref = len(_pair_uccd_ov_pairs(norb, nocc))
        n_pair = len(pair_givens_indices(norb))
        object.__setattr__(self, "norb", norb)
        object.__setattr__(self, "nocc", nocc)
        object.__setattr__(
            self,
            "reference_params",
            _as_real_params(self.reference_params, n_ref, "reference_params"),
        )
        object.__setattr__(
            self,
            "left_params",
            _as_real_params(self.left_params, n_pair, "left_params"),
        )
        object.__setattr__(
            self,
            "right_params",
            _as_real_params(self.right_params, n_pair, "right_params"),
        )
        if not isinstance(self.diagonal, PairGCR2SpinRestrictedSpec):
            object.__setattr__(
                self,
                "diagonal",
                PairGCR2SpinRestrictedSpec(self.diagonal),
            )
        object.__setattr__(
            self,
            "diagonal",
            PairGCR2SpinRestrictedSpec(
                _as_real_params(
                    self.diagonal.jastrow_params,
                    n_pair,
                    "diagonal.jastrow_params",
                )
            ),
        )

    @property
    def nelec(self) -> tuple[int, int]:
        return (self.nocc, self.nocc)

    @property
    def jastrow_params(self) -> np.ndarray:
        return self.diagonal.jastrow_params


def pair_gcr2_state(
    norb: int,
    nocc: int,
    reference_params: np.ndarray,
    left_params: np.ndarray,
    jastrow_params: np.ndarray,
    right_params: np.ndarray,
    *,
    time: float = 1.0,
) -> np.ndarray:
    norb, nocc = _validate_closed_pair_count(norb, nocc)
    param = PairGCR2Parameterization(norb, nocc)
    params = param.combine_parameters(
        reference_params,
        left_params,
        jastrow_params,
        right_params,
    )
    return param.state_from_parameters(params, time=time)


def pair_gcr2_state_jacobian(
    norb: int,
    nocc: int,
    params: np.ndarray,
    *,
    time: float = 1.0,
) -> np.ndarray:
    param = PairGCR2Parameterization(norb, nocc)
    return param.state_jacobian_from_parameters(params, time=time)


def pair_gcr2_state_vjp(
    norb: int,
    nocc: int,
    params: np.ndarray,
    v: np.ndarray,
    *,
    time: float = 1.0,
) -> np.ndarray:
    norb, nocc = _validate_closed_pair_count(norb, nocc)
    parameterization = PairGCR2Parameterization(norb, nocc)
    params = np.asarray(params, dtype=np.float64)
    if params.shape != (parameterization.n_params,):
        raise ValueError(
            f"Expected {(parameterization.n_params,)}, got {params.shape}."
        )

    reference, left, jastrow, right = parameterization.split_parameters(params)
    ops = _pair_gcr2_ops(
        norb,
        nocc,
        reference,
        left,
        jastrow,
        right,
        time=time,
    )

    forward = [_hf_pair_register_state(norb, nocc)]
    current = np.array(forward[0], copy=True)
    for op in ops:
        current = np.array(current, copy=True)
        _apply_pair_gcr2_op_in_place(current, op)
        forward.append(current)

    lam = np.asarray(v, dtype=np.complex128)
    if lam.shape != forward[-1].shape:
        raise ValueError(f"v must have shape {forward[-1].shape}, got {lam.shape}.")
    lam = np.array(lam, copy=True)
    grad = np.zeros(parameterization.n_params, dtype=np.float64)

    for k in reversed(range(len(ops))):
        op = ops[k]
        kind = op[0]
        if kind == "givens":
            _, p, q, _, op_time, col = op
            d_after = op_time * _pair_register_givens_generator_apply(
                forward[k + 1], p, q
            )
        elif kind == "jastrow":
            _, p, q, _, op_time, col = op
            d_after = _pair_jastrow_generator_apply(
                forward[k + 1], p, q, time=op_time
            )
        else:
            raise ValueError(f"Unknown operation kind {kind!r}")
        grad[col] = float(2.0 * np.real(np.vdot(d_after, lam)))

        inverse = list(op)
        inverse[3] = -float(inverse[3])
        _apply_pair_gcr2_op_in_place(lam, tuple(inverse))

    return grad


@dataclass(frozen=True)
class PairGCR2Parameterization:
    norb: int
    nocc: int

    def __post_init__(self):
        norb, nocc = _validate_closed_pair_count(self.norb, self.nocc)
        object.__setattr__(self, "norb", norb)
        object.__setattr__(self, "nocc", nocc)

    @property
    def nelec(self) -> tuple[int, int]:
        return (self.nocc, self.nocc)

    @property
    def pair_reference_indices(self) -> tuple[tuple[int, int], ...]:
        return _pair_uccd_ov_pairs(self.norb, self.nocc)

    @property
    def pair_givens_indices(self) -> tuple[tuple[int, int], ...]:
        return pair_givens_indices(self.norb)

    @property
    def pair_jastrow_indices(self) -> tuple[tuple[int, int], ...]:
        return pair_jastrow_indices(self.norb)

    @property
    def n_reference_params(self) -> int:
        return len(self.pair_reference_indices)

    @property
    def n_pair_givens_params(self) -> int:
        return len(self.pair_givens_indices)

    @property
    def n_jastrow_params(self) -> int:
        return len(self.pair_jastrow_indices)

    @property
    def n_ansatz_params(self) -> int:
        return 2 * self.n_pair_givens_params + self.n_jastrow_params

    @property
    def n_params(self) -> int:
        return self.n_reference_params + self.n_ansatz_params

    def split_parameters(
        self,
        params: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        params = np.asarray(params, dtype=np.float64)
        if params.shape != (self.n_params,):
            raise ValueError(f"Expected {(self.n_params,)}, got {params.shape}.")
        n_ref = self.n_reference_params
        n_pair = self.n_pair_givens_params
        reference = params[:n_ref]
        left = params[n_ref : n_ref + n_pair]
        jastrow = params[n_ref + n_pair : n_ref + 2 * n_pair]
        right = params[n_ref + 2 * n_pair :]
        return reference, left, jastrow, right

    def combine_parameters(
        self,
        reference_params: np.ndarray,
        left_params: np.ndarray,
        jastrow_params: np.ndarray,
        right_params: np.ndarray,
    ) -> np.ndarray:
        return np.concatenate(
            [
                _as_real_params(
                    reference_params,
                    self.n_reference_params,
                    "reference_params",
                ),
                _as_real_params(left_params, self.n_pair_givens_params, "left_params"),
                _as_real_params(
                    jastrow_params,
                    self.n_jastrow_params,
                    "jastrow_params",
                ),
                _as_real_params(
                    right_params,
                    self.n_pair_givens_params,
                    "right_params",
                ),
            ]
        )

    def zero_parameters(self) -> np.ndarray:
        return np.zeros(self.n_params, dtype=np.float64)

    def random_parameters(
        self,
        *,
        scale: float = 1e-3,
        seed: int | np.random.Generator | None = None,
    ) -> np.ndarray:
        rng = (
            seed
            if isinstance(seed, np.random.Generator)
            else np.random.default_rng(seed)
        )
        return float(scale) * rng.standard_normal(self.n_params)

    def parameters_from_t2(
        self,
        t2: np.ndarray,
        *,
        pair_scale: float = 0.5,
    ) -> np.ndarray:
        reference = pair_uccd_parameters_from_t2(t2, scale=pair_scale)
        return self.combine_parameters(
            reference,
            np.zeros(self.n_pair_givens_params, dtype=np.float64),
            np.zeros(self.n_jastrow_params, dtype=np.float64),
            np.zeros(self.n_pair_givens_params, dtype=np.float64),
        )

    def ansatz_from_parameters(self, params: np.ndarray) -> PairGCR2Ansatz:
        reference, left, jastrow, right = self.split_parameters(params)
        return PairGCR2Ansatz(
            self.norb,
            self.nocc,
            reference,
            left,
            PairGCR2SpinRestrictedSpec(jastrow),
            right,
        )

    def state_from_parameters(
        self,
        params: np.ndarray,
        *,
        time: float = 1.0,
    ) -> np.ndarray:
        reference, left, jastrow, right = self.split_parameters(params)
        state = _hf_pair_register_state(self.norb, self.nocc)

        for theta, (i, a) in zip(reference, self.pair_reference_indices):
            _apply_pair_register_givens_in_place(state, i, a, time * float(theta))
        for theta, (p, q) in zip(right, self.pair_givens_indices):
            _apply_pair_register_givens_in_place(state, p, q, time * float(theta))
        _apply_pair_jastrow_phase_in_place(
            state,
            jastrow,
            self.pair_jastrow_indices,
            time=time,
        )
        for theta, (p, q) in zip(left, self.pair_givens_indices):
            _apply_pair_register_givens_in_place(state, p, q, time * float(theta))
        return state

    def state_jacobian_from_parameters(
        self,
        params: np.ndarray,
        *,
        time: float = 1.0,
    ) -> np.ndarray:
        params = np.asarray(params, dtype=np.float64)
        if params.shape != (self.n_params,):
            raise ValueError(f"Expected {(self.n_params,)}, got {params.shape}.")
        reference, left, jastrow, right = self.split_parameters(params)
        ops = _pair_gcr2_ops(
            self.norb,
            self.nocc,
            reference,
            left,
            jastrow,
            right,
            time=time,
        )

        forward = [_hf_pair_register_state(self.norb, self.nocc)]
        current = np.array(forward[0], copy=True)
        for op in ops:
            current = np.array(current, copy=True)
            _apply_pair_gcr2_op_in_place(current, op)
            forward.append(current)

        out = np.zeros((1 << self.norb, self.n_params), dtype=np.complex128)
        for k, op in enumerate(ops):
            kind = op[0]
            if kind == "givens":
                _, p, q, _, op_time, col = op
                vec = op_time * _pair_register_givens_generator_apply(
                    forward[k + 1], p, q
                )
            elif kind == "jastrow":
                _, p, q, _, op_time, col = op
                vec = _pair_jastrow_generator_apply(
                    forward[k + 1], p, q, time=op_time
                )
            else:
                raise ValueError(f"Unknown operation kind {kind!r}")
            vec = np.asarray(vec, dtype=np.complex128)
            for later in ops[k + 1 :]:
                _apply_pair_gcr2_op_in_place(vec, later)
            out[:, col] = vec
        return out

    def embedded_state_from_parameters(
        self,
        params: np.ndarray,
        *,
        time: float = 1.0,
    ) -> np.ndarray:
        return embed_pair_register_state(
            self.norb,
            self.nocc,
            self.state_from_parameters(params, time=time),
        )

    def embedded_state_jacobian_from_parameters(
        self,
        params: np.ndarray,
        *,
        time: float = 1.0,
    ) -> np.ndarray:
        return embed_pair_register_jacobian(
            self.norb,
            self.nocc,
            self.state_jacobian_from_parameters(params, time=time),
        )

    def energy_gradient_from_parameters(self, params: np.ndarray, H):
        params = np.asarray(params, dtype=np.float64)
        ham = getattr(H, "ham", H)
        use_embedded = (
            getattr(ham, "norb", None) == self.norb
            and tuple(getattr(ham, "nelec", ())) == self.nelec
        )
        if use_embedded:
            psi = self.embedded_state_from_parameters(params)
            hpsi = H @ psi if hasattr(H, "__matmul__") else ham.matvec(psi)
        else:
            psi = self.state_from_parameters(params)
            hpsi = H @ psi if hasattr(H, "__matmul__") else ham.matvec(psi)
        e = float(np.vdot(psi, hpsi).real)
        residual = hpsi - e * psi
        if use_embedded:
            residual = project_spin_orbital_vector_to_pair_register(
                self.norb,
                self.nocc,
                residual,
            )
        grad = pair_gcr2_state_vjp(self.norb, self.nocc, params, residual)
        return e, grad

    def transfer_parameters_from(
        self,
        previous_parameters: np.ndarray,
        previous_parameterization: object | None = None,
        **_,
    ) -> np.ndarray:
        if previous_parameterization is None:
            previous_parameterization = self
        prev = np.asarray(previous_parameters, dtype=np.float64)
        if (
            getattr(previous_parameterization, "norb", None) == self.norb
            and getattr(previous_parameterization, "nocc", None) == self.nocc
            and prev.shape == (self.n_params,)
        ):
            return np.array(prev, copy=True)
        out = np.zeros(self.n_params, dtype=np.float64)
        n = min(out.size, prev.size)
        out[:n] = prev[:n]
        return out

    def params_to_state(self) -> Callable[[np.ndarray], np.ndarray]:
        def func(params: np.ndarray) -> np.ndarray:
            return self.state_from_parameters(params)

        return func


__all__ = [
    "PairGCR2Ansatz",
    "PairGCR2Parameterization",
    "PairGCR2SpinRestrictedSpec",
    "embed_pair_register_jacobian",
    "embed_pair_register_state",
    "pair_gcr2_state",
    "pair_gcr2_state_jacobian",
    "pair_gcr2_state_vjp",
    "pair_givens_indices",
    "pair_jastrow_indices",
    "project_spin_orbital_vector_to_pair_register",
]
