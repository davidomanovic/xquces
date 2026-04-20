from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from functools import cache
from typing import Callable

import numpy as np
import scipy.linalg

from xquces.basis import occ_rows
from xquces.gcr.commutator_gcr2 import _diag2_features, _validate_pairs
from xquces.gcr.igcr2 import (
    IGCR2Ansatz,
    IGCR2SpinRestrictedParameterization,
    IGCR2SpinRestrictedSpec,
    _symmetric_matrix_from_values,
    orbital_relabeling_from_overlap,
)
from xquces.orbitals import apply_orbital_rotation
from xquces.ucj.model import UCJAnsatz


def _default_pair_reference_pairs(
    norb: int,
    nocc: int,
) -> tuple[tuple[int, int], ...]:
    del nocc
    return _validate_pairs(None, norb)


@cache
def _doci_spatial_basis(
    norb: int,
    nocc: int,
) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(x) for x in itertools.combinations(range(norb), nocc))


@cache
def _sector_occ_tuples(
    norb: int,
    nocc: int,
) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(x) for x in row) for row in occ_rows(norb, nocc))


@cache
def _doci_subspace_indices(
    norb: int,
    nelec: tuple[int, int],
) -> np.ndarray:
    if nelec[0] != nelec[1]:
        raise ValueError("DOCI middle block requires nalpha == nbeta")
    nocc = nelec[0]
    sector = _sector_occ_tuples(norb, nocc)
    sector_index = {occ: i for i, occ in enumerate(sector)}
    dim_beta = len(sector)
    basis = _doci_spatial_basis(norb, nocc)
    return np.asarray(
        [sector_index[occ] * dim_beta + sector_index[occ] for occ in basis],
        dtype=np.intp,
    )


def _doci_generator_from_params(
    pair_reference_params: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
) -> np.ndarray:
    if nelec[0] != nelec[1]:
        raise ValueError("DOCI middle block requires nalpha == nbeta")
    nocc = nelec[0]
    dim = len(_doci_spatial_basis(norb, nocc))
    expected = dim * (dim - 1) // 2
    pair_reference_params = np.asarray(pair_reference_params, dtype=np.float64)
    if pair_reference_params.shape != (expected,):
        raise ValueError(
            f"pair_reference_params has wrong shape: expected ({expected},), got {pair_reference_params.shape}"
        )
    generator = np.zeros((dim, dim), dtype=np.float64)
    if expected == 0:
        return generator
    iu = np.triu_indices(dim, k=1)
    generator[iu] = pair_reference_params
    generator[(iu[1], iu[0])] = -pair_reference_params
    return generator


def _params_from_doci_generator(
    generator: np.ndarray,
) -> np.ndarray:
    generator = np.asarray(generator, dtype=np.float64)
    if generator.ndim != 2 or generator.shape[0] != generator.shape[1]:
        raise ValueError("generator must be square")
    iu = np.triu_indices(generator.shape[0], k=1)
    return np.asarray(generator[iu], dtype=np.float64)


def _doci_unitary_from_params(
    pair_reference_params: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
) -> np.ndarray:
    generator = _doci_generator_from_params(pair_reference_params, norb, nelec)
    return np.asarray(scipy.linalg.expm(generator), dtype=np.complex128)


def apply_pair_reference_global(
    vec: np.ndarray,
    pair_reference_params: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    pairs: tuple[tuple[int, int], ...],
    copy: bool = True,
    unitary: np.ndarray | None = None,
) -> np.ndarray:
    del pairs
    out = np.array(vec, dtype=np.complex128, copy=copy)
    indices = _doci_subspace_indices(norb, nelec)
    if indices.size == 0:
        return out
    if unitary is None:
        unitary = _doci_unitary_from_params(pair_reference_params, norb, nelec)
    subvec = np.asarray(out[indices], dtype=np.complex128)
    out[indices] = unitary @ subvec
    return out


def _apply_pair_reference_global_batch(
    batch: np.ndarray,
    pair_reference_params: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    pairs: tuple[tuple[int, int], ...],
    unitary: np.ndarray | None = None,
) -> np.ndarray:
    del pairs
    batch = np.asarray(batch, dtype=np.complex128)
    if batch.ndim != 2:
        raise ValueError("batch must be two-dimensional")
    out = np.array(batch, dtype=np.complex128, copy=True)
    indices = _doci_subspace_indices(norb, nelec)
    if indices.size == 0 or out.shape[0] == 0:
        return out
    if unitary is None:
        unitary = _doci_unitary_from_params(pair_reference_params, norb, nelec)
    sub = np.asarray(out[:, indices], dtype=np.complex128)
    out[:, indices] = sub @ unitary.T
    return out


def _transfer_pair_reference_params(
    pair_reference_params: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    old_for_new: np.ndarray | None,
    phases: np.ndarray | None,
) -> np.ndarray:
    pair_reference_params = np.asarray(pair_reference_params, dtype=np.float64)
    if old_for_new is None:
        return np.array(pair_reference_params, copy=True)

    if nelec[0] != nelec[1]:
        raise ValueError("DOCI middle block requires nalpha == nbeta")

    old_for_new = np.asarray(old_for_new, dtype=np.int64)
    if old_for_new.ndim != 1 or old_for_new.shape[0] != norb:
        raise ValueError("old_for_new must have shape (norb,)")

    if phases is None:
        phase_arr = np.ones(norb, dtype=np.complex128)
    else:
        phase_arr = np.asarray(phases, dtype=np.complex128)
        if phase_arr.shape != (norb,):
            raise ValueError("phases must have shape (norb,)")

    current_for_old = np.empty_like(old_for_new)
    current_for_old[old_for_new] = np.arange(norb)

    basis_old = _doci_spatial_basis(norb, nelec[0])
    basis_new = _doci_spatial_basis(norb, nelec[0])
    basis_new_index = {occ: i for i, occ in enumerate(basis_new)}

    generator_old = _doci_generator_from_params(pair_reference_params, norb, nelec)
    dim = generator_old.shape[0]
    transform = np.zeros((dim, dim), dtype=np.float64)

    for i_old, occ_old in enumerate(basis_old):
        occ_new = tuple(sorted(int(current_for_old[p]) for p in occ_old))
        i_new = basis_new_index[occ_new]
        gamma = 1.0 + 0.0j
        for p_new in occ_new:
            gamma *= phase_arr[p_new] ** 2
        if abs(np.imag(gamma)) > 1e-8 or not np.isclose(
            abs(np.real(gamma)), 1.0, atol=1e-8
        ):
            raise ValueError(
                "pair-reference transfer encountered non-real DOCI basis phase; this parameterization only supports real DOCI generators."
            )
        transform[i_new, i_old] = float(np.real(gamma))

    generator_new = transform @ generator_old @ transform.T
    return _params_from_doci_generator(generator_new)


@dataclass(frozen=True)
class GCR2PairReferenceAnsatz:
    pair_params: np.ndarray
    pair_reference_params: np.ndarray
    left: np.ndarray
    right: np.ndarray
    norb: int
    nocc: int
    diag_pairs: tuple[tuple[int, int], ...]
    pair_reference_pairs: tuple[tuple[int, int], ...]
    use_rust: bool = False

    def apply(self, vec: np.ndarray, nelec: tuple[int, int], copy: bool = True) -> np.ndarray:
        out = apply_pair_reference_global(
            vec,
            self.pair_reference_params,
            self.norb,
            nelec,
            self.pair_reference_pairs,
            copy=copy,
        )
        out = apply_orbital_rotation(
            out,
            self.right,
            self.norb,
            nelec,
            copy=False,
        )
        phases = _diag2_features(self.norb, nelec, self.diag_pairs) @ self.pair_params
        out *= np.exp(1j * phases)
        return apply_orbital_rotation(
            out,
            self.left,
            self.norb,
            nelec,
            copy=False,
        )


@dataclass(frozen=True)
class GCR2PairReferenceParameterization:
    norb: int
    nocc: int
    pairs: list[tuple[int, int]] | None = None
    pair_reference_pairs: list[tuple[int, int]] | None = None
    base_parameterization: IGCR2SpinRestrictedParameterization | None = None
    left_right_ov_relative_scale: float | None = 1.0
    real_right_orbital_chart: bool = False
    _interaction_pairs: tuple[tuple[int, int], ...] = field(init=False, repr=False)
    _pair_reference_pairs: tuple[tuple[int, int], ...] = field(init=False, repr=False)

    def __post_init__(self):
        if self.base_parameterization is not None:
            if self.base_parameterization.norb != self.norb:
                raise ValueError("base_parameterization.norb does not match")
            if self.base_parameterization.nocc != self.nocc:
                raise ValueError("base_parameterization.nocc does not match")
            base_pairs = tuple(self.base_parameterization.pair_indices)
        else:
            base_pairs = _validate_pairs(self.pairs, self.norb)
        object.__setattr__(self, "_interaction_pairs", tuple(base_pairs))
        pair_reference_pairs = (
            _default_pair_reference_pairs(self.norb, self.nocc)
            if self.pair_reference_pairs is None
            else _validate_pairs(self.pair_reference_pairs, self.norb)
        )
        object.__setattr__(self, "_pair_reference_pairs", tuple(pair_reference_pairs))

    @property
    def pair_indices(self) -> tuple[tuple[int, int], ...]:
        return self._interaction_pairs

    @property
    def pair_reference_indices(self) -> tuple[tuple[int, int], ...]:
        return self._pair_reference_pairs

    @property
    def _base(self) -> IGCR2SpinRestrictedParameterization:
        if self.base_parameterization is not None:
            return self.base_parameterization
        return IGCR2SpinRestrictedParameterization(
            self.norb,
            self.nocc,
            interaction_pairs=list(self._interaction_pairs),
            real_right_orbital_chart=self.real_right_orbital_chart,
            left_right_ov_relative_scale=self.left_right_ov_relative_scale,
        )

    @property
    def _mid_orbital_chart(self):
        return self._base._left_orbital_chart

    @property
    def right_orbital_chart(self):
        return self._mid_orbital_chart

    @property
    def _left_orbital_chart(self):
        return self._base._left_orbital_chart

    @property
    def n_left_orbital_rotation_params(self) -> int:
        return self._base.n_left_orbital_rotation_params

    @property
    def n_pair_params(self) -> int:
        return len(self._interaction_pairs)

    @property
    def n_pair_reference_params(self) -> int:
        dim = len(_doci_spatial_basis(self.norb, self.nocc))
        return dim * (dim - 1) // 2

    @property
    def n_pair_hop_params(self) -> int:
        return self.n_pair_reference_params

    @property
    def n_right_orbital_rotation_params(self) -> int:
        return self.n_left_orbital_rotation_params

    @property
    def _right_orbital_rotation_start(self) -> int:
        return (
            self.n_left_orbital_rotation_params
            + self.n_pair_params
            + self.n_pair_reference_params
        )

    @property
    def _left_right_ov_transform_scale(self):
        return None

    def _native_parameters_from_public(self, params: np.ndarray) -> np.ndarray:
        return np.asarray(params, dtype=np.float64)

    def _public_parameters_from_native(self, params: np.ndarray) -> np.ndarray:
        return np.asarray(params, dtype=np.float64)

    @property
    def n_params(self) -> int:
        return (
            self.n_left_orbital_rotation_params
            + self.n_pair_params
            + self.n_pair_reference_params
            + self.n_right_orbital_rotation_params
        )

    def _split(
        self,
        params: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
        n = self.n_pair_reference_params
        pair_reference = params[idx : idx + n]
        idx += n
        right = params[idx:]
        return left, pair, pair_reference, right

    def _base_params_from_split(
        self,
        left: np.ndarray,
        pair: np.ndarray,
        right: np.ndarray,
    ) -> np.ndarray:
        return np.concatenate([left, pair, right]).astype(np.float64, copy=False)

    def _zero_pair_matrix(self) -> np.ndarray:
        return np.zeros((self.norb, self.norb), dtype=np.float64)

    def _identity_orbital_rotation(self) -> np.ndarray:
        return np.eye(self.norb, dtype=np.complex128)

    def _extract_full_rotation_params(
        self,
        unitary: np.ndarray,
    ) -> np.ndarray:
        dummy = IGCR2Ansatz(
            diagonal=IGCR2SpinRestrictedSpec(pair=self._zero_pair_matrix()),
            left=np.asarray(unitary, dtype=np.complex128),
            right=self._identity_orbital_rotation(),
            nocc=self.nocc,
        )
        base_params = self._base.parameters_from_ansatz(dummy)
        return np.asarray(
            base_params[: self.n_left_orbital_rotation_params],
            dtype=np.float64,
        )

    def _transfer_full_rotation_params(
        self,
        params: np.ndarray,
        previous_base: IGCR2SpinRestrictedParameterization,
        old_for_new: np.ndarray | None,
        phases: np.ndarray | None,
        block_diagonal: bool,
    ) -> np.ndarray:
        params = np.asarray(params, dtype=np.float64)
        prev_left = np.asarray(params, dtype=np.float64)
        prev_pair = np.zeros(len(previous_base.pair_indices), dtype=np.float64)
        prev_right = np.zeros(previous_base.n_right_orbital_rotation_params, dtype=np.float64)
        transferred = self._base.transfer_parameters_from(
            np.concatenate([prev_left, prev_pair, prev_right]),
            previous_parameterization=previous_base,
            old_for_new=old_for_new,
            phases=phases,
            orbital_overlap=None,
            block_diagonal=block_diagonal,
        )
        return np.asarray(
            transferred[: self.n_left_orbital_rotation_params],
            dtype=np.float64,
        )

    def ansatz_from_parameters(self, params: np.ndarray) -> GCR2PairReferenceAnsatz:
        left, pair, pair_reference, right = self._split(params)

        left_dummy = self._base.ansatz_from_parameters(
            self._base_params_from_split(
                left,
                pair,
                np.zeros(self._base.n_right_orbital_rotation_params, dtype=np.float64),
            )
        )
        if not isinstance(left_dummy, IGCR2Ansatz):
            raise TypeError("base parameterization returned an unexpected ansatz")

        mid_unitary = self._mid_orbital_chart.unitary_from_parameters(
            np.asarray(right, dtype=np.float64),
            self.norb,
        )

        pair_matrix = np.asarray(left_dummy.diagonal.pair, dtype=np.float64)
        pair_values = np.asarray(
            [pair_matrix[p, q] for p, q in self._interaction_pairs],
            dtype=np.float64,
        )

        return GCR2PairReferenceAnsatz(
            pair_params=pair_values,
            pair_reference_params=np.asarray(pair_reference, dtype=np.float64),
            left=np.asarray(left_dummy.left, dtype=np.complex128),
            right=np.asarray(mid_unitary, dtype=np.complex128),
            norb=self.norb,
            nocc=self.nocc,
            diag_pairs=self._interaction_pairs,
            pair_reference_pairs=self._pair_reference_pairs,
        )

    def parameters_from_ansatz(
        self, ansatz: GCR2PairReferenceAnsatz | IGCR2Ansatz
    ) -> np.ndarray:
        if isinstance(ansatz, GCR2PairReferenceAnsatz):
            left_dummy = IGCR2Ansatz(
                diagonal=IGCR2SpinRestrictedSpec(
                    pair=_symmetric_matrix_from_values(
                        ansatz.pair_params,
                        self.norb,
                        list(self._interaction_pairs),
                    )
                ),
                left=np.asarray(ansatz.left, dtype=np.complex128),
                right=self._identity_orbital_rotation(),
                nocc=ansatz.nocc,
            )
            base_params = self._base.parameters_from_ansatz(left_dummy)
            left = np.asarray(
                base_params[: self.n_left_orbital_rotation_params],
                dtype=np.float64,
            )
            pair_start = self.n_left_orbital_rotation_params
            pair_stop = pair_start + self.n_pair_params
            pair = np.asarray(base_params[pair_start:pair_stop], dtype=np.float64)
            right = self._extract_full_rotation_params(ansatz.right)
            return np.concatenate(
                [
                    left,
                    pair,
                    np.asarray(ansatz.pair_reference_params, dtype=np.float64),
                    right,
                ]
            )

        if isinstance(ansatz, IGCR2Ansatz):
            base_params = self._base.parameters_from_ansatz(ansatz)
            left = np.asarray(
                base_params[: self.n_left_orbital_rotation_params],
                dtype=np.float64,
            )
            pair_start = self.n_left_orbital_rotation_params
            pair_stop = pair_start + self.n_pair_params
            pair = np.asarray(base_params[pair_start:pair_stop], dtype=np.float64)
            right = np.zeros(self.n_right_orbital_rotation_params, dtype=np.float64)
            return np.concatenate(
                [
                    left,
                    pair,
                    np.zeros(self.n_pair_reference_params, dtype=np.float64),
                    right,
                ]
            )

        raise TypeError(type(ansatz).__name__)

    def parameters_from_igcr2(
        self,
        params: np.ndarray,
        parameterization: IGCR2SpinRestrictedParameterization | None = None,
    ) -> np.ndarray:
        parameterization = self._base if parameterization is None else parameterization
        if parameterization.norb != self.norb or parameterization.nocc != self.nocc:
            raise ValueError("IGCR2 parameterization shape does not match")
        ansatz = parameterization.ansatz_from_parameters(params)
        return self.parameters_from_ansatz(ansatz)

    def parameters_from_ucj_ansatz(self, ansatz: UCJAnsatz) -> np.ndarray:
        base_params = self._base.parameters_from_ucj_ansatz(ansatz)
        left = np.asarray(
            base_params[: self.n_left_orbital_rotation_params],
            dtype=np.float64,
        )
        pair_start = self.n_left_orbital_rotation_params
        pair_stop = pair_start + self.n_pair_params
        pair = np.asarray(base_params[pair_start:pair_stop], dtype=np.float64)
        right = np.zeros(self.n_right_orbital_rotation_params, dtype=np.float64)
        return np.concatenate(
            [
                left,
                pair,
                np.zeros(self.n_pair_reference_params, dtype=np.float64),
                right,
            ]
        )

    def transfer_parameters_from(
        self,
        previous_parameters: np.ndarray,
        previous_parameterization: (
            "GCR2PairReferenceParameterization | IGCR2SpinRestrictedParameterization | None"
        ) = None,
        old_for_new: np.ndarray | None = None,
        phases: np.ndarray | None = None,
        orbital_overlap: np.ndarray | None = None,
        block_diagonal: bool = True,
    ) -> np.ndarray:
        if previous_parameterization is None:
            previous_parameterization = self

        if (
            isinstance(previous_parameterization, GCR2PairReferenceParameterization)
            and previous_parameterization.norb == self.norb
            and previous_parameterization.nocc == self.nocc
            and old_for_new is None
            and phases is None
            and orbital_overlap is None
            and previous_parameterization.pair_indices == self.pair_indices
            and previous_parameterization.pair_reference_indices == self.pair_reference_indices
        ):
            params = np.asarray(previous_parameters, dtype=np.float64)
            if params.shape != (self.n_params,):
                raise ValueError(f"Expected {(self.n_params,)}, got {params.shape}.")
            return np.array(params, copy=True)

        if orbital_overlap is not None:
            if old_for_new is not None or phases is not None:
                raise ValueError(
                    "Pass either orbital_overlap or explicit relabeling, not both."
                )
            old_for_new, phases = orbital_relabeling_from_overlap(
                orbital_overlap,
                nocc=self.nocc,
                block_diagonal=block_diagonal,
            )

        if isinstance(previous_parameterization, GCR2PairReferenceParameterization):
            prev_params = np.asarray(previous_parameters, dtype=np.float64)
            if prev_params.shape != (previous_parameterization.n_params,):
                raise ValueError(
                    f"Expected {(previous_parameterization.n_params,)}, got {prev_params.shape}."
                )

            prev_left, prev_pair, prev_pair_reference, prev_right = (
                previous_parameterization._split(prev_params)
            )

            base_params = self._base.transfer_parameters_from(
                previous_parameterization._base_params_from_split(
                    prev_left,
                    prev_pair,
                    np.zeros(
                        previous_parameterization._base.n_right_orbital_rotation_params,
                        dtype=np.float64,
                    ),
                ),
                previous_parameterization=previous_parameterization._base,
                old_for_new=old_for_new,
                phases=phases,
                orbital_overlap=None,
                block_diagonal=block_diagonal,
            )

            n_left = self.n_left_orbital_rotation_params
            n_pair = self.n_pair_params
            left = np.asarray(base_params[:n_left], dtype=np.float64)
            pair = np.asarray(base_params[n_left : n_left + n_pair], dtype=np.float64)

            right = self._transfer_full_rotation_params(
                prev_right,
                previous_parameterization._base,
                old_for_new,
                phases,
                block_diagonal,
            )

            pair_reference = _transfer_pair_reference_params(
                prev_pair_reference,
                self.norb,
                (self.nocc, self.nocc),
                old_for_new,
                phases,
            )

            return np.concatenate([left, pair, pair_reference, right])

        base_params = self._base.transfer_parameters_from(
            previous_parameters,
            previous_parameterization=previous_parameterization,
            old_for_new=old_for_new,
            phases=phases,
            orbital_overlap=None,
            block_diagonal=block_diagonal,
        )

        left = np.asarray(
            base_params[: self.n_left_orbital_rotation_params],
            dtype=np.float64,
        )
        pair_start = self.n_left_orbital_rotation_params
        pair_stop = pair_start + self.n_pair_params
        pair = np.asarray(base_params[pair_start:pair_stop], dtype=np.float64)
        pair_reference = np.zeros(self.n_pair_reference_params, dtype=np.float64)
        right = np.zeros(self.n_right_orbital_rotation_params, dtype=np.float64)
        return np.concatenate([left, pair, pair_reference, right])

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