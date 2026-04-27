from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from xquces.gcr.commutator_gcr2 import _diag2_features
from xquces.gcr.igcr2 import (
    IGCR2Ansatz,
    IGCR2SpinRestrictedParameterization,
    IGCR2SpinRestrictedSpec,
    _orbital_relabeling_unitary,
    _symmetric_matrix_from_values,
    _validate_pairs,
    orbital_relabeling_from_overlap,
)
from xquces.orbitals import apply_orbital_rotation
from xquces.states import (
    _doci_spatial_basis,
    _doci_subspace_indices,
    doci_amplitudes_from_parameters,
    doci_dimension,
    doci_parameters_from_amplitudes,
)
from xquces.ucj.model import UCJAnsatz


def _doci_unitary_from_amplitudes(amplitudes: np.ndarray) -> np.ndarray:
    target = np.asarray(amplitudes, dtype=np.float64)
    norm = float(np.linalg.norm(target))
    if norm == 0.0:
        raise ValueError("DOCI amplitudes must be nonzero")
    target = target / norm
    for value in target:
        if abs(value) > 1e-14:
            if value < 0.0:
                target = -target
            break
    dim = target.size
    unitary = np.eye(dim, dtype=np.complex128)
    if dim == 1:
        return unitary
    e0 = np.zeros(dim, dtype=np.float64)
    e0[0] = 1.0
    diff = e0 - target
    diff_norm = float(np.linalg.norm(diff))
    if diff_norm < 1e-14:
        return unitary
    u = diff / diff_norm
    return unitary - 2.0 * np.outer(u, u).astype(np.complex128)


def _doci_unitary_from_params(
    doci_reference_params: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
) -> np.ndarray:
    amplitudes = doci_amplitudes_from_parameters(norb, nelec, doci_reference_params)
    return _doci_unitary_from_amplitudes(amplitudes)


def apply_doci_reference_global(
    vec: np.ndarray,
    doci_reference_params: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    copy: bool = True,
    unitary: np.ndarray | None = None,
) -> np.ndarray:
    out = np.array(vec, dtype=np.complex128, copy=copy)
    indices = _doci_subspace_indices(norb, nelec)
    if indices.size == 0:
        return out
    if unitary is None:
        unitary = _doci_unitary_from_params(doci_reference_params, norb, nelec)
    out[indices] = unitary @ np.asarray(out[indices], dtype=np.complex128)
    return out


def _transfer_doci_reference_params(
    doci_reference_params: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    old_for_new: np.ndarray | None,
    phases: np.ndarray | None,
) -> np.ndarray:
    doci_reference_params = np.asarray(doci_reference_params, dtype=np.float64)
    if old_for_new is None:
        return np.array(doci_reference_params, copy=True)
    if nelec[0] != nelec[1]:
        raise ValueError("DOCI reference requires nalpha == nbeta")
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
    amps_old = doci_amplitudes_from_parameters(norb, nelec, doci_reference_params)
    basis_old = _doci_spatial_basis(norb, nelec[0])
    basis_new = _doci_spatial_basis(norb, nelec[0])
    basis_new_index = {occ: i for i, occ in enumerate(basis_new)}
    amps_new = np.zeros_like(amps_old)
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
                "DOCI reference transfer encountered non-real DOCI basis phase"
            )
        amps_new[i_new] = float(np.real(gamma)) * amps_old[i_old]
    return doci_parameters_from_amplitudes(amps_new)


@dataclass(frozen=True)
class GCR2DOCIReferenceAnsatz:
    diag_params: np.ndarray
    doci_reference_params: np.ndarray
    left: np.ndarray
    middle: np.ndarray
    norb: int
    nocc: int
    diag_pairs: tuple[tuple[int, int], ...]

    @property
    def pair_params(self) -> np.ndarray:
        return self.diag_params

    @property
    def right(self) -> np.ndarray:
        return self.middle

    def apply(
        self, vec: np.ndarray, nelec: tuple[int, int], copy: bool = True
    ) -> np.ndarray:
        out = apply_doci_reference_global(
            vec,
            self.doci_reference_params,
            self.norb,
            nelec,
            copy=copy,
        )
        out = apply_orbital_rotation(
            out,
            self.middle,
            self.norb,
            nelec,
            copy=False,
        )
        phases = _diag2_features(self.norb, nelec, self.diag_pairs) @ self.diag_params
        out *= np.exp(1j * phases)
        return apply_orbital_rotation(
            out,
            self.left,
            self.norb,
            nelec,
            copy=False,
        )


@dataclass(frozen=True)
class GCR2DOCIReferenceParameterization:
    norb: int
    nocc: int
    interaction_pairs: list[tuple[int, int]] | None = None
    base_parameterization: IGCR2SpinRestrictedParameterization | None = None
    left_right_ov_relative_scale: float | None = 1.0
    real_right_orbital_chart: bool = False
    _interaction_pairs: tuple[tuple[int, int], ...] = field(init=False, repr=False)

    def __post_init__(self):
        if self.base_parameterization is not None:
            if self.base_parameterization.norb != self.norb:
                raise ValueError("base_parameterization.norb does not match")
            if self.base_parameterization.nocc != self.nocc:
                raise ValueError("base_parameterization.nocc does not match")
            base_pairs = tuple(self.base_parameterization.pair_indices)
        else:
            base_pairs = _validate_pairs(
                self.interaction_pairs, self.norb, allow_diagonal=False
            )
        object.__setattr__(self, "_interaction_pairs", tuple(base_pairs))

    @property
    def pair_indices(self) -> tuple[tuple[int, int], ...]:
        return self._interaction_pairs

    @property
    def diag_indices(self) -> tuple[tuple[int, int], ...]:
        return self._interaction_pairs

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
    def n_diag_params(self) -> int:
        return len(self._interaction_pairs)

    @property
    def n_pair_params(self) -> int:
        return self.n_diag_params

    @property
    def n_doci_reference_params(self) -> int:
        return doci_dimension(self.norb, (self.nocc, self.nocc)) - 1

    @property
    def n_pair_reference_params(self) -> int:
        return self.n_doci_reference_params

    @property
    def n_middle_orbital_rotation_params(self) -> int:
        return self.n_left_orbital_rotation_params

    @property
    def n_right_orbital_rotation_params(self) -> int:
        return self.n_middle_orbital_rotation_params

    @property
    def _right_orbital_rotation_start(self) -> int:
        return (
            self.n_left_orbital_rotation_params
            + self.n_diag_params
            + self.n_doci_reference_params
        )

    @property
    def _left_right_ov_transform_scale(self):
        return None

    @property
    def n_params(self) -> int:
        return (
            self.n_left_orbital_rotation_params
            + self.n_diag_params
            + self.n_doci_reference_params
            + self.n_middle_orbital_rotation_params
        )

    def _split(
        self, params: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        params = np.asarray(params, dtype=np.float64)
        if params.shape != (self.n_params,):
            raise ValueError(f"Expected {(self.n_params,)}, got {params.shape}.")
        idx = 0
        n = self.n_left_orbital_rotation_params
        left = params[idx : idx + n]
        idx += n
        n = self.n_diag_params
        diag = params[idx : idx + n]
        idx += n
        n = self.n_doci_reference_params
        doci_reference = params[idx : idx + n]
        idx += n
        middle = params[idx:]
        return left, diag, doci_reference, middle

    def _base_params_from_split(
        self,
        left: np.ndarray,
        diag: np.ndarray,
        right: np.ndarray,
    ) -> np.ndarray:
        return np.concatenate([left, diag, right]).astype(np.float64, copy=False)

    def _zero_diag_matrix(self) -> np.ndarray:
        return np.zeros((self.norb, self.norb), dtype=np.float64)

    def _identity_orbital_rotation(self) -> np.ndarray:
        return np.eye(self.norb, dtype=np.complex128)

    def _extract_full_rotation_params(self, unitary: np.ndarray) -> np.ndarray:
        dummy = IGCR2Ansatz(
            diagonal=IGCR2SpinRestrictedSpec(pair=self._zero_diag_matrix()),
            left=np.asarray(unitary, dtype=np.complex128),
            right=self._identity_orbital_rotation(),
            nocc=self.nocc,
        )
        base_params = self._base.parameters_from_ansatz(dummy)
        return np.asarray(
            base_params[: self.n_left_orbital_rotation_params], dtype=np.float64
        )

    def _transfer_full_rotation_params(
        self,
        params: np.ndarray,
        previous_base: IGCR2SpinRestrictedParameterization,
        old_for_new: np.ndarray | None,
        phases: np.ndarray | None,
        block_diagonal: bool,
    ) -> np.ndarray:
        prev_left = np.asarray(params, dtype=np.float64)
        prev_diag = np.zeros(len(previous_base.pair_indices), dtype=np.float64)
        prev_right = np.zeros(
            previous_base.n_right_orbital_rotation_params, dtype=np.float64
        )
        transferred = self._base.transfer_parameters_from(
            np.concatenate([prev_left, prev_diag, prev_right]),
            previous_parameterization=previous_base,
            old_for_new=old_for_new,
            phases=phases,
            orbital_overlap=None,
            block_diagonal=block_diagonal,
        )
        return np.asarray(
            transferred[: self.n_left_orbital_rotation_params], dtype=np.float64
        )

    def ansatz_from_parameters(self, params: np.ndarray) -> GCR2DOCIReferenceAnsatz:
        left, diag, doci_reference, middle = self._split(params)
        left_dummy = self._base.ansatz_from_parameters(
            self._base_params_from_split(
                left,
                diag,
                np.zeros(self._base.n_right_orbital_rotation_params, dtype=np.float64),
            )
        )
        mid_unitary = self._mid_orbital_chart.unitary_from_parameters(
            np.asarray(middle, dtype=np.float64), self.norb
        )
        diag_matrix = np.asarray(left_dummy.diagonal.pair, dtype=np.float64)
        diag_values = np.asarray(
            [diag_matrix[p, q] for p, q in self._interaction_pairs], dtype=np.float64
        )
        return GCR2DOCIReferenceAnsatz(
            diag_params=diag_values,
            doci_reference_params=np.asarray(doci_reference, dtype=np.float64),
            left=np.asarray(left_dummy.left, dtype=np.complex128),
            middle=np.asarray(mid_unitary, dtype=np.complex128),
            norb=self.norb,
            nocc=self.nocc,
            diag_pairs=self._interaction_pairs,
        )

    def parameters_from_ansatz(
        self, ansatz: GCR2DOCIReferenceAnsatz | IGCR2Ansatz
    ) -> np.ndarray:
        if isinstance(ansatz, GCR2DOCIReferenceAnsatz):
            left_dummy = IGCR2Ansatz(
                diagonal=IGCR2SpinRestrictedSpec(
                    pair=_symmetric_matrix_from_values(
                        ansatz.diag_params,
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
                base_params[: self.n_left_orbital_rotation_params], dtype=np.float64
            )
            diag_start = self.n_left_orbital_rotation_params
            diag_stop = diag_start + self.n_diag_params
            diag = np.asarray(base_params[diag_start:diag_stop], dtype=np.float64)
            middle = self._extract_full_rotation_params(ansatz.middle)
            return np.concatenate(
                [
                    left,
                    diag,
                    np.asarray(ansatz.doci_reference_params, dtype=np.float64),
                    middle,
                ]
            )
        if isinstance(ansatz, IGCR2Ansatz):
            base_params = self._base.parameters_from_ansatz(ansatz)
            left = np.asarray(
                base_params[: self.n_left_orbital_rotation_params], dtype=np.float64
            )
            diag_start = self.n_left_orbital_rotation_params
            diag_stop = diag_start + self.n_diag_params
            diag = np.asarray(base_params[diag_start:diag_stop], dtype=np.float64)
            middle = np.zeros(self.n_middle_orbital_rotation_params, dtype=np.float64)
            return np.concatenate(
                [
                    left,
                    diag,
                    np.zeros(self.n_doci_reference_params, dtype=np.float64),
                    middle,
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
            base_params[: self.n_left_orbital_rotation_params], dtype=np.float64
        )
        diag_start = self.n_left_orbital_rotation_params
        diag_stop = diag_start + self.n_diag_params
        diag = np.asarray(base_params[diag_start:diag_stop], dtype=np.float64)
        middle = np.zeros(self.n_middle_orbital_rotation_params, dtype=np.float64)
        return np.concatenate(
            [
                left,
                diag,
                np.zeros(self.n_doci_reference_params, dtype=np.float64),
                middle,
            ]
        )

    def transfer_parameters_from(
        self,
        previous_parameters: np.ndarray,
        previous_parameterization: "GCR2DOCIReferenceParameterization | IGCR2SpinRestrictedParameterization | None" = None,
        old_for_new: np.ndarray | None = None,
        phases: np.ndarray | None = None,
        orbital_overlap: np.ndarray | None = None,
        block_diagonal: bool = True,
    ) -> np.ndarray:
        if previous_parameterization is None:
            previous_parameterization = self
        if (
            isinstance(previous_parameterization, GCR2DOCIReferenceParameterization)
            and previous_parameterization.norb == self.norb
            and previous_parameterization.nocc == self.nocc
            and old_for_new is None
            and phases is None
            and orbital_overlap is None
            and previous_parameterization.pair_indices == self.pair_indices
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
        if isinstance(previous_parameterization, GCR2DOCIReferenceParameterization):
            prev_params = np.asarray(previous_parameters, dtype=np.float64)
            if prev_params.shape != (previous_parameterization.n_params,):
                raise ValueError(
                    f"Expected {(previous_parameterization.n_params,)}, got {prev_params.shape}."
                )
            prev_left, prev_diag, prev_doci_reference, prev_middle = (
                previous_parameterization._split(prev_params)
            )
            base_params = self._base.transfer_parameters_from(
                previous_parameterization._base_params_from_split(
                    prev_left,
                    prev_diag,
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
            n_diag = self.n_diag_params
            left = np.asarray(base_params[:n_left], dtype=np.float64)
            diag = np.asarray(base_params[n_left : n_left + n_diag], dtype=np.float64)
            middle = self._transfer_full_rotation_params(
                prev_middle,
                previous_parameterization._base,
                old_for_new,
                phases,
                block_diagonal,
            )
            doci_reference = _transfer_doci_reference_params(
                prev_doci_reference,
                self.norb,
                (self.nocc, self.nocc),
                old_for_new,
                phases,
            )
            return np.concatenate([left, diag, doci_reference, middle])
        base_params = self._base.transfer_parameters_from(
            previous_parameters,
            previous_parameterization=previous_parameterization,
            old_for_new=old_for_new,
            phases=phases,
            orbital_overlap=None,
            block_diagonal=block_diagonal,
        )
        left = np.asarray(
            base_params[: self.n_left_orbital_rotation_params], dtype=np.float64
        )
        diag_start = self.n_left_orbital_rotation_params
        diag_stop = diag_start + self.n_diag_params
        diag = np.asarray(base_params[diag_start:diag_stop], dtype=np.float64)
        doci_reference = np.zeros(self.n_doci_reference_params, dtype=np.float64)
        middle = np.zeros(self.n_middle_orbital_rotation_params, dtype=np.float64)
        return np.concatenate([left, diag, doci_reference, middle])

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
