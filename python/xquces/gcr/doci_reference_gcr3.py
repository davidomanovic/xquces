from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from xquces.gcr.doci_reference_gcr2 import (
    _transfer_doci_reference_params,
    apply_doci_reference_global,
)
from xquces.gcr.igcr2 import orbital_relabeling_from_overlap
from xquces.gcr.igcr3 import (
    IGCR3Ansatz,
    IGCR3SpinRestrictedParameterization,
    IGCR3SpinRestrictedSpec,
    _default_pair_indices,
    _default_triple_indices,
)
from xquces.orbitals import apply_orbital_rotation
from xquces.states import doci_dimension
from xquces.ucj.model import UCJAnsatz


@dataclass(frozen=True)
class GCR3DOCIReferenceAnsatz:
    base_ansatz: IGCR3Ansatz
    doci_reference_params: np.ndarray
    middle: np.ndarray

    @property
    def norb(self) -> int:
        return self.base_ansatz.norb

    @property
    def nocc(self) -> int:
        return self.base_ansatz.nocc

    @property
    def left(self) -> np.ndarray:
        return self.base_ansatz.left

    @property
    def right(self) -> np.ndarray:
        return self.middle

    @property
    def diagonal(self):
        return self.base_ansatz.diagonal

    def apply(self, vec: np.ndarray, nelec: tuple[int, int], copy: bool = True) -> np.ndarray:
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
        return self.base_ansatz.apply(out, nelec=nelec, copy=False)


@dataclass(frozen=True)
class GCR3DOCIReferenceParameterization:
    norb: int
    nocc: int
    base_parameterization: IGCR3SpinRestrictedParameterization | None = None

    @property
    def _base(self) -> IGCR3SpinRestrictedParameterization:
        if self.base_parameterization is not None:
            return self.base_parameterization
        return IGCR3SpinRestrictedParameterization(self.norb, self.nocc)

    @property
    def pair_indices(self):
        return tuple(self._base.pair_indices)

    @property
    def _left_orbital_chart(self):
        return self._base._left_orbital_chart

    @property
    def _mid_orbital_chart(self):
        return self._base._left_orbital_chart

    @property
    def right_orbital_chart(self):
        return self._mid_orbital_chart

    @property
    def n_left_orbital_rotation_params(self) -> int:
        return self._base.n_left_orbital_rotation_params

    @property
    def n_diag_params(self) -> int:
        return self._base.n_params - self._base.n_left_orbital_rotation_params - self._base.n_right_orbital_rotation_params

    @property
    def n_pair_params(self) -> int:
        return self._base.n_pair_params

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
        return self.n_left_orbital_rotation_params + self.n_diag_params + self.n_doci_reference_params

    @property
    def n_params(self) -> int:
        return self.n_left_orbital_rotation_params + self.n_diag_params + self.n_doci_reference_params + self.n_middle_orbital_rotation_params

    def _split(self, params: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        params = np.asarray(params, dtype=np.float64)
        if params.shape != (self.n_params,):
            raise ValueError(f"Expected {(self.n_params,)}, got {params.shape}.")
        n_left = self.n_left_orbital_rotation_params
        n_diag = self.n_diag_params
        n_doci = self.n_doci_reference_params
        left = params[:n_left]
        diag = params[n_left : n_left + n_diag]
        doci_reference = params[n_left + n_diag : n_left + n_diag + n_doci]
        middle = params[n_left + n_diag + n_doci :]
        return left, diag, doci_reference, middle

    def _zero_diagonal(self) -> IGCR3SpinRestrictedSpec:
        return IGCR3SpinRestrictedSpec(
            double_params=np.zeros(self.norb, dtype=np.float64),
            pair_values=np.zeros(len(_default_pair_indices(self.norb)), dtype=np.float64),
            tau=np.zeros((self.norb, self.norb), dtype=np.float64),
            omega_values=np.zeros(len(_default_triple_indices(self.norb)), dtype=np.float64),
        )

    def _identity_orbital_rotation(self) -> np.ndarray:
        return np.eye(self.norb, dtype=np.complex128)

    def _base_params_from_split(self, left: np.ndarray, diag: np.ndarray, right: np.ndarray) -> np.ndarray:
        return np.concatenate([left, diag, right]).astype(np.float64, copy=False)

    def _extract_full_rotation_params(self, unitary: np.ndarray) -> np.ndarray:
        dummy = IGCR3Ansatz(
            diagonal=self._zero_diagonal(),
            left=np.asarray(unitary, dtype=np.complex128),
            right=self._identity_orbital_rotation(),
            nocc=self.nocc,
        )
        base_params = self._base.parameters_from_ansatz(dummy)
        return np.asarray(base_params[: self.n_left_orbital_rotation_params], dtype=np.float64)

    def _transfer_full_rotation_params(
        self,
        params: np.ndarray,
        previous_base: IGCR3SpinRestrictedParameterization,
        old_for_new: np.ndarray | None,
        phases: np.ndarray | None,
        block_diagonal: bool,
    ) -> np.ndarray:
        prev = np.zeros(previous_base.n_params, dtype=np.float64)
        prev[: previous_base.n_left_orbital_rotation_params] = np.asarray(params, dtype=np.float64)
        transferred = self._base.transfer_parameters_from(
            prev,
            previous_parameterization=previous_base,
            old_for_new=old_for_new,
            phases=phases,
            orbital_overlap=None,
            block_diagonal=block_diagonal,
        )
        return np.asarray(transferred[: self.n_left_orbital_rotation_params], dtype=np.float64)

    def ansatz_from_parameters(self, params: np.ndarray) -> GCR3DOCIReferenceAnsatz:
        left, diag, doci_reference, middle = self._split(params)
        base_ansatz = self._base.ansatz_from_parameters(
            self._base_params_from_split(
                left,
                diag,
                np.zeros(self._base.n_right_orbital_rotation_params, dtype=np.float64),
            )
        )
        mid_unitary = self._mid_orbital_chart.unitary_from_parameters(np.asarray(middle, dtype=np.float64), self.norb)
        return GCR3DOCIReferenceAnsatz(
            base_ansatz=base_ansatz,
            doci_reference_params=np.asarray(doci_reference, dtype=np.float64),
            middle=np.asarray(mid_unitary, dtype=np.complex128),
        )

    def parameters_from_ansatz(self, ansatz: GCR3DOCIReferenceAnsatz | IGCR3Ansatz) -> np.ndarray:
        if isinstance(ansatz, GCR3DOCIReferenceAnsatz):
            base_params = self._base.parameters_from_ansatz(ansatz.base_ansatz)
            left = np.asarray(base_params[: self.n_left_orbital_rotation_params], dtype=np.float64)
            diag = np.asarray(
                base_params[
                    self.n_left_orbital_rotation_params : self.n_left_orbital_rotation_params + self.n_diag_params
                ],
                dtype=np.float64,
            )
            middle = self._extract_full_rotation_params(ansatz.middle)
            return np.concatenate([
                left,
                diag,
                np.asarray(ansatz.doci_reference_params, dtype=np.float64),
                middle,
            ])
        if isinstance(ansatz, IGCR3Ansatz):
            base_params = self._base.parameters_from_ansatz(ansatz)
            left = np.asarray(base_params[: self.n_left_orbital_rotation_params], dtype=np.float64)
            diag = np.asarray(
                base_params[
                    self.n_left_orbital_rotation_params : self.n_left_orbital_rotation_params + self.n_diag_params
                ],
                dtype=np.float64,
            )
            middle = np.zeros(self.n_middle_orbital_rotation_params, dtype=np.float64)
            doci_reference = np.zeros(self.n_doci_reference_params, dtype=np.float64)
            return np.concatenate([left, diag, doci_reference, middle])
        raise TypeError(type(ansatz).__name__)

    def parameters_from_ucj_ansatz(self, ansatz: UCJAnsatz) -> np.ndarray:
        base_params = self._base.parameters_from_ucj_ansatz(ansatz)
        left = np.asarray(base_params[: self.n_left_orbital_rotation_params], dtype=np.float64)
        diag = np.asarray(
            base_params[
                self.n_left_orbital_rotation_params : self.n_left_orbital_rotation_params + self.n_diag_params
            ],
            dtype=np.float64,
        )
        middle = np.zeros(self.n_middle_orbital_rotation_params, dtype=np.float64)
        doci_reference = np.zeros(self.n_doci_reference_params, dtype=np.float64)
        return np.concatenate([left, diag, doci_reference, middle])

    def transfer_parameters_from(
        self,
        previous_parameters: np.ndarray,
        previous_parameterization: "GCR3DOCIReferenceParameterization | IGCR3SpinRestrictedParameterization | None" = None,
        old_for_new: np.ndarray | None = None,
        phases: np.ndarray | None = None,
        orbital_overlap: np.ndarray | None = None,
        block_diagonal: bool = True,
    ) -> np.ndarray:
        if previous_parameterization is None:
            previous_parameterization = self
        if orbital_overlap is not None:
            if old_for_new is not None or phases is not None:
                raise ValueError("Pass either orbital_overlap or explicit relabeling, not both.")
            old_for_new, phases = orbital_relabeling_from_overlap(
                orbital_overlap,
                nocc=self.nocc,
                block_diagonal=block_diagonal,
            )
        if isinstance(previous_parameterization, GCR3DOCIReferenceParameterization):
            prev = np.asarray(previous_parameters, dtype=np.float64)
            if prev.shape != (previous_parameterization.n_params,):
                raise ValueError(f"Expected {(previous_parameterization.n_params,)}, got {prev.shape}.")
            if (
                previous_parameterization.norb == self.norb
                and previous_parameterization.nocc == self.nocc
                and old_for_new is None
                and phases is None
            ):
                return np.array(prev, copy=True)
            prev_left, prev_diag, prev_doci_reference, prev_middle = previous_parameterization._split(prev)
            base_params = self._base.transfer_parameters_from(
                previous_parameterization._base_params_from_split(
                    prev_left,
                    prev_diag,
                    np.zeros(previous_parameterization._base.n_right_orbital_rotation_params, dtype=np.float64),
                ),
                previous_parameterization=previous_parameterization._base,
                old_for_new=old_for_new,
                phases=phases,
                orbital_overlap=None,
                block_diagonal=block_diagonal,
            )
            left = np.asarray(base_params[: self.n_left_orbital_rotation_params], dtype=np.float64)
            diag = np.asarray(
                base_params[
                    self.n_left_orbital_rotation_params : self.n_left_orbital_rotation_params + self.n_diag_params
                ],
                dtype=np.float64,
            )
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
        left = np.asarray(base_params[: self.n_left_orbital_rotation_params], dtype=np.float64)
        diag = np.asarray(
            base_params[
                self.n_left_orbital_rotation_params : self.n_left_orbital_rotation_params + self.n_diag_params
            ],
            dtype=np.float64,
        )
        doci_reference = np.zeros(self.n_doci_reference_params, dtype=np.float64)
        middle = np.zeros(self.n_middle_orbital_rotation_params, dtype=np.float64)
        return np.concatenate([left, diag, doci_reference, middle])

    def params_to_vec(self, reference_vec: np.ndarray, nelec: tuple[int, int]) -> Callable[[np.ndarray], np.ndarray]:
        reference_vec = np.asarray(reference_vec, dtype=np.complex128)
        def func(params: np.ndarray) -> np.ndarray:
            return self.ansatz_from_parameters(params).apply(reference_vec, nelec=nelec, copy=True)
        return func
