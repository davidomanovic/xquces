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
    relabel_igcr2_ansatz_orbitals,
)
from xquces.orbitals import apply_orbital_rotation
from xquces.ucj.model import UCJAnsatz


def _canonicalize_real_coefficients(coefficients: np.ndarray) -> np.ndarray:
    coeffs = np.asarray(coefficients, dtype=np.float64)
    if coeffs.ndim != 1:
        raise ValueError("coefficients must be one-dimensional")
    norm = float(np.linalg.norm(coeffs))
    if norm == 0.0:
        raise ValueError("coefficients must be nonzero")
    out = coeffs / norm
    for value in out:
        if abs(value) > 1e-14:
            if value < 0.0:
                out = -out
            break
    return out


def _real_coefficients_from_parameters(
    n_references: int,
    params: np.ndarray,
) -> np.ndarray:
    expected = max(n_references - 1, 0)
    params = np.asarray(params, dtype=np.float64)
    if params.shape != (expected,):
        raise ValueError(f"Expected {(expected,)}, got {params.shape}.")
    if n_references == 1:
        return np.ones(1, dtype=np.float64)
    coeffs = np.empty(n_references, dtype=np.float64)
    running = 1.0
    for k, theta in enumerate(params):
        coeffs[k] = running * np.cos(theta)
        running *= np.sin(theta)
    coeffs[-1] = running
    return coeffs


def _real_coefficients_parameters_from_coefficients(
    coefficients: np.ndarray,
) -> np.ndarray:
    coeffs = _canonicalize_real_coefficients(coefficients)
    n_references = coeffs.size
    if n_references == 1:
        return np.zeros(0, dtype=np.float64)
    params = np.zeros(n_references - 1, dtype=np.float64)
    for k in range(n_references - 2):
        tail_norm = float(np.linalg.norm(coeffs[k + 1 :]))
        if abs(coeffs[k]) < 1e-14 and tail_norm < 1e-14:
            params[k] = 0.0
        else:
            params[k] = float(np.arctan2(tail_norm, coeffs[k]))
    params[-1] = float(np.arctan2(coeffs[-1], coeffs[-2]))
    return params


def _phase_fixed_vector(vec: np.ndarray) -> np.ndarray:
    out = np.asarray(vec, dtype=np.complex128).copy()
    nz = np.flatnonzero(np.abs(out) > 1e-14)
    if nz.size:
        phase = np.exp(-1j * np.angle(out[int(nz[0])]))
        out *= phase
    return out


@dataclass(frozen=True)
class GCR2NOCIReferenceAnsatz:
    diag_params: np.ndarray
    reference_coefficients: np.ndarray
    left: np.ndarray
    references: tuple[np.ndarray, ...]
    norb: int
    nocc: int
    diag_pairs: tuple[tuple[int, int], ...]

    @property
    def pair_params(self) -> np.ndarray:
        return self.diag_params

    @property
    def n_references(self) -> int:
        return len(self.references)

    def apply(self, vec: np.ndarray, nelec: tuple[int, int], copy: bool = True) -> np.ndarray:
        reference_vec = np.array(vec, dtype=np.complex128, copy=copy)
        accum = np.zeros_like(reference_vec)
        for coeff, right in zip(self.reference_coefficients, self.references):
            branch = apply_orbital_rotation(
                reference_vec,
                right,
                self.norb,
                nelec,
                copy=True,
            )
            accum += float(coeff) * branch
        phases = _diag2_features(self.norb, nelec, self.diag_pairs) @ self.diag_params
        accum *= np.exp(1j * phases)
        out = apply_orbital_rotation(
            accum,
            self.left,
            self.norb,
            nelec,
            copy=False,
        )
        norm = float(np.linalg.norm(out))
        if norm == 0.0:
            raise ValueError("NOCI reference superposition has zero norm")
        return out / norm


@dataclass(frozen=True)
class GCR2NOCIReferenceParameterization:
    norb: int
    nocc: int
    n_references: int
    interaction_pairs: list[tuple[int, int]] | None = None
    base_parameterization: IGCR2SpinRestrictedParameterization | None = None
    left_right_ov_relative_scale: float | None = 1.0
    real_reference_orbital_chart: bool = False
    _interaction_pairs: tuple[tuple[int, int], ...] = field(init=False, repr=False)

    def __post_init__(self):
        if self.n_references < 1:
            raise ValueError("n_references must be at least one")
        if self.base_parameterization is not None:
            if self.base_parameterization.norb != self.norb:
                raise ValueError("base_parameterization.norb does not match")
            if self.base_parameterization.nocc != self.nocc:
                raise ValueError("base_parameterization.nocc does not match")
            base_pairs = tuple(self.base_parameterization.pair_indices)
        else:
            base_pairs = tuple(
                _validate_pairs(self.interaction_pairs, self.norb, allow_diagonal=False)
            )
        object.__setattr__(self, "_interaction_pairs", base_pairs)

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
            real_right_orbital_chart=self.real_reference_orbital_chart,
            left_right_ov_relative_scale=self.left_right_ov_relative_scale,
        )

    @property
    def right_orbital_chart(self):
        return self._base.right_orbital_chart

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
    def n_reference_coeff_params(self) -> int:
        return self.n_references - 1

    @property
    def n_reference_orbital_rotation_params_per_reference(self) -> int:
        return self.right_orbital_chart.n_params(self.norb)

    @property
    def n_reference_orbital_rotation_params(self) -> int:
        return self.n_references * self.n_reference_orbital_rotation_params_per_reference

    @property
    def n_right_orbital_rotation_params(self) -> int:
        return self.n_reference_orbital_rotation_params

    @property
    def n_active_params(self) -> int:
        return (
            self.n_left_orbital_rotation_params
            + self.n_diag_params
            + self.n_reference_orbital_rotation_params
        )

    @property
    def n_params(self) -> int:
        return (
            self.n_left_orbital_rotation_params
            + self.n_diag_params
            + self.n_reference_coeff_params
            + self.n_reference_orbital_rotation_params
        )

    def reference_coefficients_from_parameters(self, params: np.ndarray) -> np.ndarray:
        return _real_coefficients_from_parameters(self.n_references, params)

    def reference_coeff_params_from_coefficients(
        self,
        coefficients: np.ndarray,
    ) -> np.ndarray:
        coeffs = np.asarray(coefficients, dtype=np.float64)
        if coeffs.shape != (self.n_references,):
            raise ValueError(f"Expected {(self.n_references,)}, got {coeffs.shape}.")
        return _real_coefficients_parameters_from_coefficients(coeffs)

    def split_parameters(
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
        n = self.n_diag_params
        diag = params[idx : idx + n]
        idx += n
        n = self.n_reference_coeff_params
        coeff = params[idx : idx + n]
        idx += n
        n = self.n_reference_orbital_rotation_params_per_reference
        references = np.asarray(params[idx:], dtype=np.float64).reshape(self.n_references, n)
        return left, diag, coeff, references

    def combine_parameters(
        self,
        left: np.ndarray,
        diag: np.ndarray,
        reference_coeff_params: np.ndarray,
        reference_orbital_rotation_params: np.ndarray,
    ) -> np.ndarray:
        left = np.asarray(left, dtype=np.float64)
        diag = np.asarray(diag, dtype=np.float64)
        coeff = np.asarray(reference_coeff_params, dtype=np.float64)
        refs = np.asarray(reference_orbital_rotation_params, dtype=np.float64)
        expected_refs = (
            self.n_references,
            self.n_reference_orbital_rotation_params_per_reference,
        )
        if left.shape != (self.n_left_orbital_rotation_params,):
            raise ValueError(
                f"Expected {(self.n_left_orbital_rotation_params,)}, got {left.shape}."
            )
        if diag.shape != (self.n_diag_params,):
            raise ValueError(f"Expected {(self.n_diag_params,)}, got {diag.shape}.")
        if coeff.shape != (self.n_reference_coeff_params,):
            raise ValueError(
                f"Expected {(self.n_reference_coeff_params,)}, got {coeff.shape}."
            )
        if refs.shape != expected_refs:
            raise ValueError(f"Expected {expected_refs}, got {refs.shape}.")
        return np.concatenate([left, diag, coeff, refs.reshape(-1)])

    def split_active_parameters(
        self,
        params: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        params = np.asarray(params, dtype=np.float64)
        if params.shape != (self.n_active_params,):
            raise ValueError(f"Expected {(self.n_active_params,)}, got {params.shape}.")
        idx = 0
        n = self.n_left_orbital_rotation_params
        left = params[idx : idx + n]
        idx += n
        n = self.n_diag_params
        diag = params[idx : idx + n]
        idx += n
        n = self.n_reference_orbital_rotation_params_per_reference
        references = np.asarray(params[idx:], dtype=np.float64).reshape(self.n_references, n)
        return left, diag, references

    def active_parameters_from_parameters(self, params: np.ndarray) -> np.ndarray:
        left, diag, _, references = self.split_parameters(params)
        return np.concatenate([left, diag, references.reshape(-1)])

    def parameters_from_active_parameters(
        self,
        active_parameters: np.ndarray,
        reference_coeff_params: np.ndarray | None = None,
    ) -> np.ndarray:
        left, diag, references = self.split_active_parameters(active_parameters)
        if reference_coeff_params is None:
            reference_coeff_params = np.zeros(self.n_reference_coeff_params, dtype=np.float64)
        return self.combine_parameters(left, diag, reference_coeff_params, references)

    def _base_params_from_split(
        self,
        left: np.ndarray,
        diag: np.ndarray,
        right: np.ndarray,
    ) -> np.ndarray:
        return np.concatenate([left, diag, right]).astype(np.float64, copy=False)

    def _identity_orbital_rotation(self) -> np.ndarray:
        return np.eye(self.norb, dtype=np.complex128)

    def _reference_params_from_unitary(self, unitary: np.ndarray) -> np.ndarray:
        return np.asarray(
            self.right_orbital_chart.parameters_from_unitary(
                np.asarray(unitary, dtype=np.complex128)
            ),
            dtype=np.float64,
        )

    def _reference_unitary_from_params(self, params: np.ndarray) -> np.ndarray:
        return np.asarray(
            self.right_orbital_chart.unitary_from_parameters(
                np.asarray(params, dtype=np.float64),
                self.norb,
            ),
            dtype=np.complex128,
        )

    def _shared_left_and_diag_from_parameters(
        self,
        params: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        left, diag, _, reference_params = self.split_parameters(params)
        left_dummy = self._base.ansatz_from_parameters(
            self._base_params_from_split(
                left,
                diag,
                np.zeros(self._base.n_right_orbital_rotation_params, dtype=np.float64),
            )
        )
        diag_matrix = np.asarray(left_dummy.diagonal.pair, dtype=np.float64)
        diag_values = np.asarray(
            [diag_matrix[p, q] for p, q in self._interaction_pairs],
            dtype=np.float64,
        )
        return np.asarray(left_dummy.left, dtype=np.complex128), diag_values, reference_params

    def basis_states_from_parameters(
        self,
        params: np.ndarray,
        reference_vec: np.ndarray,
        nelec: tuple[int, int],
    ) -> np.ndarray:
        reference_vec = np.asarray(reference_vec, dtype=np.complex128)
        left, diag_values, reference_params = self._shared_left_and_diag_from_parameters(params)
        phases = _diag2_features(self.norb, nelec, self._interaction_pairs) @ diag_values
        phase_vector = np.exp(1j * phases)
        out = np.empty((reference_vec.size, self.n_references), dtype=np.complex128)
        for k in range(self.n_references):
            right = self._reference_unitary_from_params(reference_params[k])
            branch = apply_orbital_rotation(
                reference_vec,
                right,
                self.norb,
                nelec,
                copy=True,
            )
            branch *= phase_vector
            out[:, k] = apply_orbital_rotation(
                branch,
                left,
                self.norb,
                nelec,
                copy=False,
            )
        return out

    def subspace_matrices_from_parameters(
        self,
        params: np.ndarray,
        reference_vec: np.ndarray,
        nelec: tuple[int, int],
        hamiltonian,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        basis = self.basis_states_from_parameters(params, reference_vec, nelec)
        h_basis = np.column_stack([hamiltonian @ basis[:, k] for k in range(basis.shape[1])])
        s_matrix = basis.conj().T @ basis
        h_matrix = basis.conj().T @ h_basis
        s_matrix = 0.5 * (s_matrix + s_matrix.conj().T)
        h_matrix = 0.5 * (h_matrix + h_matrix.conj().T)
        return basis, h_matrix, s_matrix

    def solve_subspace(
        self,
        params: np.ndarray,
        reference_vec: np.ndarray,
        nelec: tuple[int, int],
        hamiltonian,
        *,
        rtol: float = 1e-10,
        atol: float = 1e-12,
    ) -> dict[str, np.ndarray | float | int]:
        basis, h_matrix, s_matrix = self.subspace_matrices_from_parameters(
            params,
            reference_vec,
            nelec,
            hamiltonian,
        )
        s_evals, s_evecs = np.linalg.eigh(s_matrix)
        s_evals = np.real(s_evals)
        lam_max = max(float(s_evals[-1]), 0.0) if s_evals.size else 0.0
        cutoff = max(float(atol), float(rtol) * lam_max)
        keep = s_evals > cutoff
        if not np.any(keep):
            keep[np.argmax(s_evals)] = True
        active_evals = s_evals[keep]
        x = s_evecs[:, keep] / np.sqrt(active_evals)[np.newaxis, :]
        h_orth = x.conj().T @ h_matrix @ x
        h_orth = 0.5 * (h_orth + h_orth.conj().T)
        evals, evecs = np.linalg.eigh(h_orth)
        energy = float(np.real(evals[0]))
        coeffs = x @ evecs[:, 0]
        coeffs = _phase_fixed_vector(coeffs)
        state = basis @ coeffs
        norm = float(np.linalg.norm(state))
        if norm == 0.0:
            raise ValueError("subspace eigenstate has zero norm")
        coeffs = coeffs / norm
        state = state / norm
        smallest_kept = float(active_evals[0])
        condition = lam_max / max(smallest_kept, cutoff) if lam_max > 0.0 else 1.0
        return {
            "energy": energy,
            "state": state,
            "coefficients": coeffs,
            "basis_states": basis,
            "h_matrix": h_matrix,
            "s_matrix": s_matrix,
            "cutoff": cutoff,
            "subspace_condition": condition,
            "active_modes": int(active_evals.size),
            "dropped_modes": int(self.n_references - active_evals.size),
        }

    def energy_from_parameters(
        self,
        params: np.ndarray,
        reference_vec: np.ndarray,
        nelec: tuple[int, int],
        hamiltonian,
        *,
        rtol: float = 1e-10,
        atol: float = 1e-12,
    ) -> float:
        solved = self.solve_subspace(
            params,
            reference_vec,
            nelec,
            hamiltonian,
            rtol=rtol,
            atol=atol,
        )
        return float(solved["energy"])

    def canonicalize_parameters(
        self,
        params: np.ndarray,
        reference_vec: np.ndarray,
        nelec: tuple[int, int],
        hamiltonian,
        *,
        rtol: float = 1e-10,
        atol: float = 1e-12,
    ) -> np.ndarray:
        params = np.asarray(params, dtype=np.float64)
        left, diag, _, references = self.split_parameters(params)
        solved = self.solve_subspace(
            params,
            reference_vec,
            nelec,
            hamiltonian,
            rtol=rtol,
            atol=atol,
        )
        coeffs = np.asarray(solved["coefficients"], dtype=np.complex128)
        order = np.argsort(-np.abs(coeffs))
        weights = np.abs(coeffs[order])
        if float(np.linalg.norm(weights)) == 0.0:
            weights = np.zeros(self.n_references, dtype=np.float64)
            weights[0] = 1.0
        coeff_params = self.reference_coeff_params_from_coefficients(weights)
        return self.combine_parameters(left, diag, coeff_params, references[order])

    def ansatz_from_parameters(self, params: np.ndarray) -> GCR2NOCIReferenceAnsatz:
        left, diag_values, reference_params = self._shared_left_and_diag_from_parameters(params)
        _, _, coeff_params, _ = self.split_parameters(params)
        coefficients = self.reference_coefficients_from_parameters(coeff_params)
        references = tuple(
            self._reference_unitary_from_params(reference_params[k])
            for k in range(self.n_references)
        )
        return GCR2NOCIReferenceAnsatz(
            diag_params=diag_values,
            reference_coefficients=coefficients,
            left=left,
            references=references,
            norb=self.norb,
            nocc=self.nocc,
            diag_pairs=self._interaction_pairs,
        )

    def parameters_from_ansatz(
        self,
        ansatz: GCR2NOCIReferenceAnsatz | IGCR2Ansatz,
    ) -> np.ndarray:
        if isinstance(ansatz, GCR2NOCIReferenceAnsatz):
            if ansatz.norb != self.norb:
                raise ValueError("ansatz norb does not match parameterization")
            if ansatz.nocc != self.nocc:
                raise ValueError("ansatz nocc does not match parameterization")
            if len(ansatz.references) != self.n_references:
                raise ValueError("ansatz reference count does not match parameterization")
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
            n_left = self.n_left_orbital_rotation_params
            n_diag = self.n_diag_params
            left = np.asarray(base_params[:n_left], dtype=np.float64)
            diag = np.asarray(base_params[n_left : n_left + n_diag], dtype=np.float64)
            coeff = self.reference_coeff_params_from_coefficients(
                np.asarray(ansatz.reference_coefficients, dtype=np.float64)
            )
            refs = np.vstack(
                [self._reference_params_from_unitary(u) for u in ansatz.references]
            )
            return self.combine_parameters(left, diag, coeff, refs)
        if isinstance(ansatz, IGCR2Ansatz):
            if ansatz.norb != self.norb:
                raise ValueError("ansatz norb does not match parameterization")
            base_params = self._base.parameters_from_ansatz(ansatz)
            n_left = self.n_left_orbital_rotation_params
            n_diag = self.n_diag_params
            left = np.asarray(base_params[:n_left], dtype=np.float64)
            diag = np.asarray(base_params[n_left : n_left + n_diag], dtype=np.float64)
            coeff = self.reference_coeff_params_from_coefficients(
                np.array([1.0] + [0.0] * (self.n_references - 1), dtype=np.float64)
            )
            refs = np.zeros(
                (
                    self.n_references,
                    self.n_reference_orbital_rotation_params_per_reference,
                ),
                dtype=np.float64,
            )
            refs[0] = self._reference_params_from_unitary(ansatz.right)
            return self.combine_parameters(left, diag, coeff, refs)
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
        return self.parameters_from_igcr2(self._base.parameters_from_ucj_ansatz(ansatz))

    def transfer_parameters_from(
        self,
        previous_parameters: np.ndarray,
        previous_parameterization: "GCR2NOCIReferenceParameterization | IGCR2SpinRestrictedParameterization | None" = None,
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
        if isinstance(previous_parameterization, GCR2NOCIReferenceParameterization):
            prev_params = np.asarray(previous_parameters, dtype=np.float64)
            if prev_params.shape != (previous_parameterization.n_params,):
                raise ValueError(
                    f"Expected {(previous_parameterization.n_params,)}, got {prev_params.shape}."
                )
            prev_left, prev_diag, prev_coeff, prev_refs = previous_parameterization.split_parameters(
                prev_params
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

            prev_coefficients = previous_parameterization.reference_coefficients_from_parameters(
                prev_coeff
            )
            coefficients = np.zeros(self.n_references, dtype=np.float64)
            n_copy = min(self.n_references, previous_parameterization.n_references)
            coefficients[:n_copy] = prev_coefficients[:n_copy]
            if float(np.linalg.norm(coefficients)) == 0.0:
                coefficients[0] = 1.0
            coeff = self.reference_coeff_params_from_coefficients(coefficients)

            refs = np.zeros(
                (
                    self.n_references,
                    self.n_reference_orbital_rotation_params_per_reference,
                ),
                dtype=np.float64,
            )
            relabel = None
            if old_for_new is not None:
                relabel = _orbital_relabeling_unitary(old_for_new, phases)
            for k in range(n_copy):
                unitary = previous_parameterization._reference_unitary_from_params(
                    prev_refs[k]
                )
                if relabel is not None:
                    unitary = relabel.conj().T @ unitary @ relabel
                refs[k] = self._reference_params_from_unitary(unitary)
            return self.combine_parameters(left, diag, coeff, refs)

        previous_base = self._base if previous_parameterization is None else previous_parameterization
        base_params = self._base.transfer_parameters_from(
            previous_parameters,
            previous_parameterization=previous_base,
            old_for_new=old_for_new,
            phases=phases,
            orbital_overlap=None,
            block_diagonal=block_diagonal,
        )
        n_left = self.n_left_orbital_rotation_params
        n_diag = self.n_diag_params
        left = np.asarray(base_params[:n_left], dtype=np.float64)
        diag = np.asarray(base_params[n_left : n_left + n_diag], dtype=np.float64)
        coeff = self.reference_coeff_params_from_coefficients(
            np.array([1.0] + [0.0] * (self.n_references - 1), dtype=np.float64)
        )
        refs = np.zeros(
            (
                self.n_references,
                self.n_reference_orbital_rotation_params_per_reference,
            ),
            dtype=np.float64,
        )
        ansatz = previous_base.ansatz_from_parameters(previous_parameters)
        if old_for_new is not None:
            ansatz = relabel_igcr2_ansatz_orbitals(ansatz, old_for_new, phases)
        refs[0] = self._reference_params_from_unitary(ansatz.right)
        return self.combine_parameters(left, diag, coeff, refs)

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
