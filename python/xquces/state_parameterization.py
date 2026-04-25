from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from xquces.gcr.restricted_jacobian_ext import (
    make_restricted_gcr_jacobian,
    make_restricted_gcr_subspace_jacobian,
)
from xquces.states import doci_dimension, doci_params_from_state, doci_state, doci_state_jacobian


@dataclass(frozen=True)
class DOCIStateParameterization:
    norb: int
    nelec: tuple[int, int]

    @property
    def n_params(self) -> int:
        return doci_dimension(self.norb, self.nelec) - 1

    def state_from_parameters(self, params: np.ndarray) -> np.ndarray:
        params = np.asarray(params, dtype=np.float64)
        if params.shape != (self.n_params,):
            raise ValueError(f"Expected {(self.n_params,)}, got {params.shape}.")
        return doci_state(self.norb, self.nelec, params=params)

    def state_jacobian_from_parameters(self, params: np.ndarray) -> np.ndarray:
        params = np.asarray(params, dtype=np.float64)
        if params.shape != (self.n_params,):
            raise ValueError(f"Expected {(self.n_params,)}, got {params.shape}.")
        return doci_state_jacobian(self.norb, self.nelec, params)

    def parameters_from_state(self, state: np.ndarray) -> np.ndarray:
        return doci_params_from_state(state, self.norb, self.nelec)

    def params_to_state(self) -> Callable[[np.ndarray], np.ndarray]:
        def func(params: np.ndarray) -> np.ndarray:
            return self.state_from_parameters(params)

        return func


@dataclass(frozen=True)
class CompositeReferenceAnsatzParameterization:
    reference_parameterization: object
    ansatz_parameterization: object
    nelec: tuple[int, int]

    @property
    def n_reference_params(self) -> int:
        return int(self.reference_parameterization.n_params)

    @property
    def n_ansatz_params(self) -> int:
        return int(self.ansatz_parameterization.n_params)

    @property
    def n_params(self) -> int:
        return self.n_reference_params + self.n_ansatz_params

    def split_parameters(self, params: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        params = np.asarray(params, dtype=np.float64)
        if params.shape != (self.n_params,):
            raise ValueError(f"Expected {(self.n_params,)}, got {params.shape}.")
        nref = self.n_reference_params
        return params[:nref], params[nref:]

    def reference_state_from_parameters(self, params: np.ndarray) -> np.ndarray:
        return self.reference_parameterization.state_from_parameters(params)

    def ansatz_from_parameters(self, params: np.ndarray):
        return self.ansatz_parameterization.ansatz_from_parameters(params)

    def state_from_parameters(self, params: np.ndarray) -> np.ndarray:
        reference_params, ansatz_params = self.split_parameters(params)
        reference_state = self.reference_state_from_parameters(reference_params)
        ansatz = self.ansatz_from_parameters(ansatz_params)
        return ansatz.apply(reference_state, nelec=self.nelec, copy=True)

    def parameters_from_state_and_ansatz(self, reference_state: np.ndarray, ansatz) -> np.ndarray:
        if not hasattr(self.reference_parameterization, "parameters_from_state"):
            raise TypeError("reference_parameterization does not implement parameters_from_state")
        if not hasattr(self.ansatz_parameterization, "parameters_from_ansatz"):
            raise TypeError("ansatz_parameterization does not implement parameters_from_ansatz")
        reference_params = self.reference_parameterization.parameters_from_state(reference_state)
        ansatz_params = self.ansatz_parameterization.parameters_from_ansatz(ansatz)
        return np.concatenate(
            [
                np.asarray(reference_params, dtype=np.float64),
                np.asarray(ansatz_params, dtype=np.float64),
            ]
        )

    def params_to_vec(self) -> Callable[[np.ndarray], np.ndarray]:
        def func(params: np.ndarray) -> np.ndarray:
            return self.state_from_parameters(params)

        return func


def make_composite_reference_ansatz_vjp(
    parameterization: CompositeReferenceAnsatzParameterization,
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Return a function vjp(params, v) → grad computing 2 Re(J(params)† v).

    Uses O(1) H-applications: only the residual vector v = (H-E)|ψ⟩ enters,
    with no H-applications inside this function.

    Reference gradient: applies the ansatz to each column of J_ref and dots
    with v — avoids materialising the full composite Jacobian.
    Ansatz gradient: builds J_ansatz analytically (no H-apps) and multiplies
    by v.
    """
    reference_parameterization = parameterization.reference_parameterization
    ansatz_parameterization = parameterization.ansatz_parameterization
    nelec = tuple(parameterization.nelec)

    if not hasattr(reference_parameterization, "state_jacobian_from_parameters"):
        raise TypeError(
            "reference_parameterization does not implement state_jacobian_from_parameters"
        )

    def vjp(params: np.ndarray, v: np.ndarray) -> np.ndarray:
        params = np.asarray(params, dtype=np.float64)
        v = np.asarray(v, dtype=np.complex128)
        if params.shape != (parameterization.n_params,):
            raise ValueError(f"Expected {(parameterization.n_params,)}, got {params.shape}.")
        reference_params, ansatz_params = parameterization.split_parameters(params)

        # Reference gradient: ∂E/∂θ_r^k = 2 Re(⟨U_a J_ref[:,k] | v⟩)
        # Computed column-by-column to avoid materialising the full ref_block.
        reference_jac = reference_parameterization.state_jacobian_from_parameters(reference_params)
        ansatz = ansatz_parameterization.ansatz_from_parameters(ansatz_params)
        n_ref = reference_jac.shape[1]
        if n_ref:
            grad_ref = np.array([
                2.0 * float(np.real(np.vdot(
                    ansatz.apply(reference_jac[:, k], nelec=nelec, copy=True), v
                )))
                for k in range(n_ref)
            ])
        else:
            grad_ref = np.zeros(0)

        # Ansatz gradient: build J_ansatz analytically (no H-apps), then J_ansatz† v
        reference_state = reference_parameterization.state_from_parameters(reference_params)
        n_ansatz = parameterization.n_ansatz_params
        if n_ansatz:
            J_ansatz = make_restricted_gcr_jacobian(
                ansatz_parameterization, reference_state, nelec
            )(ansatz_params)
            grad_ansatz = 2.0 * (J_ansatz.conj().T @ v).real
        else:
            grad_ansatz = np.zeros(0)

        return np.concatenate([grad_ref, grad_ansatz])

    return vjp


def make_composite_reference_ansatz_jacobian(
    parameterization: CompositeReferenceAnsatzParameterization,
) -> Callable[[np.ndarray], np.ndarray]:
    reference_parameterization = parameterization.reference_parameterization
    ansatz_parameterization = parameterization.ansatz_parameterization
    nelec = tuple(parameterization.nelec)

    if not hasattr(reference_parameterization, "state_jacobian_from_parameters"):
        raise TypeError(
            "reference_parameterization does not implement state_jacobian_from_parameters"
        )

    def jac(params: np.ndarray) -> np.ndarray:
        params = np.asarray(params, dtype=np.float64)
        if params.shape != (parameterization.n_params,):
            raise ValueError(f"Expected {(parameterization.n_params,)}, got {params.shape}.")
        reference_params, ansatz_params = parameterization.split_parameters(params)
        reference_state = reference_parameterization.state_from_parameters(reference_params)
        reference_jac = reference_parameterization.state_jacobian_from_parameters(reference_params)
        ansatz = ansatz_parameterization.ansatz_from_parameters(ansatz_params)

        if reference_jac.shape[1]:
            ref_block = np.column_stack(
                [
                    ansatz.apply(reference_jac[:, k], nelec=nelec, copy=True)
                    for k in range(reference_jac.shape[1])
                ]
            )
        else:
            ref_block = np.zeros((reference_state.size, 0), dtype=np.complex128)

        ansatz_block = make_restricted_gcr_jacobian(
            ansatz_parameterization,
            reference_state,
            nelec,
        )(ansatz_params)
        return np.hstack([ref_block, ansatz_block])

    return jac


def make_composite_reference_ansatz_subspace_jacobian(
    parameterization: CompositeReferenceAnsatzParameterization,
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    reference_parameterization = parameterization.reference_parameterization
    ansatz_parameterization = parameterization.ansatz_parameterization
    nelec = tuple(parameterization.nelec)

    if not hasattr(reference_parameterization, "state_jacobian_from_parameters"):
        raise TypeError(
            "reference_parameterization does not implement state_jacobian_from_parameters"
        )

    def subspace_jac(params: np.ndarray, directions: np.ndarray) -> np.ndarray:
        params = np.asarray(params, dtype=np.float64)
        if params.shape != (parameterization.n_params,):
            raise ValueError(f"Expected {(parameterization.n_params,)}, got {params.shape}.")
        directions = np.asarray(directions, dtype=np.float64)
        if directions.ndim != 2 or directions.shape[0] != parameterization.n_params:
            raise ValueError(
                "directions must have shape "
                f"({parameterization.n_params}, m); got {directions.shape}."
            )
        n_dir = directions.shape[1]
        reference_params, ansatz_params = parameterization.split_parameters(params)
        nref = parameterization.n_reference_params
        reference_dirs = directions[:nref]
        ansatz_dirs = directions[nref:]

        reference_state = reference_parameterization.state_from_parameters(
            reference_params
        )
        ansatz = ansatz_parameterization.ansatz_from_parameters(ansatz_params)
        out = np.zeros((reference_state.size, n_dir), dtype=np.complex128)

        if reference_dirs.size and np.any(reference_dirs):
            reference_jac = reference_parameterization.state_jacobian_from_parameters(
                reference_params
            )
            reference_jvp = reference_jac @ reference_dirs
            for k in range(n_dir):
                out[:, k] += ansatz.apply(
                    reference_jvp[:, k],
                    nelec=nelec,
                    copy=True,
                )

        if ansatz_dirs.size and np.any(ansatz_dirs):
            out += make_restricted_gcr_subspace_jacobian(
                ansatz_parameterization,
                reference_state,
                nelec,
            )(ansatz_params, ansatz_dirs)

        return out

    return subspace_jac
