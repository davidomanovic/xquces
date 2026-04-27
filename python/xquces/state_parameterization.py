from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from xquces.gcr.restricted_jacobian_ext import (
    make_restricted_gcr_jacobian,
    make_restricted_gcr_subspace_jacobian,
)
from xquces.states import (
    doci_dimension,
    doci_params_from_state,
    doci_state,
    doci_state_jacobian,
    hartree_fock_state,
)


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

    def state_jacobian_from_parameters(self, params: np.ndarray) -> np.ndarray:
        return make_composite_reference_ansatz_jacobian(self)(params)

    def state_subspace_jacobian_from_parameters(
        self,
        params: np.ndarray,
        directions: np.ndarray,
    ) -> np.ndarray:
        return make_composite_reference_ansatz_subspace_jacobian(self)(
            params,
            directions,
        )

    def parameters_from_state_and_ansatz(
        self, reference_state: np.ndarray, ansatz
    ) -> np.ndarray:
        if not hasattr(self.reference_parameterization, "parameters_from_state"):
            raise TypeError(
                "reference_parameterization does not implement parameters_from_state"
            )
        if not hasattr(self.ansatz_parameterization, "parameters_from_ansatz"):
            raise TypeError(
                "ansatz_parameterization does not implement parameters_from_ansatz"
            )
        reference_params = self.reference_parameterization.parameters_from_state(
            reference_state
        )
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

    def transfer_parameters_from(
        self,
        previous_parameters: np.ndarray,
        previous_parameterization: object | None = None,
        old_for_new: np.ndarray | None = None,
        phases: np.ndarray | None = None,
        orbital_overlap: np.ndarray | None = None,
        block_diagonal: bool = True,
    ) -> np.ndarray:
        if previous_parameterization is None:
            previous_parameterization = self
        previous_parameters = np.asarray(previous_parameters, dtype=np.float64)

        if isinstance(
            previous_parameterization,
            CompositeReferenceAnsatzParameterization,
        ):
            prev_reference, prev_ansatz = previous_parameterization.split_parameters(
                previous_parameters
            )
            previous_reference_parameterization = (
                previous_parameterization.reference_parameterization
            )
            previous_ansatz_parameterization = (
                previous_parameterization.ansatz_parameterization
            )
        else:
            prev_reference = np.zeros(0, dtype=np.float64)
            prev_ansatz = previous_parameters
            previous_reference_parameterization = None
            previous_ansatz_parameterization = previous_parameterization

        if hasattr(self.reference_parameterization, "transfer_parameters_from"):
            reference_params = self.reference_parameterization.transfer_parameters_from(
                prev_reference,
                previous_parameterization=previous_reference_parameterization,
                old_for_new=old_for_new,
                phases=phases,
                orbital_overlap=orbital_overlap,
                block_diagonal=block_diagonal,
            )
        elif (
            previous_reference_parameterization is not None
            and prev_reference.shape == (self.n_reference_params,)
        ):
            reference_params = np.array(prev_reference, copy=True)
        else:
            reference_params = np.zeros(self.n_reference_params, dtype=np.float64)

        ansatz_params = self.ansatz_parameterization.transfer_parameters_from(
            prev_ansatz,
            previous_parameterization=previous_ansatz_parameterization,
            old_for_new=old_for_new,
            phases=phases,
            orbital_overlap=orbital_overlap,
            block_diagonal=block_diagonal,
        )
        return np.concatenate(
            [
                np.asarray(reference_params, dtype=np.float64),
                np.asarray(ansatz_params, dtype=np.float64),
            ]
        )


@dataclass(frozen=True)
class FixedReferenceAnsatzParameterization:
    reference_state: np.ndarray
    ansatz_parameterization: object
    nelec: tuple[int, int]

    def __post_init__(self):
        reference = np.asarray(self.reference_state, dtype=np.complex128)
        if reference.ndim != 1:
            raise ValueError("reference_state must be a one-dimensional state vector")
        object.__setattr__(self, "reference_state", reference)
        object.__setattr__(self, "nelec", tuple(self.nelec))

    @property
    def n_params(self) -> int:
        return int(self.ansatz_parameterization.n_params)

    def ansatz_from_parameters(self, params: np.ndarray):
        return self.ansatz_parameterization.ansatz_from_parameters(params)

    def state_from_parameters(self, params: np.ndarray) -> np.ndarray:
        params = np.asarray(params, dtype=np.float64)
        if params.shape != (self.n_params,):
            raise ValueError(f"Expected {(self.n_params,)}, got {params.shape}.")
        return self.ansatz_from_parameters(params).apply(
            self.reference_state,
            nelec=self.nelec,
            copy=True,
        )

    def state_jacobian_from_parameters(self, params: np.ndarray) -> np.ndarray:
        params = np.asarray(params, dtype=np.float64)
        if params.shape != (self.n_params,):
            raise ValueError(f"Expected {(self.n_params,)}, got {params.shape}.")
        return make_restricted_gcr_jacobian(
            self.ansatz_parameterization,
            self.reference_state,
            self.nelec,
        )(params)

    def state_subspace_jacobian_from_parameters(
        self,
        params: np.ndarray,
        directions: np.ndarray,
    ) -> np.ndarray:
        params = np.asarray(params, dtype=np.float64)
        if params.shape != (self.n_params,):
            raise ValueError(f"Expected {(self.n_params,)}, got {params.shape}.")
        directions = np.asarray(directions, dtype=np.float64)
        if directions.ndim != 2 or directions.shape[0] != self.n_params:
            raise ValueError(
                f"directions must have shape ({self.n_params}, m); got "
                f"{directions.shape}."
            )
        return make_restricted_gcr_subspace_jacobian(
            self.ansatz_parameterization,
            self.reference_state,
            self.nelec,
        )(params, directions)

    def params_to_vec(self) -> Callable[[np.ndarray], np.ndarray]:
        def func(params: np.ndarray) -> np.ndarray:
            return self.state_from_parameters(params)

        return func

    def transfer_parameters_from(
        self,
        previous_parameters: np.ndarray,
        previous_parameterization: object | None = None,
        old_for_new: np.ndarray | None = None,
        phases: np.ndarray | None = None,
        orbital_overlap: np.ndarray | None = None,
        block_diagonal: bool = True,
    ) -> np.ndarray:
        if previous_parameterization is None:
            previous_parameterization = self
        previous_ansatz_parameterization = getattr(
            previous_parameterization,
            "ansatz_parameterization",
            previous_parameterization,
        )
        if isinstance(
            previous_parameterization,
            CompositeReferenceAnsatzParameterization,
        ):
            _, previous_parameters = previous_parameterization.split_parameters(
                previous_parameters
            )
        return self.ansatz_parameterization.transfer_parameters_from(
            previous_parameters,
            previous_parameterization=previous_ansatz_parameterization,
            old_for_new=old_for_new,
            phases=phases,
            orbital_overlap=orbital_overlap,
            block_diagonal=block_diagonal,
        )


def _reference_nelec(reference: object) -> tuple[int, int] | None:
    nelec = getattr(reference, "nelec", None)
    if nelec is None:
        return None
    return tuple(int(x) for x in nelec)


def _resolve_nelec(
    reference: object,
    nelec: tuple[int, int] | None,
) -> tuple[int, int]:
    reference_nelec = _reference_nelec(reference)
    if nelec is None:
        if reference_nelec is None:
            raise ValueError("nelec is required for a fixed reference state vector")
        return reference_nelec
    nelec = tuple(int(x) for x in nelec)
    if reference_nelec is not None and reference_nelec != nelec:
        raise ValueError(
            f"reference nelec {reference_nelec} does not match requested {nelec}"
        )
    return nelec


def reference_is_hartree_fock_state(
    reference: object,
    norb: int,
    nelec: tuple[int, int],
    *,
    atol: float = 1e-10,
) -> bool:
    if hasattr(reference, "state_from_parameters"):
        return False
    reference_state = np.asarray(reference, dtype=np.complex128)
    if reference_state.ndim != 1:
        return False
    hf = hartree_fock_state(norb, nelec)
    if reference_state.shape != hf.shape:
        return False
    norm = np.linalg.norm(reference_state)
    if norm <= 1e-14:
        return False
    return abs(np.vdot(hf, reference_state / norm)) >= 1.0 - atol


def apply_ansatz_parameterization(
    ansatz_parameterization: object,
    reference: object,
    nelec: tuple[int, int] | None = None,
):
    nelec = _resolve_nelec(reference, nelec)
    if hasattr(reference, "state_from_parameters"):
        return CompositeReferenceAnsatzParameterization(
            reference_parameterization=reference,
            ansatz_parameterization=ansatz_parameterization,
            nelec=nelec,
        )
    return FixedReferenceAnsatzParameterization(
        reference_state=np.asarray(reference, dtype=np.complex128),
        ansatz_parameterization=ansatz_parameterization,
        nelec=nelec,
    )


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
            raise ValueError(
                f"Expected {(parameterization.n_params,)}, got {params.shape}."
            )
        reference_params, ansatz_params = parameterization.split_parameters(params)

        # Reference gradient: ∂E/∂θ_r^k = 2 Re(⟨U_a J_ref[:,k] | v⟩)
        # Computed column-by-column to avoid materialising the full ref_block.
        reference_jac = reference_parameterization.state_jacobian_from_parameters(
            reference_params
        )
        ansatz = ansatz_parameterization.ansatz_from_parameters(ansatz_params)
        n_ref = reference_jac.shape[1]
        if n_ref:
            grad_ref = np.array(
                [
                    2.0
                    * float(
                        np.real(
                            np.vdot(
                                ansatz.apply(
                                    reference_jac[:, k], nelec=nelec, copy=True
                                ),
                                v,
                            )
                        )
                    )
                    for k in range(n_ref)
                ]
            )
        else:
            grad_ref = np.zeros(0)

        # Ansatz gradient: build J_ansatz analytically (no H-apps), then J_ansatz† v
        reference_state = reference_parameterization.state_from_parameters(
            reference_params
        )
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
            raise ValueError(
                f"Expected {(parameterization.n_params,)}, got {params.shape}."
            )
        reference_params, ansatz_params = parameterization.split_parameters(params)
        reference_state = reference_parameterization.state_from_parameters(
            reference_params
        )
        reference_jac = reference_parameterization.state_jacobian_from_parameters(
            reference_params
        )
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
            raise ValueError(
                f"Expected {(parameterization.n_params,)}, got {params.shape}."
            )
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
