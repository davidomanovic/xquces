from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from xquces.states import doci_dimension, doci_params_from_state, doci_state


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
