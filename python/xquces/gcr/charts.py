from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg

from xquces.gcr.utils import (
    _left_parameters_and_right_phase_from_unitary,
    _left_parameters_from_unitary,
    _left_unitary_from_parameters,
    _parameters_from_zero_diag_antihermitian,
    _zero_diag_antihermitian_from_parameters,
    exact_reference_ov_params_from_unitary,
)
from xquces.ucj._unitary import (
    antihermitian_from_parameters,
    parameters_from_antihermitian,
)
from xquces.ucj.parameterization import ov_final_unitary


@dataclass(frozen=True)
class GCR2FullUnitaryChart:
    """Raw anti-Hermitian chart for a full spin-restricted orbital rotation."""

    def n_params(self, norb: int) -> int:
        return norb**2

    def unitary_from_parameters(self, params: np.ndarray, norb: int) -> np.ndarray:
        return np.asarray(
            scipy.linalg.expm(antihermitian_from_parameters(params, norb)),
            dtype=np.complex128,
        )

    def parameters_from_unitary(self, unitary: np.ndarray) -> np.ndarray:
        unitary = np.asarray(unitary, dtype=np.complex128)
        if unitary.ndim != 2 or unitary.shape[0] != unitary.shape[1]:
            raise ValueError("unitary must be square")
        if not np.allclose(
            unitary.conj().T @ unitary,
            np.eye(unitary.shape[0]),
            atol=1e-10,
        ):
            raise ValueError("unitary must be unitary")
        generator = scipy.linalg.logm(unitary)
        generator = 0.5 * (generator - generator.conj().T)
        return parameters_from_antihermitian(generator)


@dataclass(frozen=True)
class GCR2TraceFixedFullUnitaryChart:
    """Full spin-restricted orbital rotation with the global phase removed.

    The omitted direction is the trace of the anti-Hermitian generator.  On a
    fixed-electron-number state it contributes only a global phase, while the
    remaining ``norb - 1`` diagonal phase differences are still variationally
    meaningful for correlated references such as pUCCD.
    """

    def n_params(self, norb: int) -> int:
        return norb**2 - 1

    def unitary_from_parameters(self, params: np.ndarray, norb: int) -> np.ndarray:
        params = np.asarray(params, dtype=np.float64)
        expected = self.n_params(norb)
        if params.shape != (expected,):
            raise ValueError(f"Expected {(expected,)}, got {params.shape}.")
        diag = np.zeros(norb, dtype=np.float64)
        if norb:
            diag[:-1] = params[: norb - 1]
            diag[-1] = -float(np.sum(diag[:-1]))
        full = np.concatenate([diag, params[norb - 1 :]])
        return np.asarray(
            scipy.linalg.expm(antihermitian_from_parameters(full, norb)),
            dtype=np.complex128,
        )

    def parameters_from_unitary(self, unitary: np.ndarray) -> np.ndarray:
        unitary = np.asarray(unitary, dtype=np.complex128)
        if unitary.ndim != 2 or unitary.shape[0] != unitary.shape[1]:
            raise ValueError("unitary must be square")
        if not np.allclose(
            unitary.conj().T @ unitary,
            np.eye(unitary.shape[0]),
            atol=1e-10,
        ):
            raise ValueError("unitary must be unitary")
        generator = scipy.linalg.logm(unitary)
        generator = 0.5 * (generator - generator.conj().T)
        diag = np.array(np.imag(np.diag(generator)), dtype=np.float64, copy=True)
        diag -= float(np.mean(diag))
        for p, value in enumerate(diag):
            generator[p, p] = 1j * value
        full = parameters_from_antihermitian(generator)
        return np.concatenate([full[: unitary.shape[0] - 1], full[unitary.shape[0] :]])


@dataclass(frozen=True)
class IGCR2LeftUnitaryChart:
    def n_params(self, norb: int) -> int:
        return norb * (norb - 1)

    def unitary_from_parameters(self, params: np.ndarray, norb: int) -> np.ndarray:
        return _left_unitary_from_parameters(params, norb)

    def parameters_from_unitary(self, u: np.ndarray) -> np.ndarray:
        return _left_parameters_from_unitary(u)

    def parameters_and_right_phase_from_unitary(
        self, u: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
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

    def unitary_from_parameters(
        self, params: np.ndarray, norb: int | None = None
    ) -> np.ndarray:
        if norb is not None and norb != self.norb:
            raise ValueError(f"norb={norb} does not match chart norb={self.norb}")
        params = np.asarray(params, dtype=np.float64)
        n_oo = self.nocc * (self.nocc - 1)
        n_vv = self.nvirt * (self.nvirt - 1)
        u = np.eye(self.norb, dtype=np.complex128)
        if self.nocc >= 1:
            kappa_oo = _zero_diag_antihermitian_from_parameters(
                params[:n_oo], self.nocc
            )
            u[: self.nocc, : self.nocc] = np.asarray(
                scipy.linalg.expm(kappa_oo), dtype=np.complex128
            )
        if self.nvirt >= 1:
            kappa_vv = _zero_diag_antihermitian_from_parameters(
                params[n_oo : n_oo + n_vv], self.nvirt
            )
            u[self.nocc :, self.nocc :] = np.asarray(
                scipy.linalg.expm(kappa_vv), dtype=np.complex128
            )
        return u

    def parameters_and_right_phase_from_unitary(
        self, u: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
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
                u_block = (
                    np.array([[val / abs(val)]], dtype=np.complex128)
                    if abs(val) > 1e-14
                    else np.eye(1, dtype=np.complex128)
                )
            else:
                u_block, _ = scipy.linalg.polar(block, side="right")
            p, ph = _left_parameters_and_right_phase_from_unitary(u_block)
            params_parts.append(p)
            phase_parts.append(ph)
        params = (
            np.concatenate(params_parts)
            if params_parts
            else np.zeros(0, dtype=np.float64)
        )
        right_phase = (
            np.concatenate(phase_parts)
            if phase_parts
            else np.zeros(self.norb, dtype=np.float64)
        )
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

    def unitary_from_parameters(
        self, params: np.ndarray, norb: int | None = None
    ) -> np.ndarray:
        if norb is not None and norb != self.norb:
            raise ValueError("norb does not match chart dimensions")
        return ov_final_unitary(
            np.asarray(params, dtype=np.float64), self.norb, self.nocc
        )

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

    def unitary_from_parameters(
        self, params: np.ndarray, norb: int | None = None
    ) -> np.ndarray:
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


__all__ = [
    "GCR2FullUnitaryChart",
    "GCR2TraceFixedFullUnitaryChart",
    "IGCR2BlockDiagLeftUnitaryChart",
    "IGCR2LeftUnitaryChart",
    "IGCR2RealReferenceOVUnitaryChart",
    "IGCR2ReferenceOVUnitaryChart",
]
