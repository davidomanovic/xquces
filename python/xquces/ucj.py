from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .gates import apply_ucj_spin_balanced, apply_ucj_spin_restricted
from .orbitals import apply_orbital_rotation, canonicalize_unitary, ov_unitary_from_t1, unitary_from_generator


def pair_params_from_t2(t2: np.ndarray) -> np.ndarray:
    t2 = np.asarray(t2, dtype=np.float64)
    if t2.ndim != 4:
        raise ValueError("t2 must have shape (nocc, nocc, nvirt, nvirt)")
    nocc1, nocc2, nvirt1, nvirt2 = t2.shape
    if nocc1 != nocc2 or nvirt1 != nvirt2:
        raise ValueError("t2 must have shape (nocc, nocc, nvirt, nvirt)")
    nocc = nocc1
    nvirt = nvirt1
    norb = nocc + nvirt
    pair = np.zeros((norb, norb), dtype=np.float64)

    for i in range(nocc):
        for j in range(i + 1, nocc):
            val = np.linalg.norm(t2[i, j])
            pair[i, j] = val
            pair[j, i] = val

    for a in range(nvirt):
        for b in range(a + 1, nvirt):
            val = np.linalg.norm(t2[:, :, a, b])
            pair[nocc + a, nocc + b] = val
            pair[nocc + b, nocc + a] = val

    for i in range(nocc):
        for a in range(nvirt):
            val = np.linalg.norm(t2[i, :, a, :])
            pair[i, nocc + a] = val
            pair[nocc + a, i] = val

    np.fill_diagonal(pair, 0.0)
    mx = np.max(np.abs(pair))
    if mx > 0:
        pair = pair / mx
    return pair


@dataclass(frozen=True)
class SpinRestrictedSpec:
    double_params: np.ndarray
    pair_params: np.ndarray

    def __post_init__(self):
        object.__setattr__(
            self,
            "double_params",
            np.asarray(self.double_params, dtype=np.float64),
        )
        object.__setattr__(
            self,
            "pair_params",
            np.asarray(self.pair_params, dtype=np.float64),
        )
        norb = self.double_params.shape[0]
        if self.double_params.shape != (norb,):
            raise ValueError("double_params must have shape (norb,)")
        if self.pair_params.shape != (norb, norb):
            raise ValueError("pair_params must have shape (norb, norb)")
        if not np.allclose(self.pair_params, self.pair_params.T):
            raise ValueError("pair_params must be symmetric")
        pair = np.array(self.pair_params, copy=True)
        np.fill_diagonal(pair, 0.0)
        object.__setattr__(self, "pair_params", pair)

    @property
    def norb(self) -> int:
        return self.double_params.shape[0]


@dataclass(frozen=True)
class SpinBalancedSpec:
    same_spin_params: np.ndarray
    mixed_spin_params: np.ndarray

    def __post_init__(self):
        object.__setattr__(
            self,
            "same_spin_params",
            np.asarray(self.same_spin_params, dtype=np.float64),
        )
        object.__setattr__(
            self,
            "mixed_spin_params",
            np.asarray(self.mixed_spin_params, dtype=np.float64),
        )
        norb = self.same_spin_params.shape[0]
        if self.same_spin_params.shape != (norb, norb):
            raise ValueError("same_spin_params must have shape (norb, norb)")
        if self.mixed_spin_params.shape != (norb, norb):
            raise ValueError("mixed_spin_params must have shape (norb, norb)")
        if not np.allclose(self.same_spin_params, self.same_spin_params.T):
            raise ValueError("same_spin_params must be symmetric")
        if not np.allclose(self.mixed_spin_params, self.mixed_spin_params.T):
            raise ValueError("mixed_spin_params must be symmetric")

    @property
    def norb(self) -> int:
        return self.same_spin_params.shape[0]


@dataclass(frozen=True)
class UCJLayer:
    diagonal: SpinRestrictedSpec | SpinBalancedSpec
    orbital_rotation: np.ndarray

    def __post_init__(self):
        u = canonicalize_unitary(np.asarray(self.orbital_rotation, dtype=np.complex128))
        norb = self.diagonal.norb
        if u.shape != (norb, norb):
            raise ValueError("orbital_rotation has wrong shape")
        if not np.allclose(u.conj().T @ u, np.eye(norb), atol=1e-10):
            raise ValueError("orbital_rotation must be unitary")
        object.__setattr__(self, "orbital_rotation", u)

    @classmethod
    def from_generator(
        cls,
        diagonal: SpinRestrictedSpec | SpinBalancedSpec,
        generator: np.ndarray,
    ) -> "UCJLayer":
        return cls(
            diagonal=diagonal,
            orbital_rotation=unitary_from_generator(generator, gauge_fix=True),
        )

    @property
    def norb(self) -> int:
        return self.diagonal.norb


@dataclass(frozen=True)
class UCJAnsatz:
    layers: tuple[UCJLayer, ...]
    final_orbital_rotation: np.ndarray | None = None

    def __post_init__(self):
        if len(self.layers) == 0:
            raise ValueError("at least one layer is required")
        norb = self.layers[0].norb
        for layer in self.layers:
            if layer.norb != norb:
                raise ValueError("all layers must have the same norb")
        if self.final_orbital_rotation is not None:
            u = canonicalize_unitary(
                np.asarray(self.final_orbital_rotation, dtype=np.complex128)
            )
            if u.shape != (norb, norb):
                raise ValueError("final_orbital_rotation has wrong shape")
            if not np.allclose(u.conj().T @ u, np.eye(norb), atol=1e-10):
                raise ValueError("final_orbital_rotation must be unitary")
            object.__setattr__(self, "final_orbital_rotation", u)

    @property
    def norb(self) -> int:
        return self.layers[0].norb

    @classmethod
    def identity(cls, norb: int, spin_restricted: bool = True) -> "UCJAnsatz":
        if spin_restricted:
            diag = SpinRestrictedSpec(
                double_params=np.zeros(norb),
                pair_params=np.zeros((norb, norb)),
            )
        else:
            diag = SpinBalancedSpec(
                same_spin_params=np.zeros((norb, norb)),
                mixed_spin_params=np.zeros((norb, norb)),
            )
        layer = UCJLayer(diagonal=diag, orbital_rotation=np.eye(norb, dtype=np.complex128))
        return cls(layers=(layer,))

    @classmethod
    def from_ov_rotation(cls, t1: np.ndarray) -> "UCJAnsatz":
        t1 = np.asarray(t1, dtype=np.complex128)
        if t1.ndim != 2:
            raise ValueError("t1 must have shape (nocc, nvirt)")
        nocc, nvirt = t1.shape
        norb = nocc + nvirt
        diag = SpinRestrictedSpec(
            double_params=np.zeros(norb),
            pair_params=np.zeros((norb, norb)),
        )
        layer = UCJLayer(diagonal=diag, orbital_rotation=np.eye(norb, dtype=np.complex128))
        return cls(
            layers=(layer,),
            final_orbital_rotation=ov_unitary_from_t1(t1),
        )

    @classmethod
    def from_t_amplitudes(
        cls,
        t2: np.ndarray,
        t1: np.ndarray | None = None,
        pair_scale: float = 1.0,
        double_params: np.ndarray | None = None,
    ) -> "UCJAnsatz":
        t2 = np.asarray(t2, dtype=np.float64)
        if t2.ndim != 4:
            raise ValueError("t2 must have shape (nocc, nocc, nvirt, nvirt)")
        nocc, _, nvirt, _ = t2.shape
        norb = nocc + nvirt

        pair = pair_scale * pair_params_from_t2(t2)
        if double_params is None:
            double_params = np.zeros(norb, dtype=np.float64)
        else:
            double_params = np.asarray(double_params, dtype=np.float64)
            if double_params.shape != (norb,):
                raise ValueError("double_params must have shape (norb,)")

        diag = SpinRestrictedSpec(
            double_params=double_params,
            pair_params=pair,
        )
        layer = UCJLayer(diagonal=diag, orbital_rotation=np.eye(norb, dtype=np.complex128))

        final = None
        if t1 is not None:
            t1 = np.asarray(t1, dtype=np.complex128)
            if t1.shape != (nocc, nvirt):
                raise ValueError("t1 must have shape (nocc, nvirt)")
            final = ov_unitary_from_t1(t1)

        return cls(
            layers=(layer,),
            final_orbital_rotation=final,
        )

    def apply(
        self,
        vec: np.ndarray,
        nelec: tuple[int, int],
        copy: bool = True,
    ) -> np.ndarray:
        out = np.array(vec, dtype=np.complex128, copy=copy)
        for layer in self.layers:
            d = layer.diagonal
            if isinstance(d, SpinRestrictedSpec):
                out = apply_ucj_spin_restricted(
                    out,
                    double_params=d.double_params,
                    pair_params=d.pair_params,
                    norb=self.norb,
                    nelec=nelec,
                    orbital_rotation=layer.orbital_rotation,
                    copy=False,
                )
            else:
                out = apply_ucj_spin_balanced(
                    out,
                    same_spin_params=d.same_spin_params,
                    mixed_spin_params=d.mixed_spin_params,
                    norb=self.norb,
                    nelec=nelec,
                    orbital_rotation=layer.orbital_rotation,
                    copy=False,
                )
        if self.final_orbital_rotation is not None:
            out = apply_orbital_rotation(
                out,
                self.final_orbital_rotation,
                norb=self.norb,
                nelec=nelec,
                copy=False,
            )
        return out