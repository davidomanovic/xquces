from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from xquces.gates import apply_ucj_spin_balanced, apply_ucj_spin_restricted
from xquces.orbitals import apply_orbital_rotation, canonicalize_unitary


@dataclass(frozen=True)
class SpinRestrictedSpec:
    double_params: np.ndarray
    pair_params: np.ndarray

    def __post_init__(self):
        d = np.asarray(self.double_params, dtype=np.float64)
        p = np.asarray(self.pair_params, dtype=np.float64)
        norb = d.shape[0]
        if d.shape != (norb,):
            raise ValueError("double_params must have shape (norb,)")
        if p.shape != (norb, norb):
            raise ValueError("pair_params must have shape (norb, norb)")
        if not np.allclose(p, p.T):
            raise ValueError("pair_params must be symmetric")
        p = np.array(p, copy=True)
        np.fill_diagonal(p, 0.0)
        object.__setattr__(self, "double_params", d)
        object.__setattr__(self, "pair_params", p)

    @property
    def norb(self) -> int:
        return self.double_params.shape[0]


@dataclass(frozen=True)
class SpinBalancedSpec:
    same_spin_params: np.ndarray
    mixed_spin_params: np.ndarray

    def __post_init__(self):
        j0 = np.asarray(self.same_spin_params, dtype=np.float64)
        j1 = np.asarray(self.mixed_spin_params, dtype=np.float64)
        norb = j0.shape[0]
        if j0.shape != (norb, norb):
            raise ValueError("same_spin_params must have shape (norb, norb)")
        if j1.shape != (norb, norb):
            raise ValueError("mixed_spin_params must have shape (norb, norb)")
        if not np.allclose(j0, j0.T):
            raise ValueError("same_spin_params must be symmetric")
        if not np.allclose(j1, j1.T):
            raise ValueError("mixed_spin_params must be symmetric")
        object.__setattr__(self, "same_spin_params", j0)
        object.__setattr__(self, "mixed_spin_params", j1)

    @property
    def norb(self) -> int:
        return self.same_spin_params.shape[0]


@dataclass(frozen=True)
class UCJLayer:
    diagonal: SpinRestrictedSpec | SpinBalancedSpec
    orbital_rotation: np.ndarray

    def __post_init__(self):
        u = np.asarray(self.orbital_rotation, dtype=np.complex128)
        norb = self.diagonal.norb
        if u.shape != (norb, norb):
            raise ValueError("orbital_rotation has wrong shape")
        if not np.allclose(u.conj().T @ u, np.eye(norb), atol=1e-10):
            raise ValueError("orbital_rotation must be unitary")
        object.__setattr__(self, "orbital_rotation", u)

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
        layer_type = type(self.layers[0].diagonal)
        for layer in self.layers:
            if layer.norb != norb:
                raise ValueError("all layers must have the same norb")
            if type(layer.diagonal) is not layer_type:
                raise ValueError("all layers must use the same diagonal spec type")
        if self.final_orbital_rotation is not None:
            u = np.asarray(self.final_orbital_rotation, dtype=np.complex128)
            if u.shape != (norb, norb):
                raise ValueError("final_orbital_rotation has wrong shape")
            if not np.allclose(u.conj().T @ u, np.eye(norb), atol=1e-10):
                raise ValueError("final_orbital_rotation must be unitary")
            object.__setattr__(self, "final_orbital_rotation", u)

    @property
    def norb(self) -> int:
        return self.layers[0].norb

    @property
    def n_layers(self) -> int:
        return len(self.layers)

    @property
    def is_spin_restricted(self) -> bool:
        return isinstance(self.layers[0].diagonal, SpinRestrictedSpec)

    @property
    def is_spin_balanced(self) -> bool:
        return isinstance(self.layers[0].diagonal, SpinBalancedSpec)

    def apply(self, vec: np.ndarray, nelec: tuple[int, int], copy: bool = True) -> np.ndarray:
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