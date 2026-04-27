from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from xquces.gates import apply_gcr_spin_balanced, apply_gcr_spin_restricted
from xquces.ucj.model import SpinBalancedSpec, SpinRestrictedSpec, UCJAnsatz


@dataclass(frozen=True)
class GCRAnsatz:
    diagonal: SpinRestrictedSpec | SpinBalancedSpec
    left_orbital_rotation: np.ndarray
    right_orbital_rotation: np.ndarray

    def __post_init__(self):
        norb = self.diagonal.norb
        left = np.asarray(self.left_orbital_rotation, dtype=np.complex128)
        right = np.asarray(self.right_orbital_rotation, dtype=np.complex128)
        if left.shape != (norb, norb):
            raise ValueError("left_orbital_rotation has wrong shape")
        if right.shape != (norb, norb):
            raise ValueError("right_orbital_rotation has wrong shape")
        if not np.allclose(left.conj().T @ left, np.eye(norb), atol=1e-10):
            raise ValueError("left_orbital_rotation must be unitary")
        if not np.allclose(right.conj().T @ right, np.eye(norb), atol=1e-10):
            raise ValueError("right_orbital_rotation must be unitary")
        object.__setattr__(self, "left_orbital_rotation", left)
        object.__setattr__(self, "right_orbital_rotation", right)

    @property
    def norb(self) -> int:
        return self.diagonal.norb

    @property
    def is_spin_restricted(self) -> bool:
        return isinstance(self.diagonal, SpinRestrictedSpec)

    @property
    def is_spin_balanced(self) -> bool:
        return isinstance(self.diagonal, SpinBalancedSpec)

    def apply(
        self, vec: np.ndarray, nelec: tuple[int, int], copy: bool = True
    ) -> np.ndarray:
        d = self.diagonal
        if isinstance(d, SpinRestrictedSpec):
            return apply_gcr_spin_restricted(
                vec,
                double_params=d.double_params,
                pair_params=d.pair_params,
                norb=self.norb,
                nelec=nelec,
                left_orbital_rotation=self.left_orbital_rotation,
                right_orbital_rotation=self.right_orbital_rotation,
                copy=copy,
            )
        return apply_gcr_spin_balanced(
            vec,
            same_spin_params=d.same_spin_params,
            mixed_spin_params=d.mixed_spin_params,
            norb=self.norb,
            nelec=nelec,
            left_orbital_rotation=self.left_orbital_rotation,
            right_orbital_rotation=self.right_orbital_rotation,
            copy=copy,
        )


def gcr_from_ucj_ansatz(ansatz: UCJAnsatz) -> GCRAnsatz:
    if ansatz.n_layers != 1:
        raise ValueError("only a single-layer UCJ ansatz can be mapped exactly to GCR")
    layer = ansatz.layers[0]
    left = np.asarray(layer.orbital_rotation, dtype=np.complex128)
    if ansatz.final_orbital_rotation is not None:
        left = np.asarray(ansatz.final_orbital_rotation, dtype=np.complex128) @ left
    right = np.asarray(layer.orbital_rotation, dtype=np.complex128).conj().T
    return GCRAnsatz(
        diagonal=layer.diagonal,
        left_orbital_rotation=left,
        right_orbital_rotation=right,
    )
