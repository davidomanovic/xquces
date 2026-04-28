from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from xquces.basis import flatten_state, occ_indicator_rows, reshape_state
from xquces.gcr.igcr4 import IGCR4Ansatz, IGCR4SpinRestrictedParameterization
from xquces.orbitals import apply_orbital_rotation


def _validate_distinct_indices(indices: tuple[int, ...], norb: int, name: str) -> tuple[int, ...]:
    out = tuple(int(x) for x in indices)
    if len(set(out)) != len(out):
        raise ValueError(f"{name} must contain distinct orbital indices")
    if any(x < 0 or x >= norb for x in out):
        raise ValueError(f"{name} index out of bounds")
    return out


def _row_lookup(occ: np.ndarray) -> dict[int, int]:
    lookup = {}
    for row, bits in enumerate(occ):
        key = 0
        for p, bit in enumerate(bits):
            if bit:
                key |= 1 << p
        lookup[key] = row
    return lookup


def _bit_key(bits: np.ndarray) -> int:
    key = 0
    for p, bit in enumerate(bits):
        if bit:
            key |= 1 << p
    return key


@dataclass(frozen=True)
class FourOpenShellSingletProjector:
    projector_orbitals: tuple[int, int, int, int] = (0, 1, 2, 3)
    spin_pair: tuple[int, int] = (0, 1)

    def validate(self, norb: int) -> "FourOpenShellSingletProjector":
        projector_orbitals = _validate_distinct_indices(self.projector_orbitals, norb, "projector_orbitals")
        spin_pair = _validate_distinct_indices(self.spin_pair, norb, "spin_pair")
        if len(projector_orbitals) != 4:
            raise ValueError("projector_orbitals must contain exactly four orbitals")
        if len(spin_pair) != 2:
            raise ValueError("spin_pair must contain exactly two orbitals")
        if not set(spin_pair).issubset(projector_orbitals):
            raise ValueError("spin_pair must be contained in projector_orbitals")
        return FourOpenShellSingletProjector(projector_orbitals, spin_pair)


def apply_four_open_shell_singlet_projector_phase(
    vec: np.ndarray,
    eta: float,
    norb: int,
    nelec: tuple[int, int],
    projector: FourOpenShellSingletProjector = FourOpenShellSingletProjector(),
    *,
    time: float = 1.0,
    copy: bool = True,
) -> np.ndarray:
    projector = projector.validate(norb)
    arr = np.array(vec, dtype=np.complex128, copy=copy)
    if abs(float(eta) * float(time)) <= 0.0:
        return arr

    state = reshape_state(arr, norb, nelec)
    occ_alpha = occ_indicator_rows(norb, nelec[0])
    occ_beta = occ_indicator_rows(norb, nelec[1])
    alpha_lookup = _row_lookup(occ_alpha)
    beta_lookup = _row_lookup(occ_beta)

    p, q = projector.spin_pair
    phase_update = 0.5 * (np.exp(1j * float(eta) * float(time)) - 1.0)

    for ia, alpha_bits in enumerate(occ_alpha):
        alpha_p = int(alpha_bits[p])
        alpha_q = int(alpha_bits[q])
        if alpha_p == alpha_q:
            continue
        for ib, beta_bits in enumerate(occ_beta):
            beta_p = int(beta_bits[p])
            beta_q = int(beta_bits[q])
            if beta_p == beta_q:
                continue
            if alpha_p != beta_q or alpha_q != beta_p:
                continue
            if alpha_p != 1 or beta_q != 1:
                continue
            if any(int(alpha_bits[r]) + int(beta_bits[r]) != 1 for r in projector.projector_orbitals):
                continue

            alpha_key = _bit_key(alpha_bits)
            beta_key = _bit_key(beta_bits)
            alpha_partner_key = alpha_key ^ (1 << p) ^ (1 << q)
            beta_partner_key = beta_key ^ (1 << p) ^ (1 << q)
            ja = alpha_lookup[alpha_partner_key]
            jb = beta_lookup[beta_partner_key]
            if (ia, ib) > (ja, jb):
                continue

            v1 = state[ia, ib]
            v2 = state[ja, jb]
            delta = phase_update * (v1 + v2)
            state[ia, ib] = v1 + delta
            state[ja, jb] = v2 + delta

    return flatten_state(state)


@dataclass(frozen=True)
class IGCR4SpinProjectorAnsatz:
    base: IGCR4Ansatz
    eta: float = 0.0
    projector: FourOpenShellSingletProjector = field(default_factory=FourOpenShellSingletProjector)

    @property
    def norb(self) -> int:
        return self.base.norb

    @property
    def nocc(self) -> int:
        return self.base.nocc

    @property
    def diagonal(self):
        return self.base.diagonal

    @property
    def left(self):
        return self.base.left

    @property
    def right(self):
        return self.base.right

    def apply(self, vec, nelec, copy=True):
        arr = np.array(vec, dtype=np.complex128, copy=copy)
        arr = apply_orbital_rotation(
            arr,
            self.right,
            norb=self.norb,
            nelec=nelec,
            copy=False,
        )
        arr = self.base.diagonal_apply(arr, nelec, copy=False) if hasattr(self.base, "diagonal_apply") else self._apply_base_diagonal(arr, nelec)
        arr = apply_four_open_shell_singlet_projector_phase(
            arr,
            self.eta,
            self.norb,
            nelec,
            self.projector,
            copy=False,
        )
        return apply_orbital_rotation(
            arr,
            self.left,
            norb=self.norb,
            nelec=nelec,
            copy=False,
        )

    def _apply_base_diagonal(self, vec, nelec):
        from xquces.gcr.igcr4 import apply_igcr4_spin_restricted_diagonal

        return apply_igcr4_spin_restricted_diagonal(
            vec,
            self.base.diagonal,
            self.norb,
            nelec,
            copy=False,
        )

    @classmethod
    def from_igcr4_ansatz(
        cls,
        ansatz: IGCR4Ansatz,
        *,
        eta: float = 0.0,
        projector: FourOpenShellSingletProjector = FourOpenShellSingletProjector(),
    ) -> "IGCR4SpinProjectorAnsatz":
        return cls(base=ansatz, eta=float(eta), projector=projector.validate(ansatz.norb))


@dataclass(frozen=True)
class IGCR4SpinProjectorParameterization:
    base: IGCR4SpinRestrictedParameterization
    projector: FourOpenShellSingletProjector = field(default_factory=FourOpenShellSingletProjector)

    def __post_init__(self):
        self.projector.validate(self.norb)

    @property
    def norb(self) -> int:
        return self.base.norb

    @property
    def nocc(self) -> int:
        return self.base.nocc

    @property
    def n_base_params(self) -> int:
        return self.base.n_params

    @property
    def n_spin_projector_params(self) -> int:
        return 1

    @property
    def n_params(self) -> int:
        return self.n_base_params + self.n_spin_projector_params

    def sector_sizes(self) -> dict[str, int]:
        out = dict(self.base.sector_sizes())
        out["spin_projector"] = self.n_spin_projector_params
        out["base_total"] = self.n_base_params
        out["total"] = self.n_params
        return out

    def _split_params(self, params: np.ndarray) -> tuple[np.ndarray, float]:
        params = np.asarray(params, dtype=np.float64)
        if params.shape != (self.n_params,):
            raise ValueError(f"Expected {(self.n_params,)}, got {params.shape}.")
        return params[: self.n_base_params], float(params[self.n_base_params])

    def ansatz_from_parameters(self, params: np.ndarray) -> IGCR4SpinProjectorAnsatz:
        base_params, eta = self._split_params(params)
        return IGCR4SpinProjectorAnsatz(
            base=self.base.ansatz_from_parameters(base_params),
            eta=eta,
            projector=self.projector.validate(self.norb),
        )

    def params_to_vec(
        self,
        reference_vec: np.ndarray,
        nelec: tuple[int, int],
    ) -> Callable[[np.ndarray], np.ndarray]:
        reference_vec = np.asarray(reference_vec, dtype=np.complex128)

        def fun(params: np.ndarray) -> np.ndarray:
            return self.ansatz_from_parameters(params).apply(reference_vec, nelec, copy=True)

        return fun

    def zeros(self) -> np.ndarray:
        return np.zeros(self.n_params, dtype=np.float64)

    @classmethod
    def from_igcr4_parameterization(
        cls,
        parameterization: IGCR4SpinRestrictedParameterization,
        *,
        projector: FourOpenShellSingletProjector = FourOpenShellSingletProjector(),
    ) -> "IGCR4SpinProjectorParameterization":
        return cls(base=parameterization, projector=projector)

    @classmethod
    def default(
        cls,
        norb: int,
        nocc: int,
        *,
        projector: FourOpenShellSingletProjector = FourOpenShellSingletProjector(),
        **kwargs,
    ) -> "IGCR4SpinProjectorParameterization":
        base = IGCR4SpinRestrictedParameterization(norb=norb, nocc=nocc, **kwargs)
        return cls(base=base, projector=projector)
