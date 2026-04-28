from __future__ import annotations

import itertools
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


def _default_spin_pairs(projector_orbitals: tuple[int, int, int, int]) -> tuple[tuple[int, int], ...]:
    return tuple(itertools.combinations(projector_orbitals, 2))


def _bit_key(bits: np.ndarray, exclude: set[int] | None = None) -> int:
    exclude = set() if exclude is None else exclude
    key = 0
    for p, bit in enumerate(bits):
        if p not in exclude and bit:
            key |= 1 << p
    return key


def _local_mask(bits: np.ndarray, orbitals: tuple[int, int, int, int]) -> int:
    key = 0
    for i, p in enumerate(orbitals):
        if bits[p]:
            key |= 1 << i
    return key


def _mask_from_orbitals(selected: tuple[int, ...], orbitals: tuple[int, int, int, int]) -> int:
    positions = {p: i for i, p in enumerate(orbitals)}
    key = 0
    for p in selected:
        key |= 1 << positions[p]
    return key


def _pairing_from_spin_pair(
    projector_orbitals: tuple[int, int, int, int],
    spin_pair: tuple[int, int],
) -> tuple[tuple[int, int], tuple[int, int]]:
    pair = tuple(sorted(spin_pair))
    complement = tuple(sorted(p for p in projector_orbitals if p not in pair))
    return pair, complement


def _valence_bond_coefficients(
    projector_orbitals: tuple[int, int, int, int],
    spin_pair: tuple[int, int],
) -> dict[int, float]:
    (a, b), (c, d) = _pairing_from_spin_pair(projector_orbitals, spin_pair)
    return {
        _mask_from_orbitals((a, c), projector_orbitals): 0.5,
        _mask_from_orbitals((a, d), projector_orbitals): 0.5,
        _mask_from_orbitals((b, c), projector_orbitals): 0.5,
        _mask_from_orbitals((b, d), projector_orbitals): 0.5,
    }


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
        return FourOpenShellSingletProjector(projector_orbitals, tuple(sorted(spin_pair)))


@dataclass(frozen=True)
class FourOpenShellSingletProjectorSet:
    projector_orbitals: tuple[int, int, int, int] = (0, 1, 2, 3)
    spin_pairs: tuple[tuple[int, int], ...] | None = None

    def validate(self, norb: int) -> "FourOpenShellSingletProjectorSet":
        projector_orbitals = _validate_distinct_indices(self.projector_orbitals, norb, "projector_orbitals")
        if len(projector_orbitals) != 4:
            raise ValueError("projector_orbitals must contain exactly four orbitals")
        if self.spin_pairs is None:
            spin_pairs = _default_spin_pairs(projector_orbitals)
        else:
            spin_pairs = tuple(tuple(int(x) for x in pair) for pair in self.spin_pairs)
        out = []
        seen = set()
        for pair in spin_pairs:
            spin_pair = _validate_distinct_indices(pair, norb, "spin_pair")
            if len(spin_pair) != 2:
                raise ValueError("each spin_pair must contain exactly two orbitals")
            if not set(spin_pair).issubset(projector_orbitals):
                raise ValueError("each spin_pair must be contained in projector_orbitals")
            spin_pair = tuple(sorted(spin_pair))
            if spin_pair in seen:
                raise ValueError("spin_pairs must not contain duplicates")
            seen.add(spin_pair)
            out.append(spin_pair)
        if len(out) == 0:
            raise ValueError("spin_pairs must not be empty")
        return FourOpenShellSingletProjectorSet(projector_orbitals, tuple(out))


def _four_open_shell_groups(
    norb: int,
    nelec: tuple[int, int],
    projector_orbitals: tuple[int, int, int, int],
) -> dict[tuple[int, int], dict[int, tuple[int, int]]]:
    occ_alpha = occ_indicator_rows(norb, nelec[0])
    occ_beta = occ_indicator_rows(norb, nelec[1])
    orbital_set = set(projector_orbitals)
    full_mask = (1 << len(projector_orbitals)) - 1
    groups: dict[tuple[int, int], dict[int, tuple[int, int]]] = {}
    for ia, alpha_bits in enumerate(occ_alpha):
        alpha_mask = _local_mask(alpha_bits, projector_orbitals)
        if alpha_mask.bit_count() != 2:
            continue
        alpha_out_key = _bit_key(alpha_bits, orbital_set)
        for ib, beta_bits in enumerate(occ_beta):
            beta_mask = _local_mask(beta_bits, projector_orbitals)
            if beta_mask != (full_mask ^ alpha_mask):
                continue
            if any(int(alpha_bits[p]) + int(beta_bits[p]) != 1 for p in projector_orbitals):
                continue
            beta_out_key = _bit_key(beta_bits, orbital_set)
            groups.setdefault((alpha_out_key, beta_out_key), {})[alpha_mask] = (ia, ib)
    return groups


def _apply_valence_bond_phase(
    state: np.ndarray,
    groups: dict[tuple[int, int], dict[int, tuple[int, int]]],
    projector_orbitals: tuple[int, int, int, int],
    spin_pair: tuple[int, int],
    eta: float,
    time: float,
) -> None:
    phase_update = np.exp(1j * float(eta) * float(time)) - 1.0
    if abs(phase_update) <= 0.0:
        return
    coeffs = _valence_bond_coefficients(projector_orbitals, spin_pair)
    for group in groups.values():
        if any(mask not in group for mask in coeffs):
            continue
        amplitude = 0.0j
        for mask, coeff in coeffs.items():
            ia, ib = group[mask]
            amplitude += coeff * state[ia, ib]
        for mask, coeff in coeffs.items():
            ia, ib = group[mask]
            state[ia, ib] += phase_update * coeff * amplitude


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
    groups = _four_open_shell_groups(norb, nelec, projector.projector_orbitals)
    _apply_valence_bond_phase(
        state,
        groups,
        projector.projector_orbitals,
        projector.spin_pair,
        float(eta),
        float(time),
    )
    return flatten_state(state)


def apply_four_open_shell_singlet_projector_phases(
    vec: np.ndarray,
    etas: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    projector_set: FourOpenShellSingletProjectorSet = FourOpenShellSingletProjectorSet(),
    *,
    time: float = 1.0,
    copy: bool = True,
) -> np.ndarray:
    projector_set = projector_set.validate(norb)
    etas = np.asarray(etas, dtype=np.float64)
    if etas.shape != (len(projector_set.spin_pairs),):
        raise ValueError(f"Expected {(len(projector_set.spin_pairs),)}, got {etas.shape}.")
    arr = np.array(vec, dtype=np.complex128, copy=copy)
    if not np.any(np.abs(etas * float(time)) > 0.0):
        return arr
    state = reshape_state(arr, norb, nelec)
    groups = _four_open_shell_groups(norb, nelec, projector_set.projector_orbitals)
    for eta, spin_pair in zip(etas, projector_set.spin_pairs):
        _apply_valence_bond_phase(
            state,
            groups,
            projector_set.projector_orbitals,
            spin_pair,
            float(eta),
            float(time),
        )
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
        arr = apply_orbital_rotation(arr, self.right, norb=self.norb, nelec=nelec, copy=False)
        arr = self.base.diagonal_apply(arr, nelec, copy=False) if hasattr(self.base, "diagonal_apply") else self._apply_base_diagonal(arr, nelec)
        arr = apply_four_open_shell_singlet_projector_phase(
            arr,
            self.eta,
            self.norb,
            nelec,
            self.projector,
            copy=False,
        )
        return apply_orbital_rotation(arr, self.left, norb=self.norb, nelec=nelec, copy=False)

    def _apply_base_diagonal(self, vec, nelec):
        from xquces.gcr.igcr4 import apply_igcr4_spin_restricted_diagonal
        return apply_igcr4_spin_restricted_diagonal(vec, self.base.diagonal, self.norb, nelec, copy=False)

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
class IGCR4SpinProjectorSetAnsatz:
    base: IGCR4Ansatz
    etas: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float64))
    projector_set: FourOpenShellSingletProjectorSet = field(default_factory=FourOpenShellSingletProjectorSet)

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
        projector_set = self.projector_set.validate(self.norb)
        etas = np.asarray(self.etas, dtype=np.float64)
        if etas.shape != (len(projector_set.spin_pairs),):
            raise ValueError(f"Expected {(len(projector_set.spin_pairs),)}, got {etas.shape}.")
        arr = np.array(vec, dtype=np.complex128, copy=copy)
        arr = apply_orbital_rotation(arr, self.right, norb=self.norb, nelec=nelec, copy=False)
        arr = self.base.diagonal_apply(arr, nelec, copy=False) if hasattr(self.base, "diagonal_apply") else self._apply_base_diagonal(arr, nelec)
        arr = apply_four_open_shell_singlet_projector_phases(
            arr,
            etas,
            self.norb,
            nelec,
            projector_set,
            copy=False,
        )
        return apply_orbital_rotation(arr, self.left, norb=self.norb, nelec=nelec, copy=False)

    def _apply_base_diagonal(self, vec, nelec):
        from xquces.gcr.igcr4 import apply_igcr4_spin_restricted_diagonal
        return apply_igcr4_spin_restricted_diagonal(vec, self.base.diagonal, self.norb, nelec, copy=False)

    @classmethod
    def from_igcr4_ansatz(
        cls,
        ansatz: IGCR4Ansatz,
        *,
        etas: np.ndarray | None = None,
        projector_set: FourOpenShellSingletProjectorSet = FourOpenShellSingletProjectorSet(),
    ) -> "IGCR4SpinProjectorSetAnsatz":
        projector_set = projector_set.validate(ansatz.norb)
        if etas is None:
            etas = np.zeros(len(projector_set.spin_pairs), dtype=np.float64)
        etas = np.asarray(etas, dtype=np.float64)
        if etas.shape != (len(projector_set.spin_pairs),):
            raise ValueError(f"Expected {(len(projector_set.spin_pairs),)}, got {etas.shape}.")
        return cls(base=ansatz, etas=etas, projector_set=projector_set)


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

    def params_to_vec(self, reference_vec: np.ndarray, nelec: tuple[int, int]) -> Callable[[np.ndarray], np.ndarray]:
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


@dataclass(frozen=True)
class IGCR4SpinProjectorSetParameterization:
    base: IGCR4SpinRestrictedParameterization
    projector_set: FourOpenShellSingletProjectorSet = field(default_factory=FourOpenShellSingletProjectorSet)

    def __post_init__(self):
        self.projector_set.validate(self.norb)

    @property
    def norb(self) -> int:
        return self.base.norb

    @property
    def nocc(self) -> int:
        return self.base.nocc

    @property
    def spin_pairs(self) -> tuple[tuple[int, int], ...]:
        return self.projector_set.validate(self.norb).spin_pairs

    @property
    def n_base_params(self) -> int:
        return self.base.n_params

    @property
    def n_spin_projector_params(self) -> int:
        return len(self.spin_pairs)

    @property
    def n_params(self) -> int:
        return self.n_base_params + self.n_spin_projector_params

    def sector_sizes(self) -> dict[str, int]:
        out = dict(self.base.sector_sizes())
        out["spin_projector"] = self.n_spin_projector_params
        out["base_total"] = self.n_base_params
        out["total"] = self.n_params
        return out

    def _split_params(self, params: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        params = np.asarray(params, dtype=np.float64)
        if params.shape != (self.n_params,):
            raise ValueError(f"Expected {(self.n_params,)}, got {params.shape}.")
        return params[: self.n_base_params], params[self.n_base_params :].copy()

    def ansatz_from_parameters(self, params: np.ndarray) -> IGCR4SpinProjectorSetAnsatz:
        base_params, etas = self._split_params(params)
        return IGCR4SpinProjectorSetAnsatz(
            base=self.base.ansatz_from_parameters(base_params),
            etas=etas,
            projector_set=self.projector_set.validate(self.norb),
        )

    def params_to_vec(self, reference_vec: np.ndarray, nelec: tuple[int, int]) -> Callable[[np.ndarray], np.ndarray]:
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
        projector_set: FourOpenShellSingletProjectorSet = FourOpenShellSingletProjectorSet(),
    ) -> "IGCR4SpinProjectorSetParameterization":
        return cls(base=parameterization, projector_set=projector_set)

    @classmethod
    def default(
        cls,
        norb: int,
        nocc: int,
        *,
        projector_set: FourOpenShellSingletProjectorSet = FourOpenShellSingletProjectorSet(),
        **kwargs,
    ) -> "IGCR4SpinProjectorSetParameterization":
        base = IGCR4SpinRestrictedParameterization(norb=norb, nocc=nocc, **kwargs)
        return cls(base=base, projector_set=projector_set)
