from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pyscf
import pyscf.ao2mo
import pyscf.fci
import pyscf.mcscf
import scipy.linalg

from xquces.basis import reshape_state, sector_shape


def spatial_integrals_from_rhf(mf) -> tuple[np.ndarray, np.ndarray]:
    mol = mf.mol
    mo = mf.mo_coeff
    hcore_ao = mf.get_hcore()
    h1 = mo.T @ hcore_ao @ mo
    eri_ao = mol.intor("int2e")
    eri_mo = pyscf.ao2mo.incore.full(eri_ao, mo)
    eri = pyscf.ao2mo.restore(1, eri_mo, mo.shape[1])
    return h1, eri


def active_space_data_from_rhf(
    mf,
    active_space,
) -> tuple[np.ndarray, np.ndarray, float, tuple[int, int]]:
    active_space = list(active_space)
    n_electrons = int(np.rint(np.sum(mf.mo_occ[active_space])))
    n_alpha = (n_electrons + mf.mol.spin) // 2
    n_beta = (n_electrons - mf.mol.spin) // 2
    nelec = (n_alpha, n_beta)

    cas = pyscf.mcscf.CASCI(mf, len(active_space), nelec)
    mo = cas.sort_mo(active_space, base=0)
    h1, ecore = cas.get_h1cas(mo)
    eri = cas.get_h2cas(mo)
    return h1, eri, float(ecore), nelec


def _sector_dimension(norb: int, nelec: tuple[int, int]) -> int:
    dim_a, dim_b = sector_shape(norb, nelec)
    return int(dim_a * dim_b)


def _dense_matrix_from_matvec(matvec, dim: int) -> np.ndarray:
    eye = np.eye(dim, dtype=np.complex128)
    out = np.empty((dim, dim), dtype=np.complex128)
    for j in range(dim):
        out[:, j] = np.asarray(matvec(eye[:, j]), dtype=np.complex128).reshape(-1)
    return out


def _validate_square_matrix(matrix: np.ndarray, name: str) -> None:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be a square matrix")


@dataclass(frozen=True)
class MolecularHamiltonianLinearOperator:
    h1: np.ndarray
    eri: np.ndarray
    ecore: float
    norb: int
    nelec: tuple[int, int]
    h2eff: np.ndarray

    @classmethod
    def from_scf(
        cls,
        mf,
        nelec: tuple[int, int] | None = None,
        active_space=None,
    ):
        if active_space is None:
            h1, eri = spatial_integrals_from_rhf(mf)
            norb = h1.shape[0]
            if nelec is None:
                n_electrons = int(np.rint(np.sum(mf.mo_occ)))
                n_alpha = (n_electrons + mf.mol.spin) // 2
                n_beta = (n_electrons - mf.mol.spin) // 2
                nelec = (n_alpha, n_beta)
            ecore = float(mf.mol.energy_nuc())
        else:
            if nelec is not None:
                raise ValueError("Pass either nelec or active_space, not both.")
            h1, eri, ecore, nelec = active_space_data_from_rhf(mf, active_space)
            norb = h1.shape[0]

        h2eff = pyscf.fci.direct_spin1.absorb_h1e(h1, eri, norb, nelec, 0.5)

        return cls(
            h1=h1,
            eri=eri,
            ecore=ecore,
            norb=norb,
            nelec=nelec,
            h2eff=h2eff,
        )

    def matvec(self, vec: np.ndarray) -> np.ndarray:
        fcivec = reshape_state(vec, self.norb, self.nelec)
        sigma = pyscf.fci.direct_spin1.contract_2e(
            self.h2eff,
            fcivec,
            self.norb,
            self.nelec,
        )
        return sigma.reshape(-1)

    def dense_electronic_matrix(self) -> np.ndarray:
        dim = _sector_dimension(self.norb, self.nelec)
        return _dense_matrix_from_matvec(self.matvec, dim)

    def expectation(self, vec: np.ndarray) -> float:
        arr = np.asarray(vec, dtype=np.complex128).reshape(-1)
        sigma = self.matvec(arr)
        return float(np.vdot(arr, sigma).real + self.ecore)


@dataclass(frozen=True)
class CanonicalTransformedHamiltonianLinearOperator:
    base: object
    unitary: np.ndarray
    h_matrix: np.ndarray
    ecore: float
    norb: int
    nelec: tuple[int, int]
    generator: np.ndarray | None = None

    @classmethod
    def from_generator(
        cls,
        base,
        generator: np.ndarray,
        *,
        validate_antihermitian: bool = True,
        symmetrize: bool = True,
        atol: float = 1e-10,
    ):
        generator = np.asarray(generator, dtype=np.complex128)
        _validate_square_matrix(generator, "generator")
        if validate_antihermitian and not np.allclose(
            generator.conj().T,
            -generator,
            atol=atol,
        ):
            raise ValueError("generator must be anti-Hermitian")
        unitary = np.asarray(scipy.linalg.expm(generator), dtype=np.complex128)
        return cls.from_unitary(
            base,
            unitary,
            generator=generator,
            validate_unitary=validate_antihermitian,
            symmetrize=symmetrize,
            atol=atol,
        )

    @classmethod
    def from_unitary(
        cls,
        base,
        unitary: np.ndarray,
        *,
        generator: np.ndarray | None = None,
        validate_unitary: bool = True,
        symmetrize: bool = True,
        atol: float = 1e-10,
    ):
        unitary = np.asarray(unitary, dtype=np.complex128)
        _validate_square_matrix(unitary, "unitary")
        dim = _sector_dimension(base.norb, base.nelec)
        if unitary.shape != (dim, dim):
            raise ValueError(f"unitary must have shape {(dim, dim)}, got {unitary.shape}")
        if validate_unitary and not np.allclose(
            unitary.conj().T @ unitary,
            np.eye(dim, dtype=np.complex128),
            atol=atol,
        ):
            raise ValueError("unitary must be unitary")
        if hasattr(base, "dense_electronic_matrix"):
            h_matrix = np.asarray(base.dense_electronic_matrix(), dtype=np.complex128)
        else:
            h_matrix = _dense_matrix_from_matvec(base.matvec, dim)
        transformed = unitary.conj().T @ h_matrix @ unitary
        if symmetrize:
            transformed = 0.5 * (transformed + transformed.conj().T)
        return cls(
            base=base,
            unitary=unitary,
            h_matrix=np.asarray(transformed, dtype=np.complex128),
            ecore=float(base.ecore),
            norb=int(base.norb),
            nelec=tuple(base.nelec),
            generator=None if generator is None else np.asarray(generator, dtype=np.complex128),
        )

    @classmethod
    def from_dense_matrix(
        cls,
        base,
        h_matrix: np.ndarray,
        *,
        unitary: np.ndarray | None = None,
        generator: np.ndarray | None = None,
        symmetrize: bool = True,
    ):
        dim = _sector_dimension(base.norb, base.nelec)
        h_matrix = np.asarray(h_matrix, dtype=np.complex128)
        if h_matrix.shape != (dim, dim):
            raise ValueError(f"h_matrix must have shape {(dim, dim)}, got {h_matrix.shape}")
        if symmetrize:
            h_matrix = 0.5 * (h_matrix + h_matrix.conj().T)
        if unitary is None:
            unitary = np.eye(dim, dtype=np.complex128)
        unitary = np.asarray(unitary, dtype=np.complex128)
        if unitary.shape != (dim, dim):
            raise ValueError(f"unitary must have shape {(dim, dim)}, got {unitary.shape}")
        return cls(
            base=base,
            unitary=unitary,
            h_matrix=h_matrix,
            ecore=float(base.ecore),
            norb=int(base.norb),
            nelec=tuple(base.nelec),
            generator=None if generator is None else np.asarray(generator, dtype=np.complex128),
        )

    def dense_electronic_matrix(self) -> np.ndarray:
        return np.array(self.h_matrix, copy=True)

    def physical_expectation(self, vec: np.ndarray) -> float:
        return float(self.base.expectation(vec))

    def matvec(self, vec: np.ndarray) -> np.ndarray:
        arr = np.asarray(vec, dtype=np.complex128).reshape(-1)
        if arr.size != self.h_matrix.shape[1]:
            raise ValueError("state size does not match transformed Hamiltonian dimension")
        return self.h_matrix @ arr

    def expectation(self, vec: np.ndarray) -> float:
        arr = np.asarray(vec, dtype=np.complex128).reshape(-1)
        sigma = self.matvec(arr)
        return float(np.vdot(arr, sigma).real + self.ecore)
