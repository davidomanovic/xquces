from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pyscf
import pyscf.ao2mo
import pyscf.fci
import pyscf.mcscf

from xquces.basis import reshape_state


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

    def expectation(self, vec: np.ndarray) -> float:
        arr = np.asarray(vec, dtype=np.complex128).reshape(-1)
        sigma = self.matvec(arr)
        return float(np.vdot(arr, sigma).real + self.ecore)
