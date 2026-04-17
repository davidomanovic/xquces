from __future__ import annotations

import numpy as np
import pyscf.ao2mo
import pyscf.cc
import pyscf.gto
import pyscf.mcscf
import pyscf.scf


def build_n2_mol(R: float, basis: str, *, symmetry: bool | str = "Dooh"):
    mol = pyscf.gto.Mole()
    mol.build(
        atom=[("N", (-0.5 * R, 0.0, 0.0)), ("N", (0.5 * R, 0.0, 0.0))],
        basis=basis,
        symmetry=symmetry,
        verbose=0,
    )
    return mol


def build_h4_square_mol(R: float, basis: str, *, symmetry: bool | str = False):
    mol = pyscf.gto.Mole()
    mol.build(
        atom=[
            ("H", (-0.5 * R, -0.5 * R, 0.0)),
            ("H", (0.5 * R, -0.5 * R, 0.0)),
            ("H", (-0.5 * R, 0.5 * R, 0.0)),
            ("H", (0.5 * R, 0.5 * R, 0.0)),
        ],
        basis=basis,
        symmetry=symmetry,
        verbose=0,
    )
    return mol


def run_rhf(mol, *, dm0=None, conv_tol: float = 1e-12, max_cycle: int = 200):
    mf = pyscf.scf.RHF(mol)
    mf.conv_tol = conv_tol
    mf.max_cycle = max_cycle
    mf.kernel(dm0=dm0)
    if not mf.converged:
        raise RuntimeError("RHF did not converge")
    return mf


def run_rccsd(
    mf,
    frozen=None,
    *,
    t1=None,
    t2=None,
    conv_tol: float = 1e-12,
    conv_tol_normt: float = 1e-10,
    max_cycle: int = 1000,
):
    cc = pyscf.cc.RCCSD(mf, frozen=frozen)
    cc.conv_tol = conv_tol
    cc.conv_tol_normt = conv_tol_normt
    cc.max_cycle = max_cycle
    cc.kernel(t1=t1, t2=t2)
    if cc.t1 is None or cc.t2 is None:
        raise RuntimeError("RCCSD failed to produce amplitudes")
    return cc


def active_space_from_frozen_core(mf, n_frozen: int) -> list[int]:
    return list(range(int(n_frozen), mf.mol.nao_nr()))


def active_nelec_from_mo_occ(mf, active_space) -> tuple[int, int]:
    n_electrons = int(round(float(np.sum(mf.mo_occ[list(active_space)]))))
    n_alpha = (n_electrons + mf.mol.spin) // 2
    n_beta = (n_electrons - mf.mol.spin) // 2
    return int(n_alpha), int(n_beta)


def frozen_orbitals_from_active_space(mf, active_space) -> list[int]:
    active = set(int(i) for i in active_space)
    return [i for i in range(mf.mol.nao_nr()) if i not in active]


def run_casscf(
    mf,
    *,
    ncas: int,
    nelecas: tuple[int, int],
    active_space=None,
    mo_coeff=None,
    conv_tol: float = 1e-10,
    max_cycle_macro: int = 100,
):
    mc = pyscf.mcscf.CASSCF(mf, ncas=ncas, nelecas=nelecas)
    mc.conv_tol = conv_tol
    mc.max_cycle_macro = max_cycle_macro
    if mo_coeff is None and active_space is not None:
        mo_coeff = mc.sort_mo(list(active_space), base=0)
    mc.kernel(mo_coeff=mo_coeff)
    if not mc.converged:
        raise RuntimeError("CASSCF did not converge")
    return mc


def active_mo_coeff_from_casscf(mc) -> np.ndarray:
    return np.asarray(
        mc.mo_coeff[:, mc.ncore : mc.ncore + mc.ncas],
        dtype=np.complex128,
    )


def orbital_overlap_between(
    prev_mol, prev_mo: np.ndarray, mol, mo: np.ndarray
) -> np.ndarray:
    ao_overlap = pyscf.gto.intor_cross("int1e_ovlp", prev_mol, mol)
    return np.asarray(prev_mo, dtype=np.complex128).conj().T @ ao_overlap @ mo


def active_hamiltonian_from_casscf(mc):
    import ffsim

    h1, ecore = mc.get_h1eff()
    h2 = mc.get_h2eff()
    eri = pyscf.ao2mo.restore(1, h2, mc.ncas)
    return ffsim.MolecularHamiltonian(
        np.asarray(h1, dtype=np.float64),
        np.asarray(eri, dtype=np.float64),
        float(ecore),
    )
