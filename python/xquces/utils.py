from __future__ import annotations

import numpy as np
import pyscf.ao2mo
import pyscf.cc
import pyscf.gto
import pyscf.mcscf
import pyscf.scf
from pyscf.fci.spin_op import contract_ss


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


def run_rhf(
    mol,
    *,
    dm0=None,
    init_guess: str | None = None,
    conv_tol: float = 1e-12,
    max_cycle: int = 200,
):
    mf = pyscf.scf.RHF(mol)
    if init_guess is not None:
        mf.init_guess = init_guess
    mf.conv_tol = conv_tol
    mf.max_cycle = max_cycle
    mf.kernel(dm0=dm0)
    if not mf.converged:
        raise RuntimeError("RHF did not converge")
    return mf


def run_lowest_rhf(
    mol,
    *,
    dm0=None,
    init_guesses=("atom", "minao", "hcore", "1e"),
    random_trials: int = 4,
    random_scale: float = 0.02,
    random_seed: int = 9173,
    conv_tol: float = 1e-12,
    max_cycle: int = 200,
):
    candidates = []

    def try_run(*, dm0_candidate=None, init_guess=None):
        try:
            candidates.append(
                run_rhf(
                    mol,
                    dm0=dm0_candidate,
                    init_guess=init_guess,
                    conv_tol=conv_tol,
                    max_cycle=max_cycle,
                )
            )
        except RuntimeError:
            return

    if dm0 is not None:
        try_run(dm0_candidate=dm0)

    for guess in init_guesses:
        try_run(init_guess=guess)

    if random_trials:
        mf = pyscf.scf.RHF(mol)
        base_dm = mf.get_init_guess(key="minao")
        rng = np.random.default_rng(random_seed)
        for _ in range(int(random_trials)):
            noise = rng.normal(size=base_dm.shape)
            noise = random_scale * (noise + noise.T)
            try_run(dm0_candidate=base_dm + noise)

    if not candidates:
        raise RuntimeError("No RHF candidate converged")
    return min(candidates, key=lambda mf: float(mf.e_tot))


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

def apply_spin_square(fcivec: np.ndarray, norb: int, nelec: tuple[int, int]) -> np.ndarray:
    """Apply the spin-squared operator in the fixed ``(n_alpha, n_beta)`` sector."""
    if np.iscomplexobj(fcivec):
        ci1 = contract_ss(fcivec.real, norb, nelec).astype(complex)
        ci1 += 1j * contract_ss(fcivec.imag, norb, nelec)
    else:
        ci1 = contract_ss(fcivec, norb, nelec)
    return np.asarray(ci1, dtype=np.complex128).reshape(np.asarray(fcivec).shape)


def spin_square(fcivec: np.ndarray, norb: int, nelec: tuple[int, int]):
    """Expectation value of spin squared operator on a state vector."""
    ci1 = apply_spin_square(fcivec, norb, nelec)
    return float(np.real(np.vdot(np.asarray(fcivec).reshape(-1), ci1.reshape(-1))))
