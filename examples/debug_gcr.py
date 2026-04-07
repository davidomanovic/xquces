import csv
from pathlib import Path

import numpy as np
import pyscf
import pyscf.gto
import pyscf.scf
import pyscf.mcscf
import pyscf.cc
import pyscf.ci
from scipy.sparse.linalg import LinearOperator
from ffsim.optimize import minimize_linear_method

from xquces.gcr import GCRSpinBalancedParameterization
from xquces.hamiltonians import MolecularHamiltonianLinearOperator
from xquces.orbitals import apply_orbital_rotation
from xquces.states import hartree_fock_state
from xquces.ucj.init import UCJBalancedDFSeed

start, stop, step = 1.2, 1.2, 0.1
bond_distance_range = np.linspace(start, stop, num=round((stop - start) / step) + 1)
molecule = "N2"
basis = "sto-6g"

OUT_CSV = Path(f"output/{molecule}_{basis}_gcr_debug.csv")
TRACE_CSV = Path(f"output/{molecule}_{basis}_gcr_debug_trace.csv")

pyscf.lib.num_threads(48)

RUN_OPTIMIZATION_IF_STABLE = True
FORWARD_REPEATS = 10
INVERSE_REPEATS = 10
RIGHT_REPEATS = 10
STABILITY_TOL = 1e-10


def phase_aligned_diff(psi, phi):
    overlap = np.vdot(phi, psi)
    if abs(overlap) < 1e-14:
        return np.linalg.norm(psi - phi)
    return np.linalg.norm(psi - overlap / abs(overlap) * phi)


def linear_operator_from_xquces_hamiltonian(ham):
    dim = hartree_fock_state(ham.norb, ham.nelec).size

    def matvec(v):
        v = np.asarray(v, dtype=np.complex128).reshape(-1)
        return ham.matvec(v) + ham.ecore * v

    def matmat(vs):
        vs = np.asarray(vs, dtype=np.complex128)
        return np.column_stack([matvec(vs[:, j]) for j in range(vs.shape[1])])

    return LinearOperator(
        shape=(dim, dim),
        matvec=matvec,
        matmat=matmat,
        dtype=np.complex128,
    )


def max_phase_diff(states):
    ref = states[0]
    return max(phase_aligned_diff(ref, psi) for psi in states[1:]) if len(states) > 1 else 0.0


def max_matrix_diff(mats):
    ref = mats[0]
    return max(np.linalg.norm(ref - m) for m in mats[1:]) if len(mats) > 1 else 0.0


def main():
    for R in bond_distance_range:
        mol = pyscf.gto.Mole()
        mol.build(
            atom=[("N", (-0.5 * R, 0, 0)), ("N", (0.5 * R, 0, 0))],
            basis=basis,
            symmetry="Dooh",
            verbose=0,
        )

        scf = pyscf.scf.RHF(mol)
        scf.kernel()

        active_space = list(range(2, mol.nao_nr()))
        norb = len(active_space)
        nelectron_cas = int(round(sum(scf.mo_occ[active_space])))
        n_alpha = (nelectron_cas + mol.spin) // 2
        n_beta = (nelectron_cas - mol.spin) // 2
        nelec = (n_alpha, n_beta)
        nocc = n_alpha

        cas = pyscf.mcscf.RCASCI(scf, ncas=norb, nelecas=nelec)
        mo_coeff = cas.sort_mo(active_space, base=0)

        cisd = pyscf.ci.RCISD(
            scf, frozen=[i for i in range(mol.nao_nr()) if i not in active_space]
        )
        cisd.kernel()

        ccsd = pyscf.cc.RCCSD(
            scf, frozen=[i for i in range(mol.nao_nr()) if i not in active_space]
        )
        ccsd.kernel()

        cas.fix_spin_(ss=0)
        cas.kernel(mo_coeff=mo_coeff)

        ham_xq = MolecularHamiltonianLinearOperator.from_scf(scf, active_space=active_space)
        H = linear_operator_from_xquces_hamiltonian(ham_xq)
        Phi0 = hartree_fock_state(norb, nelec)

        ucj_seed = UCJBalancedDFSeed(
            t2=ccsd.t2,
            t1=ccsd.t1,
            n_reps=1,
        ).build_ansatz()

        gcr_param = GCRSpinBalancedParameterization(
            norb=norb,
            nocc=nocc,
        )

        x0_seed = gcr_param.parameters_from_ucj_ansatz(ucj_seed)
        gcr_seed = gcr_param.ansatz_from_parameters(x0_seed)

        psi_ucj = ucj_seed.apply(Phi0, nelec=nelec, copy=True)
        psi_gcr = gcr_seed.apply(Phi0, nelec=nelec, copy=True)

        print("seed aligned diff =", phase_aligned_diff(psi_ucj, psi_gcr))
        print("seed energy ucj =", ham_xq.expectation(psi_ucj))
        print("seed energy gcr =", ham_xq.expectation(psi_gcr))
        
if __name__ == "__main__":
    main()