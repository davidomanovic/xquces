import numpy as np
import pyscf
import pyscf.gto
import pyscf.scf
import pyscf.mcscf
import pyscf.cc
import pyscf.ci
import ffsim
from scipy.sparse.linalg import LinearOperator
from ffsim.optimize import minimize_linear_method

from xquces.hamiltonians import MolecularHamiltonianLinearOperator
from xquces.states import hartree_fock_state


n_f = 0
R = 1.0
basis = "cc-pvdz"
threads = 48


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


pyscf.lib.num_threads(threads)

mol = pyscf.gto.Mole()
mol.build(
    atom=[("H", (-0.5 * R, 0, 0)), ("H", (0.5 * R, 0, 0))],
    basis=basis,
    symmetry="Dooh",
    verbose=0,
)

scf = pyscf.scf.RHF(mol)
scf.kernel()

active_space = list(range(n_f, mol.nao_nr()))
norb = len(active_space)
nelectron_cas = int(round(sum(scf.mo_occ[active_space])))
n_alpha = (nelectron_cas + mol.spin) // 2
n_beta = (nelectron_cas - mol.spin) // 2
nelec = (n_alpha, n_beta)

cas = pyscf.mcscf.RCASCI(scf, ncas=norb, nelecas=nelec)
mo_coeff = cas.sort_mo(active_space, base=0)

cisd = pyscf.ci.RCISD(
    scf, frozen=[i for i in range(mol.nao_nr()) if i not in active_space]
)
cisd.kernel()

ccsd = pyscf.cc.RCCSD(
    scf, frozen=[i for i in range(mol.nao_nr()) if i not in active_space]
)
ccsd.kernel(
    t1=ccsd.t1,
    t2=ccsd.t2,
)

cas.fix_spin_(ss=0)
cas.kernel(mo_coeff=mo_coeff)

ham_xq = MolecularHamiltonianLinearOperator.from_scf(scf, active_space=active_space)
H = linear_operator_from_xquces_hamiltonian(ham_xq)
Phi0 = ffsim.hartree_fock_state(norb, nelec)

ucj_seed = ffsim.UCJOpSpinBalanced.from_t_amplitudes(
    np.asarray(ccsd.t2, dtype=np.float64),
    t1=np.asarray(ccsd.t1, dtype=np.complex128),
    n_reps=1,
)
x0_seed = ucj_seed.to_parameters()

print("Number of parameters:", len(x0_seed), flush=True)

psi_seed = ffsim.apply_unitary(Phi0, ucj_seed, norb=norb, nelec=nelec, copy=True)
E_UCJ_seed = ham_xq.expectation(psi_seed)


def params_to_vec(x):
    ucj = ffsim.UCJOpSpinBalanced.from_parameters(
        np.asarray(x, dtype=np.float64),
        norb=norb,
        n_reps=1,
        with_final_orbital_rotation=True,
    )
    return ffsim.apply_unitary(Phi0, ucj, norb=norb, nelec=nelec, copy=True)


it_counter = {"k": 0}


def callback(intermediate_result):
    it_counter["k"] += 1
    energy = float(intermediate_result.fun)

    jac = getattr(intermediate_result, "jac", None)
    if jac is None:
        gmax = np.nan
    else:
        gmax = float(np.max(np.abs(jac)))

    cond = float(getattr(intermediate_result, "cond_S", np.nan))
    regularization = float(getattr(intermediate_result, "regularization", np.nan))
    variation = float(getattr(intermediate_result, "variation", np.nan))

    print(
        f"Iter {it_counter['k']}: "
        f"E = {energy:.12f}, "
        f"gmax = {gmax:.2e}, "
        f"cond_S = {cond:.2e}, "
        f"reg = {regularization:.2e}, "
        f"var = {variation:.2e}",
        flush=True,
    )


result = minimize_linear_method(
    params_to_vec,
    H,
    x0=x0_seed,
    maxiter=100,
    gtol=1e-8,
    ftol=1e-12,
    callback=callback,
)

E_UCJ_opt = float(result.fun)

E_HF = scf.e_tot
E_FCI = cas.e_tot

print("Final results:")
print(f"E(HF) = {E_HF:.12f}")
print(f"E(FCI) = {E_FCI:.12f}")
print(f"E(UCJ seed) = {E_UCJ_seed:.12f}")
print(f"E(UCJ ffsim) = {E_UCJ_opt:.12f}")
print(f"Correlation energy captured: {(E_UCJ_opt - E_HF) / (E_FCI - E_HF) * 100:.2f}%")
