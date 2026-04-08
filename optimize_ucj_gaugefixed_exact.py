import csv
from pathlib import Path

import numpy as np
import pyscf
import pyscf.cc
import pyscf.ci
import pyscf.gto
import pyscf.lib
import pyscf.mcscf
import pyscf.scf
from ffsim.optimize import minimize_linear_method
from scipy.sparse.linalg import LinearOperator
from threadpoolctl import threadpool_limits

from xquces.hamiltonians import MolecularHamiltonianLinearOperator
from xquces.states import hartree_fock_state
from xquces.ucj.gaugefixed_exact import GaugeFixedUCJBalancedDFSeedExact

start, stop, step = 1.2, 1.2, 0.1
bond_distance_range = np.linspace(start, stop, num=round((stop - start) / step) + 1)
molecule = "N2"
basis = "sto-6g"

OUT_CSV = Path(f"output/{molecule}_{basis}_ucj_gaugefixed_exact.csv")
TRACE_CSV = Path(f"output/{molecule}_{basis}_ucj_gaugefixed_exact_trace.csv")

pyscf.lib.num_threads(48)


def append_row_csv(path, row_dict, header):
    new_file = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if new_file:
            w.writeheader()
        w.writerow(row_dict)


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


def main():
    if OUT_CSV.exists():
        OUT_CSV.unlink()
    if TRACE_CSV.exists():
        TRACE_CSV.unlink()

    header = [
        "R",
        "E_FCI",
        "E_HF",
        "E_CCSD",
        "E_UCJ_seed_fresh",
        "E_UCJ_start",
        "E_UCJ_opt",
    ]
    trace_header = [
        "R",
        "iter",
        "energy",
        "max_abs_grad",
        "cond",
        "reg",
    ]

    print(",".join(header), flush=True)

    x_prev = None
    prev_ccsd_t1 = None
    prev_ccsd_t2 = None

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
            t1=prev_ccsd_t1,
            t2=prev_ccsd_t2,
        )
        prev_ccsd_t1 = np.array(ccsd.t1, copy=True)
        prev_ccsd_t2 = np.array(ccsd.t2, copy=True)

        cas.fix_spin_(ss=0)
        cas.kernel(mo_coeff=mo_coeff)

        ham_xq = MolecularHamiltonianLinearOperator.from_scf(scf, active_space=active_space)
        H = linear_operator_from_xquces_hamiltonian(ham_xq)
        Phi0 = hartree_fock_state(norb, nelec)

        ucj_seed, ucj_param, x0_seed = GaugeFixedUCJBalancedDFSeedExact(
            t2=ccsd.t2,
            t1=ccsd.t1,
            n_reps=1,
        ).build_parameters()

        print("params:", len(x0_seed), flush=True)
        if x_prev is not None and x_prev.shape == x0_seed.shape:
            x0 = x_prev
        else:
            x0 = x0_seed

        psi_seed_fresh = ucj_param.ansatz_from_parameters(x0_seed).apply(Phi0, nelec=nelec, copy=True)
        E_UCJ_seed_fresh = ham_xq.expectation(psi_seed_fresh)

        psi_start = ucj_param.ansatz_from_parameters(x0).apply(Phi0, nelec=nelec, copy=True)
        E_UCJ_start = ham_xq.expectation(psi_start)

        x0 = ucj_param.parameters_from_ansatz(ucj_seed)
        ucj0 = ucj_param.ansatz_from_parameters(x0)

        psi_seed = ucj0.apply(Phi0, nelec=nelec, copy=True)
        print("E_UCJ_seed_fresh =", ham_xq.expectation(psi_seed), flush=True)

        x1 = ucj_param.parameters_from_ansatz(ucj0)
        ucj1 = ucj_param.ansatz_from_parameters(x1)
        psi_rt = ucj1.apply(Phi0, nelec=nelec, copy=True)

        overlap = np.vdot(psi_seed, psi_rt)
        psi_rt *= overlap.conjugate() / abs(overlap)
        print("roundtrip - direct =", np.linalg.norm(psi_rt - psi_seed), flush=True)
        print("params:", ucj_param.n_params, flush=True)

        def params_to_vec(x):
            return ucj_param.ansatz_from_parameters(x).apply(Phi0, nelec=nelec, copy=True)

        it_counter = {"k": 0}

        def callback(intermediate_result):
            it_counter["k"] += 1
            energy = float(intermediate_result.fun)

            if hasattr(intermediate_result, "jac") and intermediate_result.jac is not None:
                gmax = float(np.max(np.abs(intermediate_result.jac)))
            else:
                gmax = float("nan")

            if hasattr(intermediate_result, "overlap_mat") and intermediate_result.overlap_mat is not None:
                try:
                    cond = float(np.linalg.cond(intermediate_result.overlap_mat))
                except np.linalg.LinAlgError:
                    cond = float("inf")
            else:
                cond = float("nan")
            reg = getattr(intermediate_result, "regularization", np.nan)
            append_row_csv(
                TRACE_CSV,
                {
                    "R": f"{R:.6f}",
                    "iter": str(it_counter['k']),
                    "energy": f"{energy:.12f}",
                    "max_abs_grad": f"{gmax:.12e}",
                    "cond": f"{cond:.12e}",
                    "reg": f"{reg:.12e}",
                },
                trace_header,
            )

        result = minimize_linear_method(
            params_to_vec,
            H,
            x0=x0,
            maxiter=200,
            gtol=1e-6,
            ftol=1e-12,
            callback=callback,
        )

        x_prev = result.x.copy()
        E_UCJ_opt = float(result.fun)

        row = {
            "R": f"{R:.6f}",
            "E_FCI": f"{cas.e_tot:.12f}",
            "E_HF": f"{scf.e_tot:.12f}",
            "E_CCSD": f"{ccsd.e_tot:.12f}",
            "E_UCJ_seed_fresh": f"{E_UCJ_seed_fresh:.12f}",
            "E_UCJ_start": f"{E_UCJ_start:.12f}",
            "E_UCJ_opt": f"{E_UCJ_opt:.12f}",
        }
        print(",".join([row[k] for k in header]), flush=True)
        append_row_csv(OUT_CSV, row, header)

    print(f"Wrote CSV: {OUT_CSV.resolve()}")
    print(f"Wrote trace CSV: {TRACE_CSV.resolve()}")


if __name__ == "__main__":
    pyscf.lib.num_threads(1)
    with threadpool_limits(limits=1):
        main()
