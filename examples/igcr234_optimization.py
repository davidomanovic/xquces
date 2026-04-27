import csv
from pathlib import Path

import numpy as np
import pyscf
import pyscf.cc
import pyscf.ci
import pyscf.gto
import pyscf.mcscf
import pyscf.scf
from pyscf import lib
from scipy.sparse.linalg import LinearOperator

from xquces import (
    IGCR234SpinRestrictedParameterization,
    MolecularHamiltonianLinearOperator,
    UCJRestrictedProjectedDFSeed,
    hartree_fock_state,
    minimize_linear_method,
)

lib.num_threads(12)

start, stop, step = 1.0, 3.5, 0.1
bond_distance_range = np.linspace(start, stop, num=round((stop - start) / step) + 1)
molecule = "h4"
basis = "6-31g"

OUT_CSV = Path(f"output/{molecule}_{basis}_igcr234_sr.csv")
TRACE_CSV = Path(f"output/{molecule}_{basis}_igcr234_sr_trace.csv")


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
        "E_iGCR234_seed",
        "E_iGCR234_opt",
    ]
    trace_header = [
        "R",
        "iter",
        "energy",
        "max_abs_grad",
        "cond",
    ]

    print(",".join(header), flush=True)

    x_prev1 = None
    prev_ccsd_t1 = None
    prev_ccsd_t2 = None

    for R in bond_distance_range:
        mol = pyscf.gto.Mole()
        mol.build(
            atom=[
                ("H", (-0.5 * R, -0.5 * R, 0.0)),
                ("H", (0.5 * R, -0.5 * R, 0.0)),
                ("H", (-0.5 * R, 0.5 * R, 0.0)),
                ("H", (0.5 * R, 0.5 * R, 0.0)),
            ],
            basis=basis,
            symmetry="D2h",
            verbose=0,
        )

        scf = pyscf.scf.RHF(mol)
        scf.kernel()

        active_space = list(range(0, mol.nao_nr()))
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
        prev_ccsd_t1 = None if ccsd.t1 is None else np.array(ccsd.t1, copy=True)
        prev_ccsd_t2 = None if ccsd.t2 is None else np.array(ccsd.t2, copy=True)

        cas.fix_spin_(ss=0)
        cas.kernel(mo_coeff=mo_coeff)

        ham_xq = MolecularHamiltonianLinearOperator.from_scf(
            scf,
            active_space=active_space,
        )
        H = linear_operator_from_xquces_hamiltonian(ham_xq)
        Phi0 = hartree_fock_state(norb, nelec)

        ucj_seed = UCJRestrictedProjectedDFSeed(
            t2=ccsd.t2,
            t1=ccsd.t1,
            n_reps=3,
        ).build_ansatz()

        igcr234_param = IGCR234SpinRestrictedParameterization(
            norb=norb,
            nocc=n_alpha,
        )

        x0_seed = igcr234_param.parameters_from_ucj_ansatz(ucj_seed)

        if x_prev1 is not None and x_prev1.shape == x0_seed.shape:
            x0 = x_prev1
        else:
            x0 = x0_seed

        psi_seed = igcr234_param.ansatz_from_parameters(x0_seed).apply(
            Phi0,
            nelec=nelec,
            copy=True,
        )
        E_igcr234_seed = float(ham_xq.expectation(psi_seed))

        params_to_vec = igcr234_param.params_to_vec(Phi0, nelec)
        params_to_vec_jacobian = igcr234_param.params_to_vec_jacobian(Phi0, nelec)

        it_counter = {"k": 0}

        def callback(intermediate_result):
            it_counter["k"] += 1
            energy = float(intermediate_result.fun)
            if (
                hasattr(intermediate_result, "jac")
                and intermediate_result.jac is not None
            ):
                gmax = float(np.max(np.abs(intermediate_result.jac)))
            else:
                gmax = float("nan")
            if (
                hasattr(intermediate_result, "overlap_mat")
                and intermediate_result.overlap_mat is not None
            ):
                try:
                    cond = float(np.linalg.cond(intermediate_result.overlap_mat))
                except np.linalg.LinAlgError:
                    cond = float("inf")
            else:
                cond = float("nan")
            append_row_csv(
                TRACE_CSV,
                {
                    "R": f"{R:.6f}",
                    "iter": str(it_counter["k"]),
                    "energy": f"{energy:.12f}",
                    "max_abs_grad": f"{gmax:.12e}",
                    "cond": f"{cond:.12e}",
                },
                trace_header,
            )

        result = minimize_linear_method(
            params_to_vec,
            H,
            x0=x0,
            jac=params_to_vec_jacobian,
            ftol=1e-12,
            gtol=1e-6,
            maxiter=300,
            callback=callback,
        )

        x_prev1 = result.x.copy()
        E_igcr234_opt = float(result.fun)

        row = {
            "R": f"{R:.6f}",
            "E_FCI": f"{cas.e_tot:.12f}",
            "E_HF": f"{scf.e_tot:.12f}",
            "E_CCSD": f"{ccsd.e_tot:.12f}",
            "E_iGCR234_seed": f"{E_igcr234_seed:.12f}",
            "E_iGCR234_opt": f"{E_igcr234_opt:.12f}",
        }
        print(",".join([row[k] for k in header]), flush=True)
        append_row_csv(OUT_CSV, row, header)

    print(f"Wrote CSV: {OUT_CSV.resolve()}")
    print(f"Wrote trace CSV: {TRACE_CSV.resolve()}")


if __name__ == "__main__":
    main()
