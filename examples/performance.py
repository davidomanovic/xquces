import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pyscf
import pyscf.gto
import pyscf.scf
import pyscf.mcscf
import pyscf.cc
import pyscf.fci
import pyscf.data.elements
from pyscf import lib
lib.num_threads(48)

import ffsim
from ffsim.optimize import minimize_linear_method
from scipy.sparse.linalg import LinearOperator

from xquces.hamiltonians import MolecularHamiltonianLinearOperator
from xquces.states import hartree_fock_state
from xquces.ucj.init import UCJBalancedDFSeed
from xquces.ucj.parameterization import GaugeFixedUCJSpinBalancedParameterization


molecule = "N2"
basis = "sto-6g"
R = 1.2
n_reps = 1
tol = 1e-8
optimize_df = False
maxiter = 20

OUT_DIR = Path("output")
TRACE_CSV = OUT_DIR / f"{molecule}_{basis}_xquces_ffsim_trace.csv"
SUMMARY_CSV = OUT_DIR / f"{molecule}_{basis}_xquces_ffsim_summary.csv"
FIG_PATH = OUT_DIR / f"{molecule}_{basis}_xquces_ffsim_convergence.png"


def append_row_csv(path, row_dict, header):
    new_file = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if new_file:
            w.writeheader()
        w.writerow(row_dict)


def linear_operator_from_xquces_hamiltonian(ham):
    dim = ham.matvec(hartree_fock_state(ham.norb, ham.nelec)).size

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


def default_active_space(mf, n_frozen=None):
    if n_frozen is None:
        n_frozen = pyscf.data.elements.chemcore(mf.mol)
    return list(range(n_frozen, mf.mo_coeff.shape[1]))


def frozen_from_active_space(mf, active_space):
    return sorted(set(range(mf.mo_coeff.shape[1])) - set(active_space))


def main():
    if TRACE_CSV.exists():
        TRACE_CSV.unlink()
    if SUMMARY_CSV.exists():
        SUMMARY_CSV.unlink()

    trace_header = [
        "ansatz",
        "R",
        "iter",
        "energy",
        "error_fci",
        "max_abs_grad",
        "reg",
        "cond",
    ]
    summary_header = [
        "R",
        "norb",
        "nelec",
        "E_FCI",
        "E_HF",
        "E_CCSD",
        "E_xquces_seed",
        "E_xquces_opt",
        "E_ffsim_seed",
        "E_ffsim_opt",
        "n_params_xquces",
        "n_params_ffsim",
    ]

    mol = pyscf.gto.Mole()
    mol.build(
        atom=[("N", (-0.5 * R, 0, 0)), ("N", (0.5 * R, 0, 0))],
        basis=basis,
        symmetry=False,
        unit="Angstrom",
        verbose=0,
    )

    mf = pyscf.scf.RHF(mol)
    mf.kernel()

    active_space = default_active_space(mf)
    frozen = frozen_from_active_space(mf, active_space)

    cc = pyscf.cc.CCSD(mf, frozen=frozen)
    cc.kernel()

    ham_xq = MolecularHamiltonianLinearOperator.from_scf(
        mf,
        active_space=active_space,
    )
    ham_lm_xq = linear_operator_from_xquces_hamiltonian(ham_xq)

    norb = ham_xq.norb
    nelec = ham_xq.nelec
    nocc = nelec[0]

    e_fci, _ = pyscf.fci.direct_spin1.kernel(
        ham_xq.h1,
        ham_xq.eri,
        norb,
        nelec,
        ecore=ham_xq.ecore,
    )

    e_hf = mf.e_tot

    reference_vec_xq = hartree_fock_state(norb, nelec)
    reference_vec_ff = ffsim.hartree_fock_state(norb, nelec)

    ansatz_seed_xq = UCJBalancedDFSeed(
        t2=cc.t2,
        t1=cc.t1,
        n_reps=n_reps,
        tol=tol,
        optimize=optimize_df,
    ).build_ansatz()

    parametrization_xq = GaugeFixedUCJSpinBalancedParameterization(
        norb=norb,
        nocc=nocc,
        n_layers=ansatz_seed_xq.n_layers,
        with_final_orbital_rotation=ansatz_seed_xq.final_orbital_rotation is not None,
    )

    x0_xq = parametrization_xq.parameters_from_ansatz(ansatz_seed_xq)
    n_params_xq = parametrization_xq.n_params
    params_to_vec_xq = parametrization_xq.params_to_vec(reference_vec_xq, nelec)

    vec_seed_xq = ansatz_seed_xq.apply(reference_vec_xq, nelec=nelec, copy=True)
    e_seed_xq = ham_xq.expectation(vec_seed_xq)

    moldata = ffsim.MolecularData.from_scf(mf, active_space=active_space)
    ham_ff = ffsim.linear_operator(
        moldata.hamiltonian,
        norb=moldata.norb,
        nelec=moldata.nelec,
    )

    ucj_ff = ffsim.UCJOpSpinBalanced.from_t_amplitudes(
        cc.t2,
        t1=cc.t1,
        n_reps=n_reps,
        tol=tol,
        optimize=optimize_df,
    )
    x0_ff = ucj_ff.to_parameters()
    n_params_ff = len(x0_ff)
    with_final_orbital_rotation_ff = ucj_ff.final_orbital_rotation is not None

    def params_to_vec_ff(params):
        op = ffsim.UCJOpSpinBalanced.from_parameters(
            np.asarray(params, dtype=np.float64),
            norb=norb,
            n_reps=ucj_ff.n_reps,
            interaction_pairs=None,
            with_final_orbital_rotation=with_final_orbital_rotation_ff,
        )
        return ffsim.apply_unitary(
            reference_vec_ff,
            op,
            norb=norb,
            nelec=nelec,
            copy=True,
        )

    vec_seed_ff = params_to_vec_ff(x0_ff)
    e_seed_ff = float(np.vdot(vec_seed_ff, ham_ff @ vec_seed_ff).real)

    print(f"R = {R:.6f}", flush=True)
    print(f"active_space = {active_space}", flush=True)
    print(f"frozen = {frozen}", flush=True)
    print(f"norb = {norb}", flush=True)
    print(f"nelec = {nelec}", flush=True)
    print(f"FCI energy = {e_fci:.12f}", flush=True)
    print(f"HF energy = {e_hf:.12f}", flush=True)
    print(f"CCSD energy = {cc.e_tot:.12f}", flush=True)
    print(f"xquces seed energy = {e_seed_xq:.12f}", flush=True)
    print(f"ffsim seed energy = {e_seed_ff:.12f}", flush=True)
    print(f"xquces n_params = {n_params_xq}", flush=True)
    print(f"ffsim n_params = {n_params_ff}", flush=True)

    vec_seed_direct = ansatz_seed_xq.apply(reference_vec_xq, nelec=nelec, copy=True)
    e_seed_direct = ham_xq.expectation(vec_seed_direct)

    x0_xq = parametrization_xq.parameters_from_ansatz(ansatz_seed_xq)
    vec_seed_roundtrip = params_to_vec_xq(x0_xq)
    e_seed_roundtrip = ham_xq.expectation(vec_seed_roundtrip)

    print("direct seed energy   =", e_seed_direct)
    print("roundtrip seed energy =", e_seed_roundtrip)
    print("state diff =", np.linalg.norm(vec_seed_direct - vec_seed_roundtrip))

    trace_xq = []
    trace_ff = []

    it_xq = {"k": 0}
    it_ff = {"k": 0}

    def callback_xq(intermediate_result):
        it_xq["k"] += 1
        energy = float(intermediate_result.fun)
        err = abs(energy - e_fci)
        cond = float(getattr(intermediate_result, "cond_S", np.nan))
        reg = float(getattr(intermediate_result, "regularization", np.nan))
        if hasattr(intermediate_result, "jac") and intermediate_result.jac is not None:
            gmax = float(np.max(np.abs(intermediate_result.jac)))
        else:
            gmax = float("nan")
        trace_xq.append(energy)
        print(
            f"[xquces] iter {it_xq['k']:3d}  "
            f"E = {energy:.12f}  "
            f"|E-E_FCI| = {err:.12e}  "
            f"gmax = {gmax:.12e}  "
            f"reg = {reg:.12e}  "
            f"cond = {cond:.12e}",
            flush=True,
        )
        append_row_csv(
            TRACE_CSV,
            {
                "ansatz": "xquces",
                "R": f"{R:.6f}",
                "iter": str(it_xq["k"]),
                "energy": f"{energy:.12f}",
                "error_fci": f"{err:.12e}",
                "max_abs_grad": f"{gmax:.12e}",
                "reg": f"{reg:.12e}",
                "cond": f"{cond:.12e}",
            },
            trace_header,
        )

    def callback_ff(intermediate_result):
        it_ff["k"] += 1
        energy = float(intermediate_result.fun)
        err = abs(energy - e_fci)
        cond = float(getattr(intermediate_result, "cond_S", np.nan))
        if hasattr(intermediate_result, "jac") and intermediate_result.jac is not None:
            gmax = float(np.max(np.abs(intermediate_result.jac)))
        else:
            gmax = float("nan")
        trace_ff.append(energy)
        print(
            f"[ffsim ] iter {it_ff['k']:3d}  "
            f"E = {energy:.12f}  "
            f"|E-E_FCI| = {err:.12e}  "
            f"gmax = {gmax:.12e}  "
            f"cond = {cond:.12e}",
            flush=True,
        )
        append_row_csv(
            TRACE_CSV,
            {
                "ansatz": "ffsim",
                "R": f"{R:.6f}",
                "iter": str(it_ff["k"]),
                "energy": f"{energy:.12f}",
                "error_fci": f"{err:.12e}",
                "max_abs_grad": f"{gmax:.12e}",
                "cond": f"{cond:.12e}",
            },
            trace_header,
        )

    res_xq = minimize_linear_method(
        params_to_vec_xq,
        ham_lm_xq,
        x0=x0_xq,
        maxiter=maxiter,
        ftol=1e-16,
        lindep=1e-6,
        callback=callback_xq,
    )

    res_ff = minimize_linear_method(
        params_to_vec_ff,
        ham_ff,
        x0=x0_ff,
        maxiter=maxiter,
        ftol=1e-16,
        lindep=1e-6,
        callback=callback_ff,
    )

    x_opt_xq = np.asarray(res_xq.x, dtype=np.float64)
    vec_opt_xq = params_to_vec_xq(x_opt_xq)
    e_opt_xq = ham_xq.expectation(vec_opt_xq)

    x_opt_ff = np.asarray(res_ff.x, dtype=np.float64)
    vec_opt_ff = params_to_vec_ff(x_opt_ff)
    e_opt_ff = float(np.vdot(vec_opt_ff, ham_ff @ vec_opt_ff).real)

    append_row_csv(
        SUMMARY_CSV,
        {
            "R": f"{R:.6f}",
            "norb": str(norb),
            "nelec": str(nelec),
            "E_FCI": f"{e_fci:.12f}",
            "E_HF": f"{e_hf:.12f}",
            "E_CCSD": f"{cc.e_tot:.12f}",
            "E_xquces_seed": f"{e_seed_xq:.12f}",
            "E_xquces_opt": f"{e_opt_xq:.12f}",
            "E_ffsim_seed": f"{e_seed_ff:.12f}",
            "E_ffsim_opt": f"{e_opt_ff:.12f}",
            "n_params_xquces": str(n_params_xq),
            "n_params_ffsim": str(n_params_ff),
        },
        summary_header,
    )

    iters_xq = np.arange(1, len(trace_xq) + 1)
    iters_ff = np.arange(1, len(trace_ff) + 1)
    err_xq = np.abs(np.asarray(trace_xq) - e_fci)
    err_ff = np.abs(np.asarray(trace_ff) - e_fci)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 5))
    plt.semilogy(iters_xq, err_xq, marker="o", label=f"xquces ({n_params_xq} params)")
    plt.semilogy(iters_ff, err_ff, marker="s", label=f"ffsim ({n_params_ff} params)")
    plt.xlabel("LM iteration")
    plt.ylabel(r"$|E - E_{\mathrm{FCI}}|$")
    plt.title(f"{molecule} {basis} R={R:.2f} Å")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_PATH, dpi=200)
    plt.close()

    print(f"xquces success = {res_xq.success}", flush=True)
    print(f"xquces message = {res_xq.message}", flush=True)
    print(f"xquces final energy = {e_opt_xq:.12f}", flush=True)
    print(f"ffsim success = {res_ff.success}", flush=True)
    print(f"ffsim message = {res_ff.message}", flush=True)
    print(f"ffsim final energy = {e_opt_ff:.12f}", flush=True)
    print(f"Wrote trace CSV: {TRACE_CSV.resolve()}", flush=True)
    print(f"Wrote summary CSV: {SUMMARY_CSV.resolve()}", flush=True)
    print(f"Wrote figure: {FIG_PATH.resolve()}", flush=True)


if __name__ == "__main__":
    main()
