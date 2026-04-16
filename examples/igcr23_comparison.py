import csv
from pathlib import Path

import numpy as np
import pyscf
import pyscf.gto
import pyscf.scf
import pyscf.cc
import pyscf.fci
from scipy.sparse.linalg import LinearOperator
from ffsim.optimize import minimize_linear_method

from xquces.gcr.igcr2 import IGCR2SpinRestrictedParameterization
from xquces.gcr.igcr3 import IGCR3SpinRestrictedParameterization
from xquces.hamiltonians import MolecularHamiltonianLinearOperator
from xquces.states import hartree_fock_state
from xquces.ucj.init import UCJRestrictedProjectedDFSeed


n_f = 0
basis = "sto-6g"
r_values = np.round(np.arange(0.7, 3.0 + 1e-12, 0.1), 10)
csv_path = Path("output/h4_square_igcr23_comparison_nosym.csv")
transfer_block_diagonal = False


def make_callback(tag):
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
            f"[{tag}] Iter {it_counter['k']}: "
            f"E = {energy:.12f}, "
            f"gmax = {gmax:.2e}, "
            f"cond_S = {cond:.2e}, "
            f"reg = {regularization:.8f}, "
            f"var = {variation:.8f}",
            flush=True,
        )

    return callback


def linear_operator_from_xquces_hamiltonian(ham):
    try:
        import ffsim

        return ffsim.linear_operator(
            ffsim.MolecularHamiltonian(ham.h1, ham.eri, ham.ecore),
            norb=ham.norb,
            nelec=ham.nelec,
        )
    except Exception:
        pass

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


def build_h4_square_mol(R, basis):
    half = 0.5 * R
    mol = pyscf.gto.Mole()
    mol.build(
        atom=[
            ("H", (-half, -half, 0.0)),
            ("H", (half, -half, 0.0)),
            ("H", (half, half, 0.0)),
            ("H", (-half, half, 0.0)),
        ],
        basis=basis,
        symmetry=False,
        unit="Angstrom",
        verbose=0,
    )
    return mol


def run_scf(mol, dm0=None):
    mf = pyscf.scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.max_cycle = 200
    if dm0 is None:
        mf.kernel()
    else:
        mf.kernel(dm0=dm0)
    if not mf.converged:
        raise RuntimeError("SCF did not converge")
    return mf


def run_ccsd(mf, frozen, t1_prev=None, t2_prev=None):
    cc = pyscf.cc.RCCSD(mf, frozen=frozen)
    cc.conv_tol = 1e-9
    cc.conv_tol_normt = 1e-7
    cc.max_cycle = 200

    t1_guess = t1_prev
    t2_guess = t2_prev

    try:
        cc.kernel(t1=t1_guess, t2=t2_guess)
    except Exception:
        cc.kernel()

    if not cc.converged:
        try:
            cc.max_cycle = 400
            cc.conv_tol = 1e-8
            cc.conv_tol_normt = 1e-6
            cc.kernel(t1=cc.t1, t2=cc.t2)
        except Exception:
            pass

    if cc.t1 is None or cc.t2 is None:
        raise RuntimeError("CCSD failed to produce amplitudes")

    return cc


def build_ucj_seed(ccsd):
    return UCJRestrictedProjectedDFSeed(
        t2=ccsd.t2,
        t1=ccsd.t1,
        n_reps=1,
    ).build_ansatz()


def exact_fci_energy(ham):
    energy, _ = pyscf.fci.direct_spin1.kernel(
        ham.h1,
        ham.eri,
        ham.norb,
        ham.nelec,
        ecore=ham.ecore,
        verbose=0,
    )
    return float(energy)


def append_csv_row(path, row):
    with path.open("a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def active_mo_coeff(scf, active_space):
    return np.asarray(scf.mo_coeff[:, active_space], dtype=np.complex128)


def orbital_overlap_between(prev_mol, prev_mo, mol, mo):
    ao_overlap = pyscf.gto.intor_cross("int1e_ovlp", prev_mol, mol)
    return prev_mo.conj().T @ ao_overlap @ mo


def ansatz_energy(param, params, phi0, nelec, ham):
    psi = param.ansatz_from_parameters(params).apply(phi0, nelec=nelec, copy=True)
    return ham.expectation(psi)


def initialize_igcr3_cubic(igcr3_param, x0, phi0, nelec, H, maxiter=30):
    """Optimize only the cubic (tau/omega) parameters with orbital and pair params fixed.

    Starting from x0 (which has tau=0, omega=0 from the UCJ seed), this is
    guaranteed to return a seed energy <= the iGCR2 seed energy.
    """
    n_left = igcr3_param.n_left_orbital_rotation_params
    n_pair = igcr3_param.n_pair_params
    n_cubic = igcr3_param.n_tau_params  # combined tau+omega in reduced chart
    cubic_slice = slice(n_left + n_pair, n_left + n_pair + n_cubic)

    x_fixed = np.array(x0, copy=True)

    def cubic_params_to_vec(cubic_params):
        x = np.array(x_fixed, copy=True)
        x[cubic_slice] = cubic_params
        return igcr3_param.ansatz_from_parameters(x).apply(phi0, nelec=nelec, copy=True)

    res = minimize_linear_method(
        cubic_params_to_vec,
        H,
        x0=x_fixed[cubic_slice],
        maxiter=maxiter,
        gtol=1e-4,
        ftol=1e-8,
    )
    x_fixed[cubic_slice] = res.x
    return x_fixed


def add_candidate(candidates, label, param, params, phi0, nelec, ham):
    try:
        params = np.asarray(params, dtype=np.float64)
        energy = ansatz_energy(param, params, phi0, nelec, ham)
    except Exception as exc:
        print(f"Skipping {label}: {exc}", flush=True)
        return
    candidates.append((energy, label, params))


def transferred_parameters(param, previous_record, mol, mo):
    previous_params, previous_param, previous_mol, previous_mo = previous_record
    overlap = orbital_overlap_between(previous_mol, previous_mo, mol, mo)
    return param.transfer_parameters_from(
        previous_params,
        previous_parameterization=previous_param,
        orbital_overlap=overlap,
        block_diagonal=transfer_block_diagonal,
    )


def choose_start(tag, param, ucj_seed, history, mol, mo, phi0, nelec, ham):
    candidates = []
    add_candidate(candidates, "ucj_seed", param, ucj_seed, phi0, nelec, ham)

    transferred = []
    for age, record in enumerate(reversed(history[-2:]), start=1):
        label = "warm_prev" if age == 1 else "warm_prevprev"
        try:
            params = transferred_parameters(param, record, mol, mo)
        except Exception as exc:
            print(f"[igcr{tag}] Skipping {label}: {exc}", flush=True)
            continue
        transferred.append((label, params))
        add_candidate(candidates, label, param, params, phi0, nelec, ham)

    if len(transferred) >= 2 and transferred[0][1].shape == transferred[1][1].shape:
        extrapolated = 2.0 * transferred[0][1] - transferred[1][1]
        add_candidate(candidates, "linear_extrapolated", param, extrapolated, phi0, nelec, ham)

    if not candidates:
        raise RuntimeError(f"No valid starting candidates for iGCR{tag}")

    candidates.sort(key=lambda item: item[0])
    energy, label, params = candidates[0]
    print(
        f"Using {label} iGCR{tag} parameters "
        f"(start E = {energy:.12f}, candidates = {len(candidates)})",
        flush=True,
    )
    return np.array(params, copy=True), float(energy)


prev_dm = None
prev_t1 = None
prev_t2 = None
history_igcr2 = []
history_igcr3 = []

csv_path.parent.mkdir(parents=True, exist_ok=True)
with csv_path.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["R", "E_FCI", "E_IGCR2", "E_IGCR3"])

for R in r_values:
    print("=" * 80, flush=True)
    print(f"Geometry R = {R:.2f}", flush=True)
    print("=" * 80, flush=True)

    mol = build_h4_square_mol(R, basis)
    scf = run_scf(mol, dm0=prev_dm)
    prev_dm = scf.make_rdm1()

    active_space = list(range(n_f, mol.nao_nr()))
    norb = len(active_space)
    nelectron_cas = int(round(sum(scf.mo_occ[active_space])))
    n_alpha = (nelectron_cas + mol.spin) // 2
    n_beta = (nelectron_cas - mol.spin) // 2
    nelec = (n_alpha, n_beta)

    frozen = [i for i in range(mol.nao_nr()) if i not in active_space]
    mo_active = active_mo_coeff(scf, active_space)

    ccsd = run_ccsd(
        scf,
        frozen=frozen,
        t1_prev=prev_t1,
        t2_prev=prev_t2,
    )
    prev_t1 = np.array(ccsd.t1, copy=True)
    prev_t2 = np.array(ccsd.t2, copy=True)

    ham_xq = MolecularHamiltonianLinearOperator.from_scf(scf, active_space=active_space)
    H = linear_operator_from_xquces_hamiltonian(ham_xq)
    Phi0 = hartree_fock_state(norb, nelec)

    ucj_restricted_seed = build_ucj_seed(ccsd)

    igcr2_param = IGCR2SpinRestrictedParameterization(
        norb=norb,
        nocc=n_alpha,
        real_right_orbital_chart=True,
    )
    igcr3_param = IGCR3SpinRestrictedParameterization(
        norb=norb,
        nocc=n_alpha,
        real_right_orbital_chart=True,
    )

    x0_igcr2_seed = igcr2_param.parameters_from_ucj_ansatz(ucj_restricted_seed)
    x0_igcr3_seed = igcr3_param.parameters_from_ucj_ansatz(ucj_restricted_seed)
    x0_igcr3_seed = initialize_igcr3_cubic(igcr3_param, x0_igcr3_seed, Phi0, nelec, H)

    x0_igcr2, E_iGCR2_seed = choose_start(
        "2",
        igcr2_param,
        x0_igcr2_seed,
        history_igcr2,
        mol,
        mo_active,
        Phi0,
        nelec,
        ham_xq,
    )
    x0_igcr3, E_iGCR3_seed = choose_start(
        "3",
        igcr3_param,
        x0_igcr3_seed,
        history_igcr3,
        mol,
        mo_active,
        Phi0,
        nelec,
        ham_xq,
    )

    print(f"iGCR2 number of parameters: {len(x0_igcr2)}", flush=True)
    print(f"iGCR3 number of parameters: {len(x0_igcr3)}", flush=True)
    print(f"E(iGCR2 seed) = {E_iGCR2_seed:.12f}", flush=True)
    print(f"E(iGCR3 seed) = {E_iGCR3_seed:.12f}", flush=True)

    def params_to_vec_igcr2(x):
        return igcr2_param.ansatz_from_parameters(x).apply(Phi0, nelec=nelec, copy=True)

    def params_to_vec_igcr3(x):
        return igcr3_param.ansatz_from_parameters(x).apply(Phi0, nelec=nelec, copy=True)

    res_igcr2 = minimize_linear_method(
        params_to_vec_igcr2,
        H,
        x0=x0_igcr2,
        maxiter=100,
        gtol=1e-6,
        ftol=1e-12,
        callback=make_callback("igcr2"),
    )

    res_igcr3 = minimize_linear_method(
        params_to_vec_igcr3,
        H,
        x0=x0_igcr3,
        maxiter=100,
        gtol=1e-6,
        ftol=1e-12,
        callback=make_callback("igcr3"),
    )

    history_igcr2.append(
        (
            np.array(res_igcr2.x, copy=True),
            igcr2_param,
            mol,
            mo_active,
        )
    )
    history_igcr3.append(
        (
            np.array(res_igcr3.x, copy=True),
            igcr3_param,
            mol,
            mo_active,
        )
    )
    history_igcr2 = history_igcr2[-2:]
    history_igcr3 = history_igcr3[-2:]

    E_HF = float(scf.e_tot)
    E_FCI = exact_fci_energy(ham_xq)
    E_iGCR2_opt = float(res_igcr2.fun)
    E_iGCR3_opt = float(res_igcr3.fun)

    append_csv_row(
        csv_path,
        [R, E_FCI, E_iGCR2_opt, E_iGCR3_opt],
    )

    print("Final results:", flush=True)
    print(f"E(HF) = {E_HF:.12f}", flush=True)
    print(f"E(FCI) = {E_FCI:.12f}", flush=True)
    print(f"E(iGCR2) = {E_iGCR2_opt:.12f}", flush=True)
    print(f"E(iGCR3) = {E_iGCR3_opt:.12f}", flush=True)

    denom = E_FCI - E_HF
    if abs(denom) > 1e-14:
        corr2 = (E_iGCR2_opt - E_HF) / denom * 100.0
        corr3 = (E_iGCR3_opt - E_HF) / denom * 100.0
        print(f"Correlation energy captured by iGCR2: {corr2:.2f}%", flush=True)
        print(f"Correlation energy captured by iGCR3: {corr3:.2f}%", flush=True)
    else:
        print("Correlation energy denominator too small to report percentage", flush=True)

print(f"Done. Results appended to {csv_path}", flush=True)
