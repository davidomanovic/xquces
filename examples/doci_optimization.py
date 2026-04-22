# Optimization of GCR2 applied onto a DOCI reference state

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dev" / "xquces" / "python"))

import ffsim
import numpy as np
import pyscf
import pyscf.mcscf
import xquces
from scipy.sparse.linalg import eigsh
from threadpoolctl import threadpool_limits

from xquces.gcr import GCR2DOCIReferenceParameterization, make_restricted_gcr_jacobian
from xquces.gcr.igcr2 import orbital_relabeling_from_overlap
from xquces.optimize import build_dense_hamiltonian, make_state_objective, minimize_linear_method
from xquces.ucj.init import UCJRestrictedProjectedDFSeed
from xquces.utils import (
    active_hamiltonian_from_casscf,
    active_mo_coeff_from_casscf,
    active_nelec_from_mo_occ,
    active_space_from_frozen_core,
    build_h4_square_mol,
    build_n2_mol,
    frozen_orbitals_from_active_space,
    orbital_overlap_between,
    run_casscf,
    run_lowest_rhf,
    run_rccsd,
    spin_square,
)

molecule = "h4_square"
basis = "sto-6g"
start = 1.0
stop = 3.4
step = 0.1
threads = 12
dense_h_workers = threads
n_frozen = 2
maxiter = 10000
gtol = 5e-6
ftol = 1e-12
scf_init_guess = "atom"
use_continuation = True
use_orbital_alignment = True
print_every = 1
dense_threshold = 4096
n_fci_roots = 1
s2_tol = 1e-4
output = Path(f"output/{molecule}_{basis}_gcr2_doci_reference.csv")


def append_row_csv(path: Path, row: dict[str, str], header: list[str]) -> None:
    new_file = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def build_mol(name: str, r: float, basis_name: str):
    if name == "h4_square":
        return build_h4_square_mol(r, basis_name, symmetry=False)
    if name == "n2":
        return build_n2_mol(r, basis_name, symmetry=False)
    raise ValueError(f"unknown molecule: {name}")


def run_casscf_singlet(scf, norb: int, nelec: tuple[int, int]):
    mc = pyscf.mcscf.CASSCF(scf, norb, nelec)
    mc.fix_spin_(ss=0)
    mc.kernel()
    return mc


def exact_singlet_root(
    hamiltonian,
    norb: int,
    nelec: tuple[int, int],
) -> tuple[float, float, np.ndarray]:
    dim = hamiltonian.shape[0]
    if dim <= dense_threshold:
        dense = build_dense_hamiltonian(hamiltonian, n_workers=dense_h_workers)
        dense = 0.5 * (dense + dense.conj().T)
        evals, evecs = np.linalg.eigh(dense)
    else:
        evals, evecs = eigsh(
            hamiltonian,
            k=min(n_fci_roots, dim - 1),
            which="SA",
            tol=1e-10,
            return_eigenvectors=True,
        )
        order = np.argsort(evals)
        evals = evals[order]
        evecs = evecs[:, order]

    s2_values = [spin_square(evecs[:, i], norb, nelec) for i in range(evecs.shape[1])]
    for energy, s2, vec in zip(evals, s2_values, evecs.T):
        if abs(s2) <= s2_tol:
            psi = np.asarray(vec, dtype=np.complex128)
            psi /= np.linalg.norm(psi)
            return float(np.real(energy)), float(s2), psi

    best = int(np.argmin(np.abs(s2_values)))
    psi = np.asarray(evecs[:, best], dtype=np.complex128)
    psi /= np.linalg.norm(psi)
    return float(np.real(evals[best])), float(s2_values[best]), psi


def align_active_orbitals_to_previous(casscf, previous, mol) -> np.ndarray:
    _, _, previous_mol, previous_mo = previous
    active_mo = active_mo_coeff_from_casscf(casscf)
    overlap = orbital_overlap_between(previous_mol, previous_mo, mol, active_mo)

    old_for_new, phases = orbital_relabeling_from_overlap(
        overlap,
        nocc=None,
        block_diagonal=False,
    )
    current_for_old = np.empty_like(old_for_new)
    current_for_old[old_for_new] = np.arange(old_for_new.size)
    aligned_active_mo = (
        active_mo[:, current_for_old] * np.conj(phases[current_for_old])[np.newaxis, :]
    )

    mo_coeff = np.array(casscf.mo_coeff, copy=True)
    start_idx = casscf.ncore
    stop_idx = start_idx + casscf.ncas
    mo_coeff[:, start_idx:stop_idx] = aligned_active_mo
    casscf.mo_coeff = mo_coeff
    return aligned_active_mo


def build_doci_initialized_seed(
    parameterization: GCR2DOCIReferenceParameterization,
    ucj_ansatz,
    nelec: tuple[int, int],
    hamiltonian,
) -> np.ndarray:
    base = parameterization._base
    base_params = base.parameters_from_ucj_ansatz(ucj_ansatz)
    base_ansatz = base.ansatz_from_parameters(base_params)

    n_left = parameterization.n_left_orbital_rotation_params
    n_diag = parameterization.n_diag_params

    left = np.asarray(base_params[:n_left], dtype=np.float64)
    diag = np.asarray(base_params[n_left : n_left + n_diag], dtype=np.float64)
    middle = np.asarray(
        parameterization._extract_full_rotation_params(base_ansatz.right),
        dtype=np.float64,
    )
    doci_reference = np.zeros(
        parameterization.n_pair_reference_params,
        dtype=np.float64,
    )

    x_pre = np.concatenate([left, diag, doci_reference, middle])
    seed_ansatz = parameterization.ansatz_from_parameters(x_pre)

    dim_doci = xquces.doci_dimension(parameterization.norb, nelec)
    dim_full = hamiltonian.shape[0]

    dressed_basis = np.empty((dim_full, dim_doci), dtype=np.complex128)
    for col in range(dim_doci):
        amps = np.zeros(dim_doci, dtype=np.float64)
        amps[col] = 1.0
        det = xquces.doci_state(parameterization.norb, nelec, amplitudes=amps)
        dressed_basis[:, col] = seed_ansatz.apply(det, nelec=nelec, copy=True)

    h_dressed = np.column_stack(
        [hamiltonian @ dressed_basis[:, col] for col in range(dim_doci)]
    )
    h_doci = dressed_basis.conj().T @ h_dressed
    h_real = np.asarray(0.5 * (h_doci + h_doci.conj().T).real, dtype=np.float64)

    _, evecs = np.linalg.eigh(h_real)
    coeffs = np.asarray(evecs[:, 0], dtype=np.float64)
    doci_reference = xquces.doci_parameters_from_amplitudes(coeffs)

    return np.concatenate([left, diag, doci_reference, middle])


def transfer_parameters(param, previous, mol, active_mo):
    previous_params, previous_param, previous_mol, previous_mo = previous
    overlap = orbital_overlap_between(previous_mol, previous_mo, mol, active_mo)
    return param.transfer_parameters_from(
        previous_params,
        previous_parameterization=previous_param,
        orbital_overlap=overlap,
    )


def continuation_seed(param, previous, mol, active_mo, fallback):
    if previous is None:
        return np.asarray(fallback, dtype=np.float64)
    if hasattr(param, "transfer_parameters_from"):
        return transfer_parameters(param, previous, mol, active_mo)
    previous_params, previous_param, _, _ = previous
    if previous_param.n_params != param.n_params:
        return np.asarray(fallback, dtype=np.float64)
    return np.asarray(previous_params, dtype=np.float64).copy()


def state_energy(param, x, phi0, nelec, hamiltonian) -> float:
    psi = param.ansatz_from_parameters(x).apply(phi0, nelec=nelec, copy=True)
    return float(np.real(np.vdot(psi, hamiltonian @ psi)))


def evaluate_state(param, x, phi0, nelec, norb, hamiltonian, psi_fci, e_fci):
    psi = param.ansatz_from_parameters(x).apply(phi0, nelec=nelec, copy=True)
    psi = psi / np.linalg.norm(psi)
    energy = float(np.real(np.vdot(psi, hamiltonian @ psi)))
    s2 = spin_square(psi, norb, nelec)
    overlap2 = float(abs(np.vdot(psi_fci, psi)) ** 2)
    return {
        "energy": energy,
        "s2": s2,
        "overlap2": overlap2,
        "gap_mHa": 1000.0 * (energy - e_fci),
    }


def optimize_gcr2_doci_reference(
    param: GCR2DOCIReferenceParameterization,
    x0: np.ndarray,
    phi0: np.ndarray,
    nelec: tuple[int, int],
    hamiltonian,
    trace: Path,
    r: float,
):
    trace_header = [
        "R",
        "iter",
        "energy",
        "max_abs_grad",
    ]

    params_to_vec = param.params_to_vec(phi0, nelec)
    state_jacobian = make_restricted_gcr_jacobian(param, phi0, nelec)
    fun, jac, cache = make_state_objective(params_to_vec, state_jacobian, hamiltonian)

    counter = {"value": 0}

    def callback(intermediate_result):
        counter["value"] += 1
        if counter["value"] % print_every:
            return
        energy = float(getattr(intermediate_result, "fun", np.nan))
        lm_jac = getattr(intermediate_result, "jac", None)
        if lm_jac is None:
            grad = cache["g"]
            gmax = float(np.max(np.abs(grad))) if grad is not None else float("nan")
        else:
            gmax = float(np.max(np.abs(lm_jac)))
        append_row_csv(
            trace,
            {
                "R": f"{r:.8f}",
                "iter": str(counter["value"]),
                "energy": f"{energy:.14f}",
                "max_abs_grad": f"{gmax:.12e}",
            },
            trace_header,
        )
        cond_s = float(getattr(intermediate_result, "cond_S", np.nan))
        regularization = float(getattr(intermediate_result, "regularization", np.nan))
        variation = float(getattr(intermediate_result, "variation", np.nan))
        print(
            f"[R={r:.3f}] Iter {counter['value']}: E = {energy:.12f}, gmax = {gmax:.2e}, "
            f"cond_S = {cond_s:.3e}, reg = {regularization:.2e}, var = {variation:.3f}",
            flush=True,
        )

    return minimize_linear_method(
        params_to_vec,
        hamiltonian,
        x0=x0,
        jac=state_jacobian,
        maxiter=maxiter,
        gtol=gtol,
        ftol=ftol,
        callback=callback,
    )


def main() -> None:
    pyscf.lib.num_threads(threads)

    trace = output.with_name(output.stem + "_trace.csv")
    if output.exists():
        output.unlink()
    if trace.exists():
        trace.unlink()

    header = [
        "R",
        "E_FCI",
        "E_HF",
        "E_CCSD",
        "E_initial",
        "E_opt",
        "S2",
        "overlap2",
        "n_params",
    ]

    print(",".join(header), flush=True)

    rs = np.linspace(start, stop, num=round((stop - start) / step) + 1)

    previous_dm = None
    previous_t1 = None
    previous_t2 = None
    previous_record = None

    with threadpool_limits(limits=threads):
        for i_r, r in enumerate(rs):
            print("=" * 80, flush=True)
            print(f"Geometry R = {r:.3f}", flush=True)
            print("=" * 80, flush=True)

            mol = build_mol(molecule, float(r), basis)
            scf = run_lowest_rhf(
                mol,
                dm0=previous_dm,
                init_guesses=(scf_init_guess, "atom", "minao", "hcore", "1e"),
                random_trials=0,
            )
            previous_dm = scf.make_rdm1()

            active_space = active_space_from_frozen_core(scf, n_frozen)
            frozen = frozen_orbitals_from_active_space(scf, active_space)
            norb = len(active_space)
            nelec = active_nelec_from_mo_occ(scf, active_space)
            nocc = nelec[0]

            ccsd = run_rccsd(scf, frozen=frozen, t1=previous_t1, t2=previous_t2)
            previous_t1 = np.array(ccsd.t1, copy=True)
            previous_t2 = np.array(ccsd.t2, copy=True)

            try:
                casscf = run_casscf_singlet(scf, norb, nelec)
            except Exception:
                casscf = run_casscf(
                    scf,
                    ncas=norb,
                    nelecas=nelec,
                    active_space=active_space,
                )

            if previous_record is not None and use_orbital_alignment:
                active_mo = align_active_orbitals_to_previous(casscf, previous_record, mol)
            else:
                active_mo = active_mo_coeff_from_casscf(casscf)

            ham = active_hamiltonian_from_casscf(casscf)
            h = ffsim.linear_operator(ham, norb=norb, nelec=nelec)
            e_fci, s2_fci, psi_fci = exact_singlet_root(h, norb, nelec)
            phi0 = ffsim.hartree_fock_state(norb, nelec)

            ucj_seed = UCJRestrictedProjectedDFSeed(
                t2=ccsd.t2,
                t1=ccsd.t1,
                n_reps=1,
            ).build_ansatz()

            parameterization = GCR2DOCIReferenceParameterization(norb=norb, nocc=nocc)
            x_seed = build_doci_initialized_seed(
                parameterization,
                ucj_seed,
                nelec,
                h,
            )

            if previous_record is not None and use_continuation:
                x0 = continuation_seed(
                    parameterization,
                    previous_record,
                    mol,
                    active_mo,
                    x_seed,
                )
                seed_label = "transferred"
            else:
                x0 = np.asarray(x_seed, dtype=np.float64)
                seed_label = "UCJ+dressed-DOCI"

            print(
                f"norb={norb} nelec={nelec} params={parameterization.n_params} "
                f"(diag={parameterization.n_diag_params}, "
                f"doci={parameterization.n_pair_reference_params}, "
                f"middle={parameterization.n_middle_orbital_rotation_params})",
                flush=True,
            )
            print(
                f"E(FCI singlet) = {e_fci:.12f}  <S^2>={s2_fci:.3e}",
                flush=True,
            )
            print(f"Using {seed_label} seed", flush=True)

            e_initial = state_energy(parameterization, x0, phi0, nelec, h)
            print(f"E(initial) = {e_initial:.12f}", flush=True)

            result = optimize_gcr2_doci_reference(
                parameterization,
                x0,
                phi0,
                nelec,
                h,
                trace,
                float(r),
            )

            diag = evaluate_state(
                parameterization,
                result.x,
                phi0,
                nelec,
                norb,
                h,
                psi_fci,
                e_fci,
            )
            e_opt = diag["energy"]

            previous_record = (
                np.asarray(result.x, dtype=np.float64).copy(),
                parameterization,
                mol,
                active_mo,
            )

            print(
                f"[R={r:.3f}] Done: E={e_opt:.12f}, "
                f"gap={diag['gap_mHa']:+.3f} mHa, "
                f"<S^2>={diag['s2']:.3e}, overlap2={diag['overlap2']:.6f}, "
                f"nit={getattr(result, 'nit', '')}, success={getattr(result, 'success', False)}",
                flush=True,
            )

            row = {
                "R": f"{r:.8f}",
                "E_FCI": f"{e_fci:.14f}",
                "E_HF": f"{float(scf.e_tot):.14f}",
                "E_CCSD": f"{float(ccsd.e_tot):.14f}",
                "E_initial": f"{e_initial:.14f}",
                "E_opt": f"{e_opt:.14f}",
                "S2": f"{diag['s2']:.12e}",
                "overlap2": f"{diag['overlap2']:.12e}",
                "n_params": str(parameterization.n_params),
            }
            print(",".join(row[k] for k in header), flush=True)
            append_row_csv(output, row, header)

    print(f"Wrote CSV: {output}", flush=True)
    print(f"Wrote trace CSV: {trace}", flush=True)


if __name__ == "__main__":
    main()
