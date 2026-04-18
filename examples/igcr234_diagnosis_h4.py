from __future__ import annotations

import argparse

import ffsim
import numpy as np
from pyscf import lib as pyscf_lib
from threadpoolctl import threadpool_limits

from xquces.basis import occ_rows
from xquces.gcr import make_restricted_gcr_jacobian
from xquces.gcr.igcr2 import IGCR2SpinRestrictedParameterization
from xquces.gcr.igcr3 import IGCR3SpinRestrictedParameterization
from xquces.gcr.igcr4 import IGCR4SpinRestrictedParameterization
from xquces.gcr.spin_balanced_igcr4 import (
    FixedOrbitalDiagonalModel,
    IGCR4SpinBalancedFixedSectorParameterization,
    IGCR4SpinSeparatedFixedSectorParameterization,
    make_spin_orbital_diagonal_basis,
    restricted_igcr4_phase_vector,
)
from xquces.optimize import (
    build_dense_hamiltonian,
    make_expectation_penalty_state_objective,
    make_state_objective,
    make_projector_penalty_state_objective,
    minimize_bfgs,
    minimize_metric_bfgs,
)
from xquces.states import hartree_fock_state, open_shell_singlet_state
from xquces.ucj.init import UCJRestrictedProjectedDFSeed
from xquces.utils import (
    apply_spin_square,
    active_hamiltonian_from_casscf,
    active_nelec_from_mo_occ,
    active_space_from_frozen_core,
    build_h4_square_mol,
    frozen_orbitals_from_active_space,
    run_casscf,
    run_rccsd,
    run_lowest_rhf,
    spin_square,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--R", type=float, default=1.0)
    parser.add_argument("--basis", default="6-31g")
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--scf-init-guess", default="atom")
    parser.add_argument("--restricted-maxiter", type=int, default=10000)
    parser.add_argument("--diagonal-maxiter", type=int, default=10000)
    parser.add_argument("--spin-balanced-maxiter", type=int, default=10000)
    parser.add_argument("--gtol", type=float, default=1e-8)
    parser.add_argument("--max-body", type=int, default=4)
    parser.add_argument("--unrestricted-diagonal", action="store_true")
    parser.add_argument("--spin-separated-orbitals", action="store_true")
    parser.add_argument("--root-penalty", type=float, default=0.0)
    parser.add_argument("--root-penalty-start", type=int, default=1)
    parser.add_argument("--root-penalty-count", type=int, default=1)
    parser.add_argument("--root-penalty-polish-maxiter", type=int, default=0)
    parser.add_argument("--post-root-energy-maxiter", type=int, default=0)
    parser.add_argument(
        "--spin-target",
        type=float,
        default=0.0,
        help="Target <S^2> value used for spin-root reporting and spin penalty.",
    )
    parser.add_argument(
        "--spin-root-tol",
        type=float,
        default=1e-5,
        help="Tolerance for selecting FCI roots near --spin-target.",
    )
    parser.add_argument(
        "--spin-penalty",
        type=float,
        default=0.01,
        help="Penalty weight for (<S^2> - spin_target)^2 in the full stage.",
    )
    parser.add_argument(
        "--reference",
        choices=(
            "rhf",
            "open-shell-singlet",
            "diagnostic-fci-det",
            "diagnostic-fci-pair",
        ),
        default="rhf",
        help=(
            "Reference vector for the ansatz. The diagnostic-fci-* options use "
            "the exact vector only to test reference-sector limitations."
        ),
    )
    parser.add_argument("--low-eigenvalues", type=int, default=6)
    parser.add_argument("--fci-subspace-tol", type=float, default=1e-10)
    return parser.parse_args()


def progress_callback(label, cache, every=100):
    n = 0

    def callback(_intermediate_result):
        nonlocal n
        n += 1
        if n % every:
            return
        g = cache.get("g")
        gmax = np.nan if g is None else float(np.max(np.abs(g)))
        energy = cache.get("energy")
        population = cache.get("projector_population")
        operator_expectation = cache.get("operator_expectation")
        if energy is None:
            print(
                f"[{label}] Iter {n}: E = {float(cache['f']):.12f}, "
                f"gmax = {gmax:.2e}",
                flush=True,
            )
        elif population is not None:
            print(
                f"[{label}] Iter {n}: Obj = {float(cache['f']):.12f}, "
                f"E = {float(energy):.12f}, root_pop = {float(population):.3e}, "
                f"gmax = {gmax:.2e}",
                flush=True,
            )
        elif operator_expectation is not None:
            print(
                f"[{label}] Iter {n}: Obj = {float(cache['f']):.12f}, "
                f"E = {float(energy):.12f}, "
                f"<S^2> = {float(operator_expectation):.6e}, "
                f"gmax = {gmax:.2e}",
                flush=True,
            )
        else:
            print(
                f"[{label}] Iter {n}: Obj = {float(cache['f']):.12f}, "
                f"E = {float(energy):.12f}, gmax = {gmax:.2e}",
                flush=True,
            )

    return callback


def determinant_label(index, norb, nelec):
    occ_a = occ_rows(norb, nelec[0])
    occ_b = occ_rows(norb, nelec[1])
    ia, ib = divmod(int(index), len(occ_b))
    alpha = tuple(int(i) for i in occ_a[ia])
    beta = tuple(int(i) for i in occ_b[ib])
    return f"alpha={alpha} beta={beta}"


def normalized(vec):
    vec = np.asarray(vec, dtype=np.complex128).reshape(-1)
    norm = np.linalg.norm(vec)
    if norm == 0.0:
        raise ValueError("zero vector cannot be normalized")
    return vec / norm


def make_reference_state(kind, norb, nelec, fci_vec):
    if kind == "rhf":
        return hartree_fock_state(norb, nelec), "RHF determinant"

    if kind == "open-shell-singlet":
        nclosed = nelec[0] - 1
        if nelec[0] != nelec[1] or nclosed < 0 or nclosed + 1 >= norb:
            raise ValueError(
                "open-shell-singlet reference requires n_alpha == n_beta and "
                "at least two frontier orbitals"
            )
        closed = tuple(range(nclosed))
        open_pair = (nclosed, nclosed + 1)
        return (
            open_shell_singlet_state(norb, nelec, closed, open_pair),
            f"open-shell singlet closed={closed} open={open_pair}",
        )

    weights = np.abs(fci_vec) ** 2
    if kind == "diagnostic-fci-det":
        index = int(np.argmax(weights))
        vec = np.zeros_like(fci_vec)
        vec[index] = 1.0
        return vec, f"largest FCI determinant ({determinant_label(index, norb, nelec)})"

    if kind == "diagnostic-fci-pair":
        indices = np.argsort(weights)[-2:]
        vec = np.zeros_like(fci_vec)
        vec[indices] = fci_vec[indices]
        labels = ", ".join(determinant_label(i, norb, nelec) for i in indices)
        return normalized(vec), f"largest two FCI determinants ({labels})"

    raise ValueError(f"unknown reference kind {kind!r}")


def subspace_overlap(state, subspace):
    state = normalized(state)
    subspace = np.asarray(subspace, dtype=np.complex128)
    return float(np.linalg.norm(subspace.conj().T @ state))


def nearest_subspace_state(state, subspace):
    state = normalized(state)
    coeff = subspace.conj().T @ state
    norm = np.linalg.norm(coeff)
    if norm == 0.0:
        return np.asarray(subspace[:, 0], dtype=np.complex128)
    return subspace @ (coeff / norm)


def spin_diagnostic(label, state, norb, nelec):
    s2 = spin_square(normalized(state), norb, nelec)
    print(f"{label} <S^2> = {s2:.8e}", flush=True)
    return s2


def spin_values(evecs, norb, nelec):
    return np.array(
        [spin_square(evecs[:, i], norb, nelec) for i in range(evecs.shape[1])],
        dtype=np.float64,
    )


def lowest_spin_root(evals, s2_values, target, tol):
    matches = np.flatnonzero(np.abs(s2_values - target) <= tol)
    if matches.size:
        return int(matches[0])
    return None


def print_spin_target_summary(evals, s2_values, target, tol):
    idx = lowest_spin_root(evals, s2_values, target, tol)
    print(
        f"lowest root any spin: index=0 E={float(evals[0]):.12f} "
        f"<S^2>={float(s2_values[0]):.6f}",
        flush=True,
    )
    if idx is None:
        print(
            f"lowest root near <S^2>={target:.6f}: not found "
            f"within tol={tol:.1e}",
            flush=True,
        )
        return None
    print(
        f"lowest root near <S^2>={target:.6f}: index={idx} "
        f"E={float(evals[idx]):.12f} <S^2>={float(s2_values[idx]):.6f} "
        f"gap_to_any={float(evals[idx] - evals[0]):.6e}",
        flush=True,
    )
    return idx


def print_low_state_overlaps(label, state, evals, evecs, n_low, norb, nelec):
    state = normalized(state)
    print(f"{label} overlaps with low FCI eigenstates:")
    for i in range(min(n_low, evals.size)):
        overlap2 = abs(np.vdot(evecs[:, i], state)) ** 2
        s2 = spin_square(evecs[:, i], norb, nelec)
        print(
            f"  {i:2d}: E={float(evals[i]): .12f} "
            f"gap={float(evals[i] - evals[0]):.3e} "
            f"overlap^2={overlap2:.6f} <S^2>={s2:.6f}"
        )


def print_model_overlap_diagnostics(label, model, params, subspace):
    state = model.state(params)
    overlap = subspace_overlap(state, subspace)
    nearest = nearest_subspace_state(state, subspace)
    bound = model.phase_overlap_bound(nearest)
    print(
        f"{label} FCI-subspace overlap={overlap:.6f}, "
        f"phase_bound_to_nearest={bound['best_overlap']:.6f}, "
        f"phase_bound^2={bound['best_overlap_squared']:.6f}, "
        f"amplitude_l2_to_nearest={bound['amplitude_l2']:.6f}",
        flush=True,
    )


def ansatz_energy(param, x, phi0, nelec, H):
    psi = param.ansatz_from_parameters(x).apply(phi0, nelec=nelec, copy=True)
    return float(np.real(np.vdot(psi, H @ psi)))


def optimize_restricted(
    label,
    param,
    x0,
    phi0,
    nelec,
    H,
    maxiter,
    gtol,
    *,
    projector_vectors=None,
    penalty_weight=0.0,
    spin_operator_action=None,
    spin_penalty_weight=0.0,
    spin_target=0.0,
):
    params_to_state = lambda x: param.ansatz_from_parameters(x).apply(
        phi0, nelec=nelec, copy=True
    )
    state_jacobian = make_restricted_gcr_jacobian(param, phi0, nelec)
    if (
        projector_vectors is not None
        and penalty_weight != 0.0
        and spin_operator_action is not None
        and spin_penalty_weight != 0.0
    ):
        raise ValueError("root and spin penalties should be run as separate diagnostics")

    if spin_operator_action is not None and spin_penalty_weight != 0.0:
        fun, jac, cache = make_expectation_penalty_state_objective(
            params_to_state,
            state_jacobian,
            H,
            spin_operator_action,
            penalty_weight=spin_penalty_weight,
            target=spin_target,
        )
    elif projector_vectors is None or penalty_weight == 0.0:
        fun, jac, cache = make_state_objective(params_to_state, state_jacobian, H)
    else:
        fun, jac, cache = make_projector_penalty_state_objective(
            params_to_state,
            state_jacobian,
            H,
            projector_vectors,
            penalty_weight=penalty_weight,
        )
    result = minimize_metric_bfgs(
        fun,
        jac,
        x0,
        state_jacobian,
        callback=progress_callback(label, cache),
        damping=1e-6,
        max_scale=30.0,
        gtol=gtol,
        maxiter=maxiter,
    )
    g = jac(result.x)
    physical_energy = ansatz_energy(param, result.x, phi0, nelec, H)
    result.physical_energy = physical_energy
    spin_value = None
    if spin_operator_action is not None:
        state = params_to_state(result.x)
        spin_value = float(np.real(np.vdot(state, spin_operator_action(state))))
        result.spin_square = spin_value
    if spin_operator_action is not None and spin_penalty_weight != 0.0:
        print(
            f"[{label}] Done: Obj={float(result.fun):.12f}, E={physical_energy:.12f}, "
            f"<S^2>={spin_value:.6e}, nit={result.nit}, "
            f"success={result.success}, gmax={np.max(np.abs(g)):.3e}",
            flush=True,
        )
    elif projector_vectors is None or penalty_weight == 0.0:
        print(
            f"[{label}] Done: E={physical_energy:.12f}, nit={result.nit}, "
            f"success={result.success}, gmax={np.max(np.abs(g)):.3e}",
            flush=True,
        )
    else:
        population = cache.get("projector_population")
        print(
            f"[{label}] Done: Obj={float(result.fun):.12f}, E={physical_energy:.12f}, "
            f"root_pop={float(population):.3e}, nit={result.nit}, "
            f"success={result.success}, gmax={np.max(np.abs(g)):.3e}",
            flush=True,
        )
    print(f"[{label}] {result.message}", flush=True)
    return result


def optimize_fixed_diagonal(label, model, x0, H, maxiter, gtol):
    fun, jac, cache = make_state_objective(model.state, model.jacobian, H)
    result = minimize_bfgs(
        fun,
        jac,
        x0,
        callback=progress_callback(label, cache),
        gtol=gtol,
        maxiter=maxiter,
    )
    g = jac(result.x)
    print(
        f"[{label}] Done: E={float(result.fun):.12f}, nit={result.nit}, "
        f"success={result.success}, gmax={np.max(np.abs(g)):.3e}",
        flush=True,
    )
    print(f"[{label}] {result.message}", flush=True)
    return result


def result_energy(result):
    return float(getattr(result, "physical_energy", result.fun))


def main():
    args = parse_args()
    if args.root_penalty and args.spin_penalty:
        raise ValueError("--root-penalty and --spin-penalty are separate diagnostics")

    threadpool_limits(args.threads)
    pyscf_lib.num_threads(args.threads)

    mol = build_h4_square_mol(args.R, args.basis)
    scf = run_lowest_rhf(
        mol,
        init_guesses=(args.scf_init_guess, "atom", "minao", "hcore", "1e"),
    )
    active_space = active_space_from_frozen_core(scf, 0)
    frozen = frozen_orbitals_from_active_space(scf, active_space)
    norb = len(active_space)
    nelec = active_nelec_from_mo_occ(scf, active_space)
    nocc = nelec[0]

    ccsd = run_rccsd(scf, frozen=frozen)
    casscf = run_casscf(scf, ncas=norb, nelecas=nelec, active_space=active_space)
    ham = active_hamiltonian_from_casscf(casscf)
    H_sparse = ffsim.linear_operator(ham, norb=norb, nelec=nelec)
    H = build_dense_hamiltonian(H_sparse, n_workers=1)
    evals, evecs = np.linalg.eigh(H)
    s2_roots = spin_values(evecs, norb, nelec)
    E_exact = float(evals[0])
    spin_target_root = lowest_spin_root(
        evals,
        s2_roots,
        target=args.spin_target,
        tol=args.spin_root_tol,
    )
    E_spin_target = None if spin_target_root is None else float(evals[spin_target_root])
    fci_vec = np.asarray(evecs[:, 0], dtype=np.complex128)
    fci_mask = evals <= E_exact + args.fci_subspace_tol
    fci_mask[0] = True
    fci_subspace = np.asarray(evecs[:, fci_mask], dtype=np.complex128)
    phi0, reference_label = make_reference_state(args.reference, norb, nelec, fci_vec)

    print("=" * 80)
    print(f"H4 square R = {args.R:.6g}, basis = {args.basis}")
    print("=" * 80)
    print(f"SCF init guess = {args.scf_init_guess}")
    print(f"norb={norb} nelec={nelec} dim={H.shape[0]}")
    print(f"E(HF)      = {scf.e_tot:.12f}")
    print(f"E(CCSD)    = {ccsd.e_tot:.12f}")
    print(f"E(CASSCF)  = {casscf.e_tot:.12f}")
    print(f"E(FCI/H)   = {E_exact:.12f}")
    print_spin_target_summary(
        evals,
        s2_roots,
        target=args.spin_target,
        tol=args.spin_root_tol,
    )
    print(
        f"FCI subspace dim={fci_subspace.shape[1]} "
        f"(tol={args.fci_subspace_tol:.1e})"
    )
    print(f"reference  = {reference_label}")
    print(
        f"E(reference) = {float(np.real(np.vdot(phi0, H @ phi0))):.12f}",
        flush=True,
    )
    spin_diagnostic("reference", phi0, norb, nelec)
    print_low_state_overlaps(
        "reference", phi0, evals, evecs, args.low_eigenvalues, norb, nelec
    )

    ucj_seed = UCJRestrictedProjectedDFSeed(
        t2=ccsd.t2,
        t1=ccsd.t1,
        n_reps=1,
    ).build_ansatz()
    p2 = IGCR2SpinRestrictedParameterization(norb=norb, nocc=nocc)
    p3 = IGCR3SpinRestrictedParameterization(norb=norb, nocc=nocc)
    p4 = IGCR4SpinRestrictedParameterization(norb=norb, nocc=nocc)

    if args.reference == "rhf":
        x2_seed = p2.parameters_from_ucj_ansatz(ucj_seed)
    else:
        x2_seed = np.zeros(p2.n_params, dtype=np.float64)
    print("\nRestricted chain")
    print(f"E(iGCR2 seed) = {ansatz_energy(p2, x2_seed, phi0, nelec, H):.12f}")
    res2 = optimize_restricted(
        "igcr2",
        p2,
        x2_seed,
        phi0,
        nelec,
        H,
        args.restricted_maxiter,
        args.gtol,
    )

    x3_seed = p3.parameters_from_igcr2_ansatz(p2.ansatz_from_parameters(res2.x))
    print(f"E(iGCR3 seed) = {ansatz_energy(p3, x3_seed, phi0, nelec, H):.12f}")
    res3 = optimize_restricted(
        "igcr3",
        p3,
        x3_seed,
        phi0,
        nelec,
        H,
        args.restricted_maxiter,
        args.gtol,
    )

    x4_seed = p4.parameters_from_igcr3_ansatz(p3.ansatz_from_parameters(res3.x))
    print(f"E(iGCR4 seed) = {ansatz_energy(p4, x4_seed, phi0, nelec, H):.12f}")
    res4 = optimize_restricted(
        "igcr4",
        p4,
        x4_seed,
        phi0,
        nelec,
        H,
        args.restricted_maxiter,
        args.gtol,
    )

    restricted_ansatz = p4.ansatz_from_parameters(res4.x)
    restricted_state = restricted_ansatz.apply(phi0, nelec=nelec, copy=True)
    print_low_state_overlaps(
        "restricted iGCR4",
        restricted_state,
        evals,
        evecs,
        args.low_eigenvalues,
        norb,
        nelec,
    )
    spin_diagnostic("restricted iGCR4", restricted_state, norb, nelec)
    spin_balanced = not args.unrestricted_diagonal
    basis = make_spin_orbital_diagonal_basis(
        norb,
        nelec,
        max_body=args.max_body,
        spin_balanced=spin_balanced,
    )
    phase_seed = restricted_igcr4_phase_vector(restricted_ansatz.diagonal, nelec)
    x_diag_seed = basis.project_phase(phase_seed)
    model = FixedOrbitalDiagonalModel.from_orbitals(
        basis,
        phi0,
        left=restricted_ansatz.left,
        right=restricted_ansatz.right,
        norb=norb,
        nelec=nelec,
    )

    label = "spin-balanced" if spin_balanced else "unrestricted"
    print(f"\nFixed-orbital {label} diagonal")
    print(
        f"raw_features={basis.raw_n_features}, active_phase_rank={basis.n_params}, "
        f"max_body={basis.max_body}"
    )
    E_seed = float(
        np.real(np.vdot(model.state(x_diag_seed), H @ model.state(x_diag_seed)))
    )
    print(f"E(projected restricted diagonal seed) = {E_seed:.12f}")
    res_diag = optimize_fixed_diagonal(
        f"{label}-diag",
        model,
        x_diag_seed,
        H,
        args.diagonal_maxiter,
        args.gtol,
    )
    print_model_overlap_diagnostics(
        f"{label} fixed-orbital",
        model,
        res_diag.x,
        fci_subspace,
    )
    print_low_state_overlaps(
        f"{label} fixed-orbital",
        model.state(res_diag.x),
        evals,
        evecs,
        args.low_eigenvalues,
        norb,
        nelec,
    )
    spin_diagnostic(f"{label} fixed-orbital", model.state(res_diag.x), norb, nelec)

    parameterization_cls = (
        IGCR4SpinSeparatedFixedSectorParameterization
        if args.spin_separated_orbitals
        else IGCR4SpinBalancedFixedSectorParameterization
    )
    p_sb = parameterization_cls(
        norb=norb,
        nelec=nelec,
        max_body=args.max_body,
        spin_balanced=spin_balanced,
    )
    x_sb_seed = p_sb.parameters_from_restricted_igcr4_ansatz(restricted_ansatz)
    orbital_label = "spin-separated" if args.spin_separated_orbitals else "restricted"
    print(f"\nFull-orbital {label} fixed-sector iGCR4 ({orbital_label} orbitals)")
    print(f"params={p_sb.n_params}, sectors={p_sb.sector_sizes()}")
    print(
        "E(projected restricted full seed) = "
        f"{ansatz_energy(p_sb, x_sb_seed, phi0, nelec, H):.12f}"
    )
    root_projectors = None
    use_direct_root_penalty = args.root_penalty and not args.root_penalty_polish_maxiter
    if args.root_penalty:
        start = args.root_penalty_start
        stop = start + args.root_penalty_count
        root_projectors = evecs[:, start:stop]
        if use_direct_root_penalty:
            print(
                f"Applying full-stage root penalty: weight={args.root_penalty:.6g}, "
                f"states={start}:{stop}",
                flush=True,
            )
    spin_operator_action = None
    if args.spin_penalty:
        spin_operator_action = lambda psi: apply_spin_square(psi, norb, nelec)
        print(
            f"Applying full-stage spin penalty: weight={args.spin_penalty:.6g}, "
            f"target <S^2>={args.spin_target:.6f}",
            flush=True,
        )
    res_sb = optimize_restricted(
        f"{label}-full",
        p_sb,
        x_sb_seed,
        phi0,
        nelec,
        H,
        args.spin_balanced_maxiter,
        args.gtol,
        projector_vectors=root_projectors if use_direct_root_penalty else None,
        penalty_weight=args.root_penalty if use_direct_root_penalty else 0.0,
        spin_operator_action=spin_operator_action,
        spin_penalty_weight=args.spin_penalty,
        spin_target=args.spin_target,
    )
    if args.root_penalty and args.root_penalty_polish_maxiter:
        start = args.root_penalty_start
        stop = start + args.root_penalty_count
        print(
            f"\nRoot-penalty polish: weight={args.root_penalty:.6g}, "
            f"states={start}:{stop}, maxiter={args.root_penalty_polish_maxiter}",
            flush=True,
        )
        res_sb = optimize_restricted(
            f"{label}-full-root-polish",
            p_sb,
            res_sb.x,
            phi0,
            nelec,
            H,
            args.root_penalty_polish_maxiter,
            args.gtol,
            projector_vectors=root_projectors,
            penalty_weight=args.root_penalty,
        )
        if args.post_root_energy_maxiter:
            print(
                f"\nPost-root energy polish: maxiter={args.post_root_energy_maxiter}",
                flush=True,
            )
            res_sb = optimize_restricted(
                f"{label}-full-post-root-energy",
                p_sb,
                res_sb.x,
                phi0,
                nelec,
                H,
                args.post_root_energy_maxiter,
                args.gtol,
            )
    sb_ansatz = p_sb.ansatz_from_parameters(res_sb.x)
    sb_model = FixedOrbitalDiagonalModel.from_orbitals(
        p_sb.diagonal_basis,
        phi0,
        left=sb_ansatz.left,
        right=sb_ansatz.right,
        norb=norb,
        nelec=nelec,
    )
    sb_diag_start = p_sb.n_left_orbital_rotation_params
    sb_diag_stop = sb_diag_start + p_sb.n_diagonal_params
    sb_diag_params = res_sb.x[sb_diag_start:sb_diag_stop]
    print_model_overlap_diagnostics(
        f"{label} full-orbital",
        sb_model,
        sb_diag_params,
        fci_subspace,
    )
    print_low_state_overlaps(
        f"{label} full-orbital",
        sb_model.state(sb_diag_params),
        evals,
        evecs,
        args.low_eigenvalues,
        norb,
        nelec,
    )
    spin_diagnostic(
        f"{label} full-orbital", sb_model.state(sb_diag_params), norb, nelec
    )

    print("\nSummary")
    print(f"E(FCI)                 = {E_exact:.12f}")
    if E_spin_target is not None:
        print(
            f"E(FCI <S^2>~{args.spin_target:.3f}) = "
            f"{E_spin_target:.12f} (root {spin_target_root})"
        )
    print(f"E(restricted iGCR2)    = {result_energy(res2):.12f}")
    print(f"E(restricted iGCR3)    = {result_energy(res3):.12f}")
    print(f"E(restricted iGCR4)    = {result_energy(res4):.12f}")
    print(f"E(fixed {label} diag) = {float(res_diag.fun):.12f}")
    print(f"E(full {label} iGCR4) = {result_energy(res_sb):.12f}")
    if E_spin_target is not None:
        print(
            f"spin-target error restricted iGCR4 = "
            f"{1000.0 * (result_energy(res4) - E_spin_target):.6f} mEh"
        )
        print(
            f"spin-target error full {label} iGCR4 = "
            f"{1000.0 * (result_energy(res_sb) - E_spin_target):.6f} mEh"
        )
    print(
        f"diag improvement vs restricted iGCR4 = {float(res4.fun - res_diag.fun):.6e}"
    )
    print(
        "full improvement vs restricted iGCR4 = "
        f"{result_energy(res4) - result_energy(res_sb):.6e}"
    )


if __name__ == "__main__":
    main()
