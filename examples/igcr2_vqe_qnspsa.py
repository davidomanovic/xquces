from __future__ import annotations

import argparse
import csv
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyscf
import scipy.optimize
import pyscf.cc
import pyscf.gto
import pyscf.mcscf
import pyscf.scf
from ffsim.qiskit.gates import PrepareSlaterDeterminantJW
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from xquces.basis import occ_rows, reshape_state
from xquces.hamiltonians import MolecularHamiltonianLinearOperator
from xquces.gcr.igcr2 import IGCR2SpinBalancedParameterization
from xquces.qiskit.gates.diag_2 import Diag2SpinBalancedJW
from xquces.qiskit.gates.igcr2 import spin_balanced_rzz_circuit_gauge
from xquces.qiskit.gates.orbital_rotations import OrbitalRotationJW
from xquces.states import hartree_fock_state
from xquces.ucj.init import UCJBalancedDFSeed


MOLECULE = "N2"
DEFAULT_BASIS = "sto-6g"
DEFAULT_R = 0.9
DEFAULT_MAXITER = 300
# Calibration: find lr/perturbation by sampling the gradient at the seed point.
# The calibrate_c perturbation must be small enough not to escape the basin.
DEFAULT_CALIBRATE = True
DEFAULT_CALIBRATE_C = 0.01
DEFAULT_TARGET_MAGNITUDE = 0.1  # desired SPSA step norm (before QN preconditioning)
DEFAULT_STABILITY_CONSTANT = 0.0
DEFAULT_ALPHA = 0.2
DEFAULT_GAMMA = 0.101
# Fallback values used when --no-calibrate is passed.
DEFAULT_LEARNING_RATE = 0.01
DEFAULT_PERTURBATION = 0.01
# For 134 parameters the SPSA gradient norm is ~sqrt(134)*||g_true|| ≈ 11.6*||g_true||.
# With QFI eigenvalues 0..~1, reg must be large enough that (QFI + reg*I)^{-1}
# does not amplify the ~sqrt(n) SPSA noise into huge steps.
# reg=2 makes every direction's preconditioned step norm ≤ ||g_spsa||/2,
# which with calibrated lr gives steps of the right magnitude.
DEFAULT_REGULARIZATION = 1.0
# Averaging over more resamplings reduces the SPSA noise by 1/sqrt(resamplings).
# resamplings=4 reduces the sqrt(134) noise factor to sqrt(134/4)≈5.8.
DEFAULT_RESAMPLINGS = 4
DEFAULT_PERTURBATION_DIMS = 32
# Maximum norm of the preconditioned gradient returned by _compute_update
# (before the learning-rate scaling).  Hard cap that prevents any catastrophic
# step regardless of QFI estimate quality.  None to disable.
DEFAULT_MAX_GRADIENT_NORM = 0.2
DEFAULT_ALLOWED_INCREASE = 1e-5
DEFAULT_LINE_SEARCH_FACTORS = "1,2,4,0.5,0.25"
DEFAULT_RECALIBRATE_EVERY = 50
DEFAULT_SEED = 12345
THREADS = 48


@dataclass(frozen=True)
class SystemData:
    r: float
    norb: int
    nocc: int
    nelec: tuple[int, int]
    ham: MolecularHamiltonianLinearOperator
    phi0: np.ndarray
    e_hf: float
    e_ccsd: float
    e_fci: float
    t1: np.ndarray
    t2: np.ndarray


@dataclass
class StateCache:
    full: dict[bytes, np.ndarray]
    sector: dict[bytes, np.ndarray]


def _load_qnspsa():
    try:
        from qiskit_algorithms.optimizers import QNSPSA
        from qiskit_algorithms.optimizers.optimizer import OptimizerResult
        from qiskit_algorithms.optimizers.spsa import _validate_pert_and_learningrate
        from qiskit_algorithms.utils import algorithm_globals
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "QNSPSA lives in the optional 'qiskit-algorithms' package. "
            "Install it with `python -m pip install qiskit-algorithms` "
            "and rerun this script."
        ) from exc

    class ClippedQNSPSA(QNSPSA):
        """QN-SPSA with an optional update cap and simulator line search.

        For high-dimensional parameterizations the SPSA gradient estimator has
        O(sqrt(n)) variance.  With QFI eigenvalues near zero (from the 15
        degenerate Left-OV / Right-OV modes), (QFI + reg*I)^{-1} can still
        produce huge preconditioned steps even with moderate regularisation.
        The cap ensures no single step ever moves the parameters by more than
        ``max_gradient_norm`` (before the learning-rate scaling), regardless
        of how noisy the QFI estimate becomes.

        The optional line-search factors are a noiseless-simulator convenience:
        stock QN-SPSA tests exactly one step length and throws away the whole
        iteration if that length is bad.  Here we keep the same stochastic
        natural-gradient direction but test a few scalar multiples and accept
        the best energy-decreasing point.
        """

        def __init__(
            self,
            max_gradient_norm: float | None = None,
            line_search_factors: tuple[float, ...] | None = None,
            **kwargs,
        ):
            self._max_gradient_norm = max_gradient_norm
            self._line_search_factors = line_search_factors
            super().__init__(**kwargs)

        def _compute_update(self, loss, x, k, eps, lse_solver):
            value, update = super()._compute_update(loss, x, k, eps, lse_solver)
            if self._max_gradient_norm is not None:
                norm = float(np.linalg.norm(update))
                if norm > self._max_gradient_norm:
                    update = update * (self._max_gradient_norm / norm)
            return value, update

        def minimize(self, fun, x0, jac=None, bounds=None):  # noqa: D401
            if not self._line_search_factors:
                return super().minimize(fun=fun, x0=x0, jac=jac, bounds=bounds)

            x0 = np.asarray(x0)
            if self.learning_rate is None and self.perturbation is None:
                get_eta, get_eps = self.calibrate(
                    fun,
                    x0,
                    max_evals_grouped=self._max_evals_grouped,
                )
            else:
                get_eta, get_eps = _validate_pert_and_learningrate(
                    self.perturbation,
                    self.learning_rate,
                )
            eta, eps = get_eta(), get_eps()

            lse_solver = (
                self.lse_solver if self.lse_solver is not None else np.linalg.solve
            )
            x = np.asarray(x0, dtype=float)
            if self.initial_hessian is None:
                self._smoothed_hessian = np.identity(x.size)
            else:
                self._smoothed_hessian = self.initial_hessian

            self._nfev = 0
            if self.blocking:
                fx = fun(x)
                self._nfev += 1
                if self.allowed_increase is None:
                    self.allowed_increase = 0.0
            else:
                fx = None

            last_steps = deque([x])
            k = 0
            while k < self.maxiter:
                k += 1
                _, direction = self._compute_update(fun, x, k, next(eps), lse_solver)

                if self.trust_region:
                    norm = np.linalg.norm(direction)
                    if norm > 1:
                        direction = direction / norm

                base_update = direction * next(eta)
                candidates = []
                for factor in self._line_search_factors:
                    update = factor * base_update
                    x_candidate = x - update
                    fx_candidate = fun(x_candidate)
                    self._nfev += 1
                    candidates.append(
                        (
                            float(fx_candidate),
                            float(np.linalg.norm(update)),
                            x_candidate,
                            update,
                            factor,
                        )
                    )

                best_fx, best_step, best_x, _, _ = min(
                    candidates, key=lambda item: item[0]
                )
                accepted = True
                if self.blocking and fx is not None:
                    accepted = bool(best_fx < fx + self.allowed_increase)

                if self.callback is not None:
                    self.callback(self._nfev, best_x, best_fx, best_step, accepted)

                if not accepted:
                    if (
                        self.termination_checker is not None
                        and self.termination_checker(
                            self._nfev,
                            best_x,
                            best_fx,
                            best_step,
                            False,
                        )
                    ):
                        break
                    continue

                x = best_x
                fx = best_fx

                if self.last_avg > 1:
                    last_steps.append(best_x)
                    if len(last_steps) > self.last_avg:
                        last_steps.popleft()

                if self.termination_checker is not None and self.termination_checker(
                    self._nfev,
                    best_x,
                    best_fx,
                    best_step,
                    True,
                ):
                    break

            if self.last_avg > 1:
                x = np.mean(np.asarray(last_steps), axis=0)

            result = OptimizerResult()
            result.x = x
            result.fun = fun(x)
            result.nfev = self._nfev
            result.nit = k
            return result

    return ClippedQNSPSA, algorithm_globals


def parse_line_search_factors(value: str) -> tuple[float, ...] | None:
    value = value.strip()
    if not value or value.lower() in {"none", "off", "false", "0"}:
        return None
    factors = tuple(float(item) for item in value.split(",") if item.strip())
    if any(factor <= 0.0 for factor in factors):
        raise ValueError("line-search factors must be positive")
    return factors


def bitstring_index(occ_alpha, occ_beta, norb: int) -> int:
    alpha_bits = sum(1 << int(p) for p in occ_alpha)
    beta_bits = sum(1 << (norb + int(p)) for p in occ_beta)
    return alpha_bits + beta_bits


def jw_state_to_sector(
    vec: np.ndarray, norb: int, nelec: tuple[int, int]
) -> np.ndarray:
    occ_alpha = occ_rows(norb, nelec[0])
    occ_beta = occ_rows(norb, nelec[1])
    out = np.zeros((len(occ_alpha), len(occ_beta)), dtype=np.complex128)

    for i_alpha, alpha in enumerate(occ_alpha):
        for i_beta, beta in enumerate(occ_beta):
            out[i_alpha, i_beta] = vec[bitstring_index(alpha, beta, norb)]

    return out.reshape(-1)


def igcr2_stateprep_jw_circuit(ansatz) -> QuantumCircuit:
    gauge = spin_balanced_rzz_circuit_gauge(ansatz)
    circuit = QuantumCircuit(2 * ansatz.norb)
    occupied = (range(ansatz.nocc), range(ansatz.nocc))
    circuit.append(
        PrepareSlaterDeterminantJW(
            ansatz.norb,
            occupied,
            orbital_rotation=gauge.right,
        ),
        circuit.qubits,
    )
    circuit.append(
        Diag2SpinBalancedJW(
            ansatz.norb,
            gauge.same_spin_params,
            gauge.mixed_spin_params,
            emit_one_body_phases=False,
        ),
        circuit.qubits,
    )
    circuit.append(OrbitalRotationJW(ansatz.norb, gauge.left), circuit.qubits)
    return circuit


def append_trace(path: Path, row: dict[str, str], header: list[str]) -> None:
    new_file = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def build_n2_system(r: float, basis: str) -> SystemData:
    mol = pyscf.gto.Mole()
    mol.build(
        atom=[("N", (-0.5 * r, 0, 0)), ("N", (0.5 * r, 0, 0))],
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
    cas.fix_spin_(ss=0)
    cas.kernel(mo_coeff=mo_coeff)

    ccsd = pyscf.cc.RCCSD(
        scf,
        frozen=[i for i in range(mol.nao_nr()) if i not in active_space],
    )
    ccsd.kernel()

    ham = MolecularHamiltonianLinearOperator.from_scf(scf, active_space=active_space)
    return SystemData(
        r=r,
        norb=norb,
        nocc=n_alpha,
        nelec=nelec,
        ham=ham,
        phi0=hartree_fock_state(norb, nelec),
        e_hf=float(scf.e_tot),
        e_ccsd=float(ccsd.e_tot),
        e_fci=float(cas.e_tot),
        t1=np.asarray(ccsd.t1, dtype=np.complex128),
        t2=np.asarray(ccsd.t2, dtype=np.float64),
    )


def make_seed(system: SystemData):
    ucj_seed = UCJBalancedDFSeed(
        t2=system.t2,
        t1=system.t1,
        n_reps=1,
    ).build_ansatz()
    param = IGCR2SpinBalancedParameterization(
        norb=system.norb,
        nocc=system.nocc,
    )
    x_seed = param.parameters_from_ucj_ansatz(ucj_seed)
    return param, x_seed


def parameter_key(x: np.ndarray) -> bytes:
    return np.ascontiguousarray(np.asarray(x, dtype=np.float64)).tobytes()


def state_factory(
    param: IGCR2SpinBalancedParameterization,
    system: SystemData,
    cache: StateCache,
):
    zero = Statevector.from_label("0" * (2 * system.norb))

    def state_from_params(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        key = parameter_key(x)
        if key not in cache.full:
            ansatz = param.ansatz_from_parameters(np.asarray(x, dtype=np.float64))
            circuit = igcr2_stateprep_jw_circuit(ansatz)
            full_state = zero.evolve(circuit).data
            cache.full[key] = full_state
            cache.sector[key] = jw_state_to_sector(
                full_state, system.norb, system.nelec
            )
        return cache.full[key], cache.sector[key]

    return state_from_params


def run_qnspsa_vqe(args) -> None:
    QNSPSA, algorithm_globals = _load_qnspsa()
    algorithm_globals.random_seed = args.seed
    pyscf.lib.num_threads(args.threads)
    line_search_factors = parse_line_search_factors(args.line_search_factors)
    if args.perturbation_dims is not None and args.perturbation_dims <= 0:
        args.perturbation_dims = None

    system = build_n2_system(args.r, args.basis)
    param, x_seed = make_seed(system)
    cache = StateCache(full={}, sector={})
    state_from_params = state_factory(param, system, cache)

    calls = {"energy": 0, "fidelity": 0}

    def energy(x: np.ndarray) -> float:
        calls["energy"] += 1
        _, sector_state = state_from_params(x)
        return float(system.ham.expectation(sector_state))

    def fidelity(x: np.ndarray, y: np.ndarray) -> float:
        calls["fidelity"] += 1
        full_x, _ = state_from_params(x)
        full_y, _ = state_from_params(y)
        return float(abs(np.vdot(full_x, full_y)) ** 2)

    trace_header = [
        "R",
        "iter",
        "nfev",
        "energy",
        "stepsize",
        "accepted",
        "energy_calls",
        "fidelity_calls",
        "cache_size",
    ]
    if args.trace.exists():
        args.trace.unlink()

    def callback(*callback_args):
        # Qiskit Algorithms callback order is:
        # nfev, parameters, value, stepsize, accepted.
        nfev = callback_args[0] if len(callback_args) > 0 else ""
        value = callback_args[2] if len(callback_args) > 2 else np.nan
        stepsize = callback_args[3] if len(callback_args) > 3 else np.nan
        accepted = callback_args[4] if len(callback_args) > 4 else ""
        append_trace(
            args.trace,
            {
                "R": f"{system.r:.6f}",
                "iter": str(callback.iteration),
                "nfev": str(nfev),
                "energy": f"{float(value):.12f}",
                "stepsize": f"{float(stepsize):.12e}",
                "accepted": str(accepted),
                "energy_calls": str(calls["energy"]),
                "fidelity_calls": str(calls["fidelity"]),
                "cache_size": str(len(cache.full)),
            },
            trace_header,
        )
        print(
            f"iter={callback.iteration:04d} "
            f"E={float(value): .12f} "
            f"step={float(stepsize):.3e} "
            f"accepted={accepted} "
            f"energy_calls={calls['energy']} "
            f"fidelity_calls={calls['fidelity']}",
            flush=True,
        )
        callback.iteration += 1

    callback.iteration = 1

    seed_energy = energy(x_seed)
    print(
        "R,"
        "E_FCI,"
        "E_HF,"
        "E_CCSD,"
        "E_iGCR2_seed,"
        "E_iGCR2_QNSPSA,"
        "n_params,"
        "n_energy_calls,"
        "n_fidelity_calls,"
        "trace_csv",
        flush=True,
    )
    print(
        f"# starting QN-SPSA: R={system.r:.6f}, "
        f"norb={system.norb}, nelec={system.nelec}, n_params={param.n_params}, "
        f"seed_energy={seed_energy:.12f}",
        flush=True,
    )

    def make_schedules(x0: np.ndarray, niter: int, stage: int):
        if not args.calibrate:
            return args.learning_rate, args.perturbation

        print(
            f"# calibrating stage={stage}: c={args.calibrate_c:.3e}, "
            f"target_magnitude={args.target_magnitude:.3e}, niter={niter} ...",
            flush=True,
        )
        calibrated_lr, calibrated_pert = QNSPSA.calibrate(
            loss=energy,
            initial_point=x0,
            c=args.calibrate_c,
            stability_constant=args.stability_constant,
            target_magnitude=args.target_magnitude,
            alpha=args.alpha,
            gamma=args.gamma,
        )
        # calibrate() returns zero-arg callables whose return value is a generator.
        first_lr = next(calibrated_lr())
        last_lr = next(x for i, x in enumerate(calibrated_lr()) if i == niter - 1)
        print(
            f"# calibrated stage={stage}: lr[0]={first_lr:.4e}, lr[{niter}]={last_lr:.4e} "
            f"(decay={first_lr / last_lr:.1f}×), "
            f"stability_constant={args.stability_constant}, "
            f"alpha={args.alpha}, gamma={args.gamma}",
            flush=True,
        )
        return calibrated_lr, calibrated_pert

    print(
        f"# optimizer: reg={args.regularization}, resamplings={args.resamplings}, "
        f"perturbation_dims={args.perturbation_dims}, "
        f"max_gradient_norm={args.max_gradient_norm}, "
        f"allowed_increase={args.allowed_increase}, "
        f"line_search_factors={line_search_factors}, "
        f"recalibrate_every={args.recalibrate_every}",
        flush=True,
    )

    x_current = np.asarray(x_seed, dtype=float)
    remaining = args.maxiter
    stage = 1
    while remaining > 0:
        if args.recalibrate_every <= 0:
            chunk_iters = remaining
        else:
            chunk_iters = min(args.recalibrate_every, remaining)
        learning_rate, perturbation = make_schedules(x_current, chunk_iters, stage)
        optimizer = QNSPSA(
            max_gradient_norm=args.max_gradient_norm,
            line_search_factors=line_search_factors,
            fidelity=fidelity,
            maxiter=chunk_iters,
            blocking=not args.no_blocking,
            allowed_increase=args.allowed_increase,
            learning_rate=learning_rate,
            perturbation=perturbation,
            resamplings=args.resamplings,
            perturbation_dims=args.perturbation_dims,
            regularization=args.regularization,
            callback=callback,
        )
        result = optimizer.minimize(fun=energy, x0=x_current)
        x_current = np.asarray(result.x, dtype=float)
        remaining -= chunk_iters
        stage += 1

    final_energy = energy(x_current)
    print(
        f"{system.r:.6f},"
        f"{system.e_fci:.12f},"
        f"{system.e_hf:.12f},"
        f"{system.e_ccsd:.12f},"
        f"{seed_energy:.12f},"
        f"{final_energy:.12f},"
        f"{param.n_params},"
        f"{calls['energy']},"
        f"{calls['fidelity']},"
        f"{args.trace}",
        flush=True,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Noiseless statevector VQE for iGCR2 with Qiskit's QN-SPSA optimizer.",
    )
    parser.add_argument("--r", type=float, default=DEFAULT_R)
    parser.add_argument("--basis", type=str, default=DEFAULT_BASIS)
    parser.add_argument("--maxiter", type=int, default=DEFAULT_MAXITER)
    # Calibration (default: on).  Uses QNSPSA.calibrate() to estimate the gradient
    # scale at the seed point and derive learning-rate / perturbation schedules.
    parser.add_argument(
        "--no-calibrate",
        dest="calibrate",
        action="store_false",
        help="Skip calibration and use explicit --learning-rate / --perturbation.",
    )
    parser.set_defaults(calibrate=DEFAULT_CALIBRATE)
    parser.add_argument(
        "--calibrate-c",
        type=float,
        default=DEFAULT_CALIBRATE_C,
        help="Perturbation size used during calibration (default: %(default)s).",
    )
    parser.add_argument(
        "--target-magnitude",
        type=float,
        default=DEFAULT_TARGET_MAGNITUDE,
        help="Desired Euclidean norm of the first SPSA gradient step (default: %(default)s).",
    )
    parser.add_argument(
        "--stability-constant",
        type=float,
        default=DEFAULT_STABILITY_CONSTANT,
        dest="stability_constant",
        help=(
            "A in lr_k = a/(A+k+1)^alpha.  "
            "In Qiskit's calibration this also shrinks lr[0], so the default is 0. "
            "Use a positive value only if the first calibrated steps are too large."
        ),
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help=(
            "Power-law exponent for the learning-rate schedule.  "
            "The standard SPSA value is 0.602; the default is flatter for noiseless "
            "statevector simulation (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=DEFAULT_GAMMA,
        help="Power-law exponent for the perturbation schedule (default: %(default)s).",
    )
    # Fallback manual values (used only with --no-calibrate).
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--perturbation", type=float, default=DEFAULT_PERTURBATION)
    parser.add_argument(
        "--regularization",
        type=float,
        default=DEFAULT_REGULARIZATION,
        help=(
            "Metric-tensor regularisation λ: step = (H + λI)^{-1} g.  "
            "Should be >> smallest metric eigenvalue to suppress degenerate directions "
            "(default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--allowed-increase", type=float, default=DEFAULT_ALLOWED_INCREASE
    )
    parser.add_argument(
        "--line-search-factors",
        type=str,
        default=DEFAULT_LINE_SEARCH_FACTORS,
        help=(
            "Comma-separated scalar multiples of the QN-SPSA step to test each "
            "iteration; empty/none/off disables the simulator line search "
            "(default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--recalibrate-every",
        type=int,
        default=DEFAULT_RECALIBRATE_EVERY,
        help=(
            "Restart/recalibrate QN-SPSA every N iterations.  This reheats the "
            "learning-rate schedule and recalibrates the gradient scale at the "
            "current point.  Set to 0 to use one schedule for the full run "
            "(default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--resamplings",
        type=int,
        default=DEFAULT_RESAMPLINGS,
        help="Number of gradient resamplings per step for noise reduction (default: %(default)s).",
    )
    parser.add_argument(
        "--perturbation-dims",
        type=int,
        default=DEFAULT_PERTURBATION_DIMS,
        help=(
            "Number of randomly selected parameters perturbed per SPSA sample.  "
            "Using a subspace lowers high-dimensional SPSA variance; set to 0 "
            "to perturb all parameters (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--max-gradient-norm",
        type=float,
        default=DEFAULT_MAX_GRADIENT_NORM,
        dest="max_gradient_norm",
        help=(
            "Hard cap on the preconditioned gradient norm before lr scaling.  "
            "Prevents catastrophic steps from noisy QFI estimates.  "
            "None to disable (default: %(default)s)."
        ),
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--threads", type=int, default=THREADS)
    parser.add_argument("--no-blocking", action="store_true")
    parser.add_argument(
        "--trace",
        type=Path,
        default=Path(
            f"output/{MOLECULE}_{DEFAULT_BASIS}_igcr2_qnspsa_R{DEFAULT_R:.3f}.csv"
        ),
    )
    return parser.parse_args()


def main() -> None:
    run_qnspsa_vqe(parse_args())


if __name__ == "__main__":
    main()
