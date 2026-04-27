from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import pyscf.gto
import pyscf.mcscf
import pyscf.scf
from scipy.optimize import OptimizeResult, linear_sum_assignment
from scipy.sparse.linalg import LinearOperator

from xquces.gcr import (
    IGCR2SpinRestrictedParameterization,
    make_restricted_gcr_jacobian,
)
from xquces.optimize import minimize_linear_method
from xquces.ucj.init import UCJRestrictedProjectedDFSeed
from xquces.utils import (
    active_nelec_from_mo_occ,
    frozen_orbitals_from_active_space,
    run_lowest_rhf,
    run_rccsd,
    run_rhf,
)


@dataclass(frozen=True)
class OrbitalTrackingDiagnostics:
    occ_sv_min: float = float("nan")
    virt_sv_min: float = float("nan")
    diag_min: float = float("nan")
    diag_mean: float = float("nan")
    occ_cols: tuple[int, ...] = ()
    virt_cols: tuple[int, ...] = ()

    def as_dict(self, prefix: str = "mo_") -> dict[str, object]:
        return {
            f"{prefix}occ_sv_min": self.occ_sv_min,
            f"{prefix}virt_sv_min": self.virt_sv_min,
            f"{prefix}diag_min": self.diag_min,
            f"{prefix}diag_mean": self.diag_mean,
            f"{prefix}occ_cols": ";".join(str(i) for i in self.occ_cols),
            f"{prefix}virt_cols": ";".join(str(i) for i in self.virt_cols),
        }


def cross_ao_overlap(prev_mol, mol, intor: str = "int1e_ovlp_sph") -> np.ndarray:
    try:
        return pyscf.gto.intor_cross(intor, prev_mol, mol)
    except Exception:
        if intor == "int1e_ovlp":
            raise
        return pyscf.gto.intor_cross("int1e_ovlp", prev_mol, mol)


def track_occ_virt_orbitals(
    prev_mol,
    prev_mo_coeff: np.ndarray,
    mol,
    mo_coeff: np.ndarray,
    nocc: int,
    *,
    intor: str = "int1e_ovlp_sph",
) -> tuple[np.ndarray, OrbitalTrackingDiagnostics]:
    """Track occupied and virtual MO subspaces by cross-geometry overlap.

    Occupied columns are first selected by maximum overlap with the previous
    occupied subspace. Occupied and virtual blocks are then Procrustes-rotated
    separately, preserving the reference determinant while removing arbitrary
    rotations inside near-degenerate occupied/virtual subspaces.
    """

    prev_c = np.asarray(prev_mo_coeff)
    c = np.asarray(mo_coeff)
    if prev_c.ndim != 2 or c.ndim != 2:
        raise ValueError("MO coefficients must be matrices")
    if prev_c.shape[1] != c.shape[1]:
        raise ValueError("Previous and current MO coefficients must have the same nmo")

    nmo = c.shape[1]
    nocc = int(nocc)
    if nocc < 0 or nocc > nmo:
        raise ValueError("nocc must be between 0 and nmo")

    s_cross = cross_ao_overlap(prev_mol, mol, intor=intor)
    prev_occ = prev_c[:, :nocc]
    prev_virt = prev_c[:, nocc:]

    if nocc:
        occ_overlap = prev_occ.conj().T @ s_cross @ c
        row, col = linear_sum_assignment(-np.abs(occ_overlap))
        occ_cols = np.empty(nocc, dtype=int)
        occ_cols[row] = col
    else:
        occ_cols = np.array([], dtype=int)

    used = set(int(i) for i in occ_cols)
    virt_cols = np.array([j for j in range(nmo) if j not in used], dtype=int)

    c_occ0 = c[:, occ_cols]
    c_virt0 = c[:, virt_cols]

    if nocc:
        m_occ = prev_occ.conj().T @ s_cross @ c_occ0
        uo, so, vho = np.linalg.svd(m_occ, full_matrices=False)
        c_occ = c_occ0 @ (vho.conj().T @ uo.conj().T)
    else:
        so = np.array([], dtype=float)
        c_occ = c_occ0

    if c_virt0.shape[1]:
        m_virt = prev_virt.conj().T @ s_cross @ c_virt0
        uv, sv, vhv = np.linalg.svd(m_virt, full_matrices=False)
        c_virt = c_virt0 @ (vhv.conj().T @ uv.conj().T)
    else:
        sv = np.array([], dtype=float)
        c_virt = c_virt0

    tracked = np.hstack([c_occ, c_virt])
    if np.max(np.abs(np.imag(tracked))) < 1e-12:
        tracked = np.real(tracked)

    final_overlap = prev_c.conj().T @ s_cross @ tracked
    diag_abs = np.abs(np.diag(final_overlap))
    diagnostics = OrbitalTrackingDiagnostics(
        occ_sv_min=float(np.min(so)) if so.size else float("nan"),
        virt_sv_min=float(np.min(sv)) if sv.size else float("nan"),
        diag_min=float(np.min(diag_abs)) if diag_abs.size else float("nan"),
        diag_mean=float(np.mean(diag_abs)) if diag_abs.size else float("nan"),
        occ_cols=tuple(int(i) for i in occ_cols),
        virt_cols=tuple(int(i) for i in virt_cols),
    )
    return tracked, diagnostics


class OrbitalTracker:
    """Stateful forward orbital gauge tracker."""

    def __init__(self, *, enabled: bool = True, intor: str = "int1e_ovlp_sph"):
        self.enabled = bool(enabled)
        self.intor = intor
        self.prev_mol = None
        self.prev_mo_coeff: np.ndarray | None = None

    def reset(self):
        self.prev_mol = None
        self.prev_mo_coeff = None

    def track(
        self,
        mol,
        mo_coeff: np.ndarray,
        nocc: int,
    ) -> tuple[np.ndarray, OrbitalTrackingDiagnostics]:
        if not self.enabled or self.prev_mol is None or self.prev_mo_coeff is None:
            tracked = np.array(mo_coeff, copy=True)
            diagnostics = OrbitalTrackingDiagnostics()
        else:
            tracked, diagnostics = track_occ_virt_orbitals(
                self.prev_mol,
                self.prev_mo_coeff,
                mol,
                mo_coeff,
                nocc,
                intor=self.intor,
            )

        self.prev_mol = mol
        self.prev_mo_coeff = np.array(tracked, copy=True)
        return tracked, diagnostics


@dataclass
class MolecularContinuationPoint:
    mol: object
    mf: object
    ccsd: object | None
    mo_coeff: np.ndarray
    mo_tracking: OrbitalTrackingDiagnostics
    mol_data: object
    operator: LinearOperator
    norb: int
    nelec: tuple[int, int]

    @property
    def ccsd_energy(self) -> float:
        if self.ccsd is None:
            return float("nan")
        return float(np.real(self.ccsd.e_tot))


def _active_nelec(mf, active_space: Sequence[int] | None) -> tuple[int, int]:
    if active_space is None:
        n_alpha = (mf.mol.nelectron + mf.mol.spin) // 2
        n_beta = (mf.mol.nelectron - mf.mol.spin) // 2
        return int(n_alpha), int(n_beta)
    return active_nelec_from_mo_occ(mf, active_space)


def _mo_for_cas(
    mo_coeff: np.ndarray,
    active_space: Sequence[int] | None,
    ncore: int,
) -> np.ndarray:
    if active_space is None:
        return np.asarray(mo_coeff)

    nmo = mo_coeff.shape[1]
    active = [int(i) for i in active_space]
    active_set = set(active)
    if len(active_set) != len(active):
        raise ValueError("active_space contains duplicates")
    if min(active, default=0) < 0 or max(active, default=-1) >= nmo:
        raise ValueError("active_space index out of range")

    inactive = [i for i in range(nmo) if i not in active_set]
    core = inactive[:ncore]
    external = inactive[ncore:]
    if len(core) != ncore:
        raise ValueError("Not enough inactive orbitals to form the requested core")
    return np.asarray(mo_coeff)[:, core + active + external]


def make_ffsim_molecular_data(
    mf,
    *,
    mo_coeff: np.ndarray,
    active_space: Sequence[int] | None = None,
    ccsd=None,
):
    import ffsim

    nelec = _active_nelec(mf, active_space)
    ncas = mo_coeff.shape[1] if active_space is None else len(active_space)
    ncore = (mf.mol.nelectron - sum(nelec)) // 2
    mo_for_cas = _mo_for_cas(mo_coeff, active_space, ncore)

    cas = pyscf.mcscf.RCASCI(mf, ncas=ncas, nelecas=nelec)
    h1, ecore = cas.get_h1eff(mo_coeff=mo_for_cas)
    h2 = cas.get_h2eff(mo_coeff=mo_for_cas)

    return ffsim.MolecularData(
        norb=ncas,
        nelec=nelec,
        core_energy=float(ecore),
        one_body_integrals=h1,
        two_body_integrals=h2,
        ccsd_energy=float(np.real(ccsd.e_tot)) if ccsd is not None else None,
        ccsd_t1=None if ccsd is None else ccsd.t1,
        ccsd_t2=None if ccsd is None else ccsd.t2,
    )


class RCCSDContinuation:
    """Warm-start RCCSD from the previous geometry's amplitudes."""

    def __init__(
        self,
        *,
        active_space: Sequence[int] | None = None,
        conv_tol: float = 1e-10,
        conv_tol_normt: float = 1e-8,
        max_cycle: int = 200,
    ):
        self.active_space = (
            None if active_space is None else tuple(int(i) for i in active_space)
        )
        self.conv_tol = float(conv_tol)
        self.conv_tol_normt = float(conv_tol_normt)
        self.max_cycle = int(max_cycle)
        self.t1: np.ndarray | None = None
        self.t2: np.ndarray | None = None

    def reset(self):
        self.t1 = None
        self.t2 = None

    def run(self, mf):
        frozen = None
        if self.active_space is not None:
            frozen = frozen_orbitals_from_active_space(mf, self.active_space)
        cc = run_rccsd(
            mf,
            frozen=frozen,
            t1=self.t1,
            t2=self.t2,
            conv_tol=self.conv_tol,
            conv_tol_normt=self.conv_tol_normt,
            max_cycle=self.max_cycle,
        )
        self.t1 = np.array(cc.t1, copy=True)
        self.t2 = np.array(cc.t2, copy=True)
        return cc


class MolecularForwardContinuator:
    """Prepare a continuous forward molecular chart for compact scans."""

    def __init__(
        self,
        *,
        active_space: Sequence[int] | None = None,
        track_orbitals: bool = True,
        warm_start_scf: bool = True,
        run_ccsd: bool = True,
        rhf_conv_tol: float = 1e-12,
        rhf_max_cycle: int = 200,
        ccsd_conv_tol: float = 1e-10,
        ccsd_conv_tol_normt: float = 1e-8,
        ccsd_max_cycle: int = 200,
        rhf_init_guesses: Sequence[str] = ("atom", "minao", "hcore", "1e"),
        rhf_random_trials: int = 4,
    ):
        self.active_space = (
            None if active_space is None else tuple(int(i) for i in active_space)
        )
        self.track_orbitals = bool(track_orbitals)
        self.warm_start_scf = bool(warm_start_scf)
        self.run_ccsd = bool(run_ccsd)
        self.rhf_conv_tol = float(rhf_conv_tol)
        self.rhf_max_cycle = int(rhf_max_cycle)
        self.rhf_init_guesses = tuple(rhf_init_guesses)
        self.rhf_random_trials = int(rhf_random_trials)
        self.orbital_tracker = OrbitalTracker(enabled=track_orbitals)
        self.ccsd_continuation = RCCSDContinuation(
            active_space=self.active_space,
            conv_tol=ccsd_conv_tol,
            conv_tol_normt=ccsd_conv_tol_normt,
            max_cycle=ccsd_max_cycle,
        )
        self.prev_dm = None

    def reset(self):
        self.orbital_tracker.reset()
        self.ccsd_continuation.reset()
        self.prev_dm = None

    def prepare(self, mol) -> MolecularContinuationPoint:
        import ffsim

        if self.warm_start_scf and self.prev_dm is not None:
            mf = run_rhf(
                mol,
                dm0=self.prev_dm,
                conv_tol=self.rhf_conv_tol,
                max_cycle=self.rhf_max_cycle,
            )
        else:
            mf = run_lowest_rhf(
                mol,
                init_guesses=self.rhf_init_guesses,
                random_trials=self.rhf_random_trials,
                conv_tol=self.rhf_conv_tol,
                max_cycle=self.rhf_max_cycle,
            )
        self.prev_dm = mf.make_rdm1()

        total_nocc = (mol.nelectron + mol.spin) // 2
        mo_coeff, mo_diag = self.orbital_tracker.track(mol, mf.mo_coeff, total_nocc)

        ccsd = self.ccsd_continuation.run(mf) if self.run_ccsd else None
        mol_data = make_ffsim_molecular_data(
            mf,
            mo_coeff=mo_coeff,
            active_space=self.active_space,
            ccsd=ccsd,
        )
        operator = ffsim.linear_operator(
            mol_data.hamiltonian,
            norb=mol_data.norb,
            nelec=mol_data.nelec,
        )
        return MolecularContinuationPoint(
            mol=mol,
            mf=mf,
            ccsd=ccsd,
            mo_coeff=mo_coeff,
            mo_tracking=mo_diag,
            mol_data=mol_data,
            operator=operator,
            norb=mol_data.norb,
            nelec=mol_data.nelec,
        )


def dense_matrix_from_linear_operator(op: LinearOperator) -> np.ndarray:
    n, m = op.shape
    if n != m:
        raise ValueError("operator must be square")
    eye = np.eye(n, dtype=np.complex128)
    try:
        mat = op @ eye
    except Exception:
        mat = np.empty((n, n), dtype=np.complex128)
        for j in range(n):
            mat[:, j] = op @ eye[:, j]
    mat = np.asarray(mat, dtype=np.complex128)
    return 0.5 * (mat + mat.conj().T)


def exact_ground_energy(op: LinearOperator) -> float:
    return float(np.linalg.eigvalsh(dense_matrix_from_linear_operator(op))[0].real)


def projector_shifted_operator(
    operator: LinearOperator,
    target: np.ndarray,
    penalty: float,
) -> LinearOperator:
    target = np.asarray(target, dtype=np.complex128).reshape(-1)
    target = target / np.linalg.norm(target)

    def matvec(v):
        original_shape = np.asarray(v).shape
        x = np.asarray(v, dtype=np.complex128).reshape(-1)
        y = operator @ x - penalty * target * np.vdot(target, x)
        if len(original_shape) == 2:
            return y.reshape(-1, 1)
        return y

    def matmat(x):
        x = np.asarray(x, dtype=np.complex128)
        return operator @ x - penalty * np.outer(target, target.conj() @ x)

    return LinearOperator(
        shape=operator.shape,
        matvec=matvec,
        rmatvec=matvec,
        matmat=matmat,
        rmatmat=matmat,
        dtype=np.complex128,
    )


@dataclass
class GCR2OptimizationResult:
    energy: float
    objective: float
    params: np.ndarray
    state: np.ndarray
    scipy_result: OptimizeResult
    start_label: str


class GCR2ForwardOptimizer:
    """One-layer spin-restricted GCR2 optimizer with forward continuation starts."""

    def __init__(
        self,
        *,
        random_multistarts: int = 3,
        noise_scale: float = 1e-2,
        maxiter: int = 1000,
        ftol: float = 1e-16,
        gtol: float = 1e-6,
        homing_penalty: float | None = None,
        use_ccsd_seed: bool = False,
        n_reps: int = 1,
        rng_seed: int = 1234,
        start_selector: Callable[
            [list[tuple[str, np.ndarray]]], list[tuple[str, np.ndarray]]
        ]
        | None = None,
    ):
        self.random_multistarts = int(random_multistarts)
        self.noise_scale = float(noise_scale)
        self.maxiter = int(maxiter)
        self.ftol = float(ftol)
        self.gtol = float(gtol)
        self.homing_penalty = None if homing_penalty is None else float(homing_penalty)
        self.use_ccsd_seed = bool(use_ccsd_seed)
        self.n_reps = int(n_reps)
        self.rng = np.random.default_rng(rng_seed)
        self.start_selector = start_selector
        self.prev_params: np.ndarray | None = None
        self.prev_state: np.ndarray | None = None

    def reset(self):
        self.prev_params = None
        self.prev_state = None

    def _starts(
        self,
        parameterization,
        x_zero: np.ndarray,
        ccsd_t1=None,
        ccsd_t2=None,
    ) -> list[tuple[str, np.ndarray]]:
        starts: list[tuple[str, np.ndarray]] = []
        if self.prev_params is not None and self.prev_params.shape == x_zero.shape:
            starts.append(("prev_params", self.prev_params.copy()))

        x_seed = None
        if self.use_ccsd_seed and ccsd_t1 is not None and ccsd_t2 is not None:
            seed = UCJRestrictedProjectedDFSeed(
                t1=ccsd_t1,
                t2=ccsd_t2,
                n_reps=self.n_reps,
            ).build_ansatz()
            x_seed = parameterization.parameters_from_ucj_ansatz(seed)
            starts.append(("ccsd_seed", x_seed.copy()))

        starts.append(("zero", x_zero.copy()))
        noise_center = x_seed if x_seed is not None else x_zero
        for k in range(self.random_multistarts):
            noise = self.noise_scale * self.rng.normal(size=x_zero.shape)
            starts.append((f"noise_{k}", noise_center + noise))

        if self.start_selector is not None:
            starts = self.start_selector(starts)
        return starts

    def optimize(
        self,
        operator: LinearOperator,
        *,
        norb: int,
        nelec: tuple[int, int],
        ccsd_t1=None,
        ccsd_t2=None,
    ) -> GCR2OptimizationResult:
        import ffsim

        phi0 = ffsim.hartree_fock_state(norb, nelec)
        parameterization = IGCR2SpinRestrictedParameterization(norb=norb, nocc=nelec[0])
        params_to_vec = parameterization.params_to_vec(phi0, nelec)
        jac = make_restricted_gcr_jacobian(parameterization, phi0, nelec)
        x_zero = np.zeros(parameterization.n_params, dtype=np.float64)

        objective_operator = operator
        if (
            self.homing_penalty is not None
            and self.prev_state is not None
            and self.prev_state.shape == (operator.shape[0],)
        ):
            objective_operator = projector_shifted_operator(
                operator,
                self.prev_state,
                self.homing_penalty,
            )

        best = None
        for label, x0 in self._starts(parameterization, x_zero, ccsd_t1, ccsd_t2):
            try:
                result = minimize_linear_method(
                    params_to_vec,
                    objective_operator,
                    x0=np.asarray(x0, dtype=np.float64),
                    jac=jac,
                    ftol=self.ftol,
                    gtol=self.gtol,
                    maxiter=self.maxiter,
                )
                state = np.asarray(params_to_vec(result.x), dtype=np.complex128)
                energy = float(np.real(np.vdot(state, operator @ state)))
                objective = float(np.real(np.vdot(state, objective_operator @ state)))
            except (
                FloatingPointError,
                RuntimeError,
                ValueError,
                np.linalg.LinAlgError,
            ):
                continue
            if not np.isfinite(energy) or not np.isfinite(objective):
                continue
            candidate = GCR2OptimizationResult(
                energy=energy,
                objective=objective,
                params=np.array(result.x, copy=True),
                state=state,
                scipy_result=result,
                start_label=label,
            )
            if best is None or candidate.objective < best.objective:
                best = candidate

        if best is None:
            raise RuntimeError("No GCR2 optimization starts were run")

        self.prev_params = best.params.copy()
        self.prev_state = best.state.copy()
        return best
