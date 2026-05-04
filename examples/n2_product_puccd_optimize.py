import csv
import os

import numpy as np
import pyscf
import pyscf.ao2mo
import pyscf.cc
import pyscf.fci.direct_spin1
import pyscf.gto
import pyscf.lib
import pyscf.mcscf
import pyscf.scf
import scipy.optimize
from threadpoolctl import threadpool_limits

from xquces.gcr.product_pair_uccd import (
    ProductPairUCCDStateParameterization,
    product_pair_uccd_state_vjp,
)
from xquces.gcr.igcr import IGCR2SpinRestrictedParameterization
from xquces.gcr.pair_uccd_reference import GCR2ProductPairUCCDParameterization
from xquces.gcr.references import apply_ansatz_parameterization
from xquces.hamiltonians import MolecularHamiltonianLinearOperator
from xquces.states import hartree_fock_state
from xquces.ucj.init import UCJRestrictedProjectedDFSeed


START = 0.9
STOP = 3.5
STEP = 0.1
RS = np.linspace(START, STOP, round((STOP - START) / STEP) + 1)

BASIS = "sto-6g"
N_FROZEN = 2
N_REPS = 1
N_THREADS = 12
OUTPUT_FILE = "output/n2_product_puccd_sweep.csv"

PAIR_SCALE_BOUNDS = (-2.0, 2.0)

MAXITER = 1000
GTOL = 1e-8
FTOL = 1e-14
MAXLS = 100
LOG_EVERY_N = 25


class ElectronicHamiltonian:
    def __init__(self, ham):
        self.ham = ham

    def __matmul__(self, vec):
        return self.ham.matvec(vec)


def build_n2(r):
    mol = pyscf.gto.Mole()
    mol.build(
        atom=[("N", (-0.5 * r, 0.0, 0.0)), ("N", (0.5 * r, 0.0, 0.0))],
        basis=BASIS,
        symmetry="Dooh",
        verbose=0,
    )
    return mol


def build_point(r, prev_ccsd_t1=None, prev_ccsd_t2=None):
    mol = build_n2(r)

    scf = pyscf.scf.RHF(mol)
    scf.conv_tol = 1e-12
    scf.kernel()
    if not scf.converged:
        raise RuntimeError(f"RHF failed at R={r:.6f}")

    active_space = list(range(N_FROZEN, mol.nao_nr()))
    if not active_space:
        raise ValueError("No active orbitals remain after freezing")
    norb = len(active_space)
    nelectron = int(round(sum(scf.mo_occ[active_space])))
    n_alpha = (nelectron + mol.spin) // 2
    n_beta = (nelectron - mol.spin) // 2
    nelec = (n_alpha, n_beta)

    cas = pyscf.mcscf.RCASCI(scf, ncas=norb, nelecas=nelec)
    mo_coeff = cas.sort_mo(active_space, base=0)

    h1, ecore = cas.get_h1eff(mo_coeff=mo_coeff)
    h2 = cas.get_h2eff(mo_coeff=mo_coeff)
    eri = pyscf.ao2mo.restore(1, h2, norb)
    h2eff = pyscf.fci.direct_spin1.absorb_h1e(h1, eri, norb, nelec, 0.5)

    cas.fix_spin_(ss=0)
    cas.kernel(mo_coeff=mo_coeff)
    if not cas.converged:
        raise RuntimeError(f"FCI failed at R={r:.6f}")

    frozen = [i for i in range(mol.nao_nr()) if i not in active_space]
    ccsd = pyscf.cc.RCCSD(scf, frozen=frozen)
    ccsd.conv_tol = 1e-12
    ccsd.conv_tol_normt = 1e-10
    ccsd.max_cycle = 1000
    ccsd.kernel(t1=prev_ccsd_t1, t2=prev_ccsd_t2)
    if ccsd.t1 is None or ccsd.t2 is None:
        raise RuntimeError(f"RCCSD failed at R={r:.6f}")

    ucj_seed = UCJRestrictedProjectedDFSeed(
        t2=ccsd.t2,
        t1=ccsd.t1,
        n_reps=N_REPS,
    ).build_ansatz()

    mo_active = np.asarray(
        mo_coeff[:, cas.ncore : cas.ncore + cas.ncas],
        dtype=np.complex128,
    )

    ham = MolecularHamiltonianLinearOperator(
        h1=np.asarray(h1, dtype=np.float64),
        eri=np.asarray(eri, dtype=np.float64),
        ecore=float(ecore),
        norb=norb,
        nelec=nelec,
        h2eff=np.asarray(h2eff, dtype=np.float64),
    )

    return {
        "R": float(r),
        "mol": mol,
        "scf": scf,
        "cas": cas,
        "ccsd": ccsd,
        "ham": ham,
        "ucj_seed": ucj_seed,
        "mo_active": mo_active,
        "nocc": n_alpha,
    }


def active_orbital_overlap(prev_point, point):
    s_cross = pyscf.gto.intor_cross("int1e_ovlp", prev_point["mol"], point["mol"])
    return np.asarray(prev_point["mo_active"].conj().T @ s_cross @ point["mo_active"])


def transfer_parameters(prev_point, point, prev_param, param, prev_x):
    s_active = active_orbital_overlap(prev_point, point)
    return np.asarray(
        param.transfer_parameters_from(
            prev_x,
            previous_parameterization=prev_param,
            orbital_overlap=s_active,
        ),
        dtype=np.float64,
    )


def energy_and_grad(param, ham, x):
    x = np.asarray(x, dtype=np.float64)

    if hasattr(param, "energy_gradient_from_parameters"):
        e_elec, grad = param.energy_gradient_from_parameters(
            x,
            ElectronicHamiltonian(ham),
        )
        return float(e_elec) + ham.ecore, np.asarray(grad, dtype=np.float64)

    psi = param.state_from_parameters(x)
    hpsi = ham.matvec(psi)
    e_elec = float(np.vdot(psi, hpsi).real)

    if isinstance(param, ProductPairUCCDStateParameterization):
        residual = hpsi - e_elec * psi
        grad = product_pair_uccd_state_vjp(
            param.norb,
            param.nelec,
            x,
            residual,
        )
        return e_elec + ham.ecore, np.asarray(grad, dtype=np.float64)

    jac = param.state_jacobian_from_parameters(x)
    grad = 2.0 * np.real(jac.conj().T @ (hpsi - e_elec * psi))
    return e_elec + ham.ecore, np.asarray(grad, dtype=np.float64)


def make_gcr2_hf_parameterization(norb, nocc, nelec):
    igcr = IGCR2SpinRestrictedParameterization(norb=norb, nocc=nocc)
    ref = hartree_fock_state(norb, nelec)
    return apply_ansatz_parameterization(igcr, ref, nelec)


def make_gcr2_puccd_parameterization(norb, nocc):
    return GCR2ProductPairUCCDParameterization(norb=norb, nocc=nocc)


def gcr2_hf_seed(param, point):
    return np.asarray(
        param.ansatz_parameterization.parameters_from_ucj_ansatz(point["ucj_seed"]),
        dtype=np.float64,
    )


def gcr2_puccd_seed(param, point, puccd_x):
    ucj_seed = np.asarray(
        param.parameters_from_ucj_ansatz(point["ucj_seed"]),
        dtype=np.float64,
    )
    _, ansatz_x = param.split_parameters(ucj_seed)
    return np.concatenate([np.asarray(puccd_x, dtype=np.float64), ansatz_x])


def optimize_pair_scale(param, point):
    t2 = np.asarray(point["ccsd"].t2, dtype=np.float64)
    ham = point["ham"]

    def scaled_energy(scale):
        x = param.parameters_from_t2(t2, scale=scale)
        e, _ = energy_and_grad(param, ham, x)
        return e

    print("  [pair scale] optimizing first-geometry CCSD scale", flush=True)
    result = scipy.optimize.minimize_scalar(
        scaled_energy,
        bounds=PAIR_SCALE_BOUNDS,
        method="bounded",
        options={"xatol": 1e-6},
    )

    scale = float(result.x)
    x = param.parameters_from_t2(t2, scale=scale)
    e, g = energy_and_grad(param, ham, x)
    gmax = float(np.max(np.abs(g))) if g.size else float("nan")

    print(
        f"  [pair scale] scale={scale:.8f}  E={e:.12f}  gmax={gmax:.3e}",
        flush=True,
    )
    return scale, x


def optimize_one(param, ham, x_start, label):
    x_start = np.asarray(x_start, dtype=np.float64)
    last = {}

    def fun_and_grad(x):
        e, g = energy_and_grad(param, ham, x)
        last["e"] = e
        last["g"] = g
        return e, g

    nit = [0]

    def callback(_xk):
        nit[0] += 1
        if nit[0] % LOG_EVERY_N == 0:
            e = float(last.get("e", np.nan))
            g = np.asarray(last.get("g", np.zeros(0)))
            gmax = float(np.max(np.abs(g))) if g.size else float("nan")
            print(
                f"  [{label}] iter {nit[0]:4d}: E={e:.12f}  gmax={gmax:.3e}",
                flush=True,
            )

    e0, g0 = energy_and_grad(param, ham, x_start)
    gmax0 = float(np.max(np.abs(g0))) if g0.size else float("nan")

    print(
        f"  [{label}] seed: E={e0:.12f}  gmax={gmax0:.3e}  nparams={x_start.size}",
        flush=True,
    )

    result = scipy.optimize.minimize(
        fun_and_grad,
        x_start,
        jac=True,
        method="L-BFGS-B",
        callback=callback,
        options={
            "maxiter": MAXITER,
            "gtol": GTOL,
            "ftol": FTOL,
            "maxls": MAXLS,
        },
    )

    e, g = energy_and_grad(param, ham, result.x)
    gmax = float(np.max(np.abs(g))) if g.size else float("nan")

    print(
        f"  [{label}] done: E={e:.12f}  gmax={gmax:.3e}  "
        f"nit={int(result.nit)}  success={bool(result.success)}",
        flush=True,
    )

    return {
        "x": np.asarray(result.x, dtype=np.float64),
        "seed_energy": float(e0),
        "seed_gmax": float(gmax0),
        "energy": float(e),
        "gmax": gmax,
        "nit": int(result.nit),
        "success": bool(result.success),
    }


pyscf.lib.num_threads(N_THREADS)
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

columns = [
    "R",
    "E_FCI",
    "E_HF",
    "E_opt_pUCCD",
    "E_opt_GCR2_HF",
    "E_opt_GCR2_pUCCD",
]

prev_point = None
prev_puccd_x = None
prev_gcr2_hf_x = None
prev_gcr2_hf_param = None
prev_gcr2_puccd_x = None
prev_gcr2_puccd_param = None
prev_ccsd_t1 = None
prev_ccsd_t2 = None

with threadpool_limits(limits=N_THREADS):
    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        f.flush()
        os.fsync(f.fileno())
        print(",".join(columns), flush=True)

        for r in RS:
            point = build_point(
                float(r),
                prev_ccsd_t1=prev_ccsd_t1,
                prev_ccsd_t2=prev_ccsd_t2,
            )
            ham = point["ham"]
            nocc = point["nocc"]

            print(
                f"\n=== N2 R={point['R']:.2f}  norb={ham.norb}  "
                f"nelec={ham.nelec} ===",
                flush=True,
            )
            print(f"  HF   E={point['scf'].e_tot:.12f}", flush=True)
            print(f"  CCSD E={point['ccsd'].e_tot:.12f}", flush=True)
            print(f"  FCI  E={point['cas'].e_tot:.12f}", flush=True)

            puccd_param = ProductPairUCCDStateParameterization(ham.norb, ham.nelec)
            if prev_puccd_x is None:
                _, puccd_start = optimize_pair_scale(puccd_param, point)
                seed_label = "scaled_ccsd"
            else:
                puccd_start = np.asarray(prev_puccd_x, dtype=np.float64).copy()
                seed_label = "warm_start"

            puccd_result = optimize_one(
                puccd_param,
                ham,
                puccd_start,
                f"pUCCD/{seed_label}",
            )

            gcr2_hf_param = make_gcr2_hf_parameterization(ham.norb, nocc, ham.nelec)
            if prev_gcr2_hf_x is None:
                gcr2_hf_start = gcr2_hf_seed(gcr2_hf_param, point)
                seed_label = "ccsd_ucj"
            else:
                gcr2_hf_start = transfer_parameters(
                    prev_point,
                    point,
                    prev_gcr2_hf_param,
                    gcr2_hf_param,
                    prev_gcr2_hf_x,
                )
                seed_label = "warm_start"

            gcr2_hf_result = optimize_one(
                gcr2_hf_param,
                ham,
                gcr2_hf_start,
                f"GCR2-HF/{seed_label}",
            )

            gcr2_puccd_param = make_gcr2_puccd_parameterization(ham.norb, nocc)
            if prev_gcr2_puccd_x is None:
                gcr2_puccd_start = gcr2_puccd_seed(
                    gcr2_puccd_param,
                    point,
                    puccd_result["x"],
                )
                seed_label = "puccd_opt_ccsd_ucj"
            else:
                gcr2_puccd_start = transfer_parameters(
                    prev_point,
                    point,
                    prev_gcr2_puccd_param,
                    gcr2_puccd_param,
                    prev_gcr2_puccd_x,
                )
                seed_label = "warm_start"

            gcr2_puccd_result = optimize_one(
                gcr2_puccd_param,
                ham,
                gcr2_puccd_start,
                f"GCR2-pUCCD/{seed_label}",
            )

            row = [
                f"{point['R']:.6f}",
                f"{point['cas'].e_tot:.12f}",
                f"{point['scf'].e_tot:.12f}",
                f"{puccd_result['energy']:.12f}",
                f"{gcr2_hf_result['energy']:.12f}",
                f"{gcr2_puccd_result['energy']:.12f}",
            ]
            writer.writerow(row)
            f.flush()
            os.fsync(f.fileno())
            print(",".join(row), flush=True)

            prev_point = point
            prev_puccd_x = puccd_result["x"]
            prev_gcr2_hf_x = gcr2_hf_result["x"]
            prev_gcr2_hf_param = gcr2_hf_param
            prev_gcr2_puccd_x = gcr2_puccd_result["x"]
            prev_gcr2_puccd_param = gcr2_puccd_param
            prev_ccsd_t1 = np.asarray(point["ccsd"].t1)
            prev_ccsd_t2 = np.asarray(point["ccsd"].t2)

print(f"Wrote CSV: {OUTPUT_FILE}", flush=True)
