from __future__ import annotations

import numpy as np
import pyscf.gto
import scipy.optimize

import xquces as xq
from xquces import utils as xqut
from xquces.hamiltonians import MolecularHamiltonianLinearOperator


BASIS = "sto-3g"
ORDER = 2
RS = np.linspace(0.7, 3.5, 29)


def orbital_overlap(mf_old, mf_new):
    s_cross = pyscf.gto.intor_cross("int1e_ovlp", mf_old.mol, mf_new.mol)
    return np.asarray(mf_old.mo_coeff.conj().T @ s_cross @ mf_new.mo_coeff)


def energy_and_grad(param, ham: MolecularHamiltonianLinearOperator, x):
    psi = param.state_from_parameters(x)
    hpsi = ham.matvec(psi)
    e_elec = float(np.vdot(psi, hpsi).real)
    jac = param.state_jacobian_from_parameters(x)
    grad = 2.0 * np.real(jac.conj().T @ (hpsi - e_elec * psi))
    return e_elec + ham.ecore, grad


def optimize(param, ham, x0):
    return scipy.optimize.minimize(
        lambda x: energy_and_grad(param, ham, x),
        np.asarray(x0, dtype=np.float64),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 1000, "gtol": 1e-8, "ftol": 1e-12},
    )


def main():
    x0 = {"HF": None, "pUCCD": None}
    prev_param = {"HF": None, "pUCCD": None}
    prev_mf = None
    print("R,reference,energy,n_params,nit,grad_norm,success")

    for R in RS:
        mol = xqut.build_hydrogen_ring(R, 4, BASIS, symmetry=False)
        mf = xqut.run_lowest_rhf(mol)
        ham = MolecularHamiltonianLinearOperator.from_scf(mf)
        norb, nelec = ham.norb, ham.nelec
        nocc = nelec[0]

        igcr = xq.IGCRSpinRestrictedParameterization(norb=norb, nocc=nocc, order=ORDER)
        refs = {
            "HF": xq.hartree_fock_state(norb, nelec),
            "pUCCD": xq.PairUCCDStateParameterization(norb, nelec),
        }

        for name, ref in refs.items():
            param = igcr.apply(ref, nelec)
            if x0[name] is None:
                start = np.zeros(param.n_params)
            else:
                start = param.transfer_parameters_from(
                    x0[name],
                    previous_parameterization=prev_param[name],
                    orbital_overlap=orbital_overlap(prev_mf, mf),
                )
            res = optimize(param, ham, start)
            x0[name] = res.x
            prev_param[name] = param
            print(
                f"{R:.6f},{name},{res.fun:.12f},{param.n_params},"
                f"{res.nit},{np.linalg.norm(res.jac):.3e},{res.success}"
            )
        prev_mf = mf


if __name__ == "__main__":
    main()
