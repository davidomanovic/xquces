from __future__ import annotations

import numpy as np
import scipy.optimize

from xquces.hamiltonians import MolecularHamiltonianLinearOperator
from xquces.ucj.params import ansatz_from_parameters_spin_restricted


def optimize_ucj(
    x0: np.ndarray,
    *,
    hamiltonian: MolecularHamiltonianLinearOperator,
    reference_vec: np.ndarray,
    nocc: int,
    n_layers: int,
    method: str = "Powell",
    options: dict | None = None,
):
    x0 = np.asarray(x0, dtype=np.float64)
    norb = hamiltonian.norb
    nelec = hamiltonian.nelec

    def objective(x):
        ansatz = ansatz_from_parameters_spin_restricted(
            x,
            norb=norb,
            nocc=nocc,
            n_layers=n_layers,
        )
        vec = ansatz.apply(reference_vec, nelec=nelec, copy=True)
        return hamiltonian.expectation(vec)

    res = scipy.optimize.minimize(
        objective,
        x0,
        method=method,
        options={} if options is None else options,
    )

    ansatz_opt = ansatz_from_parameters_spin_restricted(
        res.x,
        norb=norb,
        nocc=nocc,
        n_layers=n_layers,
    )
    vec_opt = ansatz_opt.apply(reference_vec, nelec=nelec, copy=True)
    e_opt = hamiltonian.expectation(vec_opt)
    return ansatz_opt, vec_opt, e_opt, res