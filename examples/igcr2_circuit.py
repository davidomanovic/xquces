import ffsim
import ffsim.qiskit
import numpy as np
import pyscf
import pyscf.cc
import pyscf.gto
import pyscf.scf
from ffsim.qiskit.gates import PrepareHartreeFockJW
from ffsim.qiskit.gates.ucj import UCJOpSpinBalancedJW
from qiskit import QuantumCircuit

from xquces.igcr2 import IGCR2SpinBalancedParameterization
from xquces.qiskit import CircuitStatsJob, pretty_print_circuit_stats
from xquces.qiskit.gates.igcr2 import igcr2_stateprep_jw_circuit
from xquces.ucj.init import _ucj_ansatz_from_ffsim_stock



R = 1.1
basis = "cc-pvdz"
threads = 48
optimization_level = 3
transpile_seed = 12345


def build_ffsim_ucj_seed(t2, t1):
    return ffsim.UCJOpSpinBalanced.from_t_amplitudes(
        np.asarray(t2, dtype=np.float64),
        t1=np.asarray(t1, dtype=np.complex128),
        n_reps=1,
    )


def ffsim_ucj_stateprep_jw_circuit(ucj_op, nelec):
    circuit = QuantumCircuit(2 * ucj_op.norb)
    circuit.append(PrepareHartreeFockJW(ucj_op.norb, nelec), circuit.qubits)
    circuit.append(UCJOpSpinBalancedJW(ucj_op), circuit.qubits)
    return circuit


def build_n2_circuits(r, basis):
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

    ccsd = pyscf.cc.RCCSD(
        scf,
        frozen=[i for i in range(mol.nao_nr()) if i not in active_space],
    )
    ccsd.kernel()

    ffsim_ucj_seed = build_ffsim_ucj_seed(ccsd.t2, ccsd.t1)
    if ffsim_ucj_seed.final_orbital_rotation is None:
        raise RuntimeError("Expected ffsim UCJ seed to include a final orbital rotation.")

    ucj_seed = _ucj_ansatz_from_ffsim_stock(ffsim_ucj_seed)
    igcr2_param = IGCR2SpinBalancedParameterization(norb=norb, nocc=n_alpha)
    x_seed = igcr2_param.parameters_from_ucj_ansatz(ucj_seed)
    igcr2_ansatz = igcr2_param.ansatz_from_parameters(x_seed)

    return (
        ffsim_ucj_stateprep_jw_circuit(ffsim_ucj_seed, nelec),
        igcr2_stateprep_jw_circuit(igcr2_ansatz),
        norb,
        nelec,
        ffsim_ucj_seed.n_params() if callable(ffsim_ucj_seed.n_params) else ffsim_ucj_seed.n_params,
        igcr2_param.n_params,
    )

ffsim_circuit, igcr2_circuit, norb, nelec, ucj_n_params, igcr2_n_params = build_n2_circuits(
    r=R,
    basis=basis,
)

title = (
    f"N2/{basis} state-preparation circuit stats "
    f"(R={R:.3f} A, norb={norb}, nelec={nelec})"
)
print(f"parameters: ffsim UCJ = {ucj_n_params}, iGCR2 = {igcr2_n_params}\n")
pretty_print_circuit_stats(
    [
        CircuitStatsJob(
            "ffsim UCJ spin-balanced stateprep before PRE_INIT",
            ffsim_circuit,
        ),
        CircuitStatsJob(
            "ffsim UCJ spin-balanced stateprep after PRE_INIT",
            ffsim_circuit,
            pre_init=ffsim.qiskit.PRE_INIT,
        ),
        CircuitStatsJob("iGCR2 stateprep plain", igcr2_circuit),
    ],
    title=title,
    optimization_level=optimization_level,
    seed=transpile_seed,
)
