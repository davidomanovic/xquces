import time

import ffsim
import numpy as np
import pyscf.cc
import pyscf.gto
import pyscf.scf
from ffsim.qiskit import jordan_wigner
from qiskit import transpile
from qiskit_aer import AerSimulator

from xquces.gcr.igcr2 import IGCR2SpinBalancedParameterization
from xquces.qiskit.gates.igcr2 import igcr2_stateprep_jw_circuit
from xquces.ucj.init import _ucj_ansatz_from_ffsim_stock


start = 0.9
stop = 2.4
step = 0.1

molecule = "N2"
basis = "cc-pvdz"
n_frozen = 2
n_reps = 1

max_bond_dim = 512
truncation = 1e-10
threads = 48

# Keep this as None for the full iGCR2 seed.  Sparse ffsim mixed-spin
# interaction-pair lists can include diagonal pairs, which are represented
# separately in iGCR2 and should not be passed straight through as pair lists.
interaction_pairs = None


simulator = AerSimulator(
    method="matrix_product_state",
    matrix_product_state_max_bond_dimension=max_bond_dim,
    matrix_product_state_truncation_threshold=truncation,
    mps_omp_threads=threads,
    mps_parallel_threshold=threads,
    max_parallel_threads=threads,
)


def energy_of_circuit(simulator, circuit, hamiltonian, n_qubits, label):
    circuit = circuit.copy()
    circuit.save_expectation_value(hamiltonian, range(n_qubits), label=label)
    transpiled = transpile(circuit, backend=simulator, optimization_level=3)
    result = simulator.run(transpiled, shots=1).result()
    return float(result.data(0)[label].real)


def build_n2_scf(r):
    mol = pyscf.gto.Mole()
    mol.build(
        atom=[("N", (-0.5 * r, 0, 0)), ("N", (0.5 * r, 0, 0))],
        basis=basis,
        symmetry="Dooh",
        unit="Angstrom",
        verbose=0,
    )
    return pyscf.scf.RHF(mol).run()


def build_igcr2_seed_circuit(scf, active_space):
    mol_data = ffsim.MolecularData.from_scf(scf, active_space=active_space)
    norb = mol_data.norb
    nelec = mol_data.nelec

    frozen = [i for i in range(scf.mol.nao_nr()) if i not in active_space]
    ccsd = pyscf.cc.RCCSD(scf, frozen=frozen).run()

    ffsim_ucj_seed = ffsim.UCJOpSpinBalanced.from_t_amplitudes(
        t2=np.asarray(ccsd.t2, dtype=np.float64),
        t1=np.asarray(ccsd.t1, dtype=np.complex128),
        n_reps=n_reps,
        interaction_pairs=interaction_pairs,
    )
    ucj_seed = _ucj_ansatz_from_ffsim_stock(ffsim_ucj_seed)

    same_pairs = None
    mixed_pairs = None
    if interaction_pairs is not None:
        same_pairs, mixed_pairs = interaction_pairs
        mixed_pairs = [(p, q) for p, q in mixed_pairs if p != q]

    parameterization = IGCR2SpinBalancedParameterization(
        norb=norb,
        nocc=nelec[0],
        same_spin_interaction_pairs=same_pairs,
        mixed_spin_interaction_pairs=mixed_pairs,
    )
    x_seed = parameterization.parameters_from_ucj_ansatz(ucj_seed)
    ansatz = parameterization.ansatz_from_parameters(x_seed)
    circuit = igcr2_stateprep_jw_circuit(ansatz)

    return circuit, norb, nelec, parameterization.n_params


bond_distance_range = np.linspace(
    start,
    stop,
    num=round((stop - start) / step) + 1,
)

print("R,E_iGCR2_MPS,norb,nelec,n_params,seconds", flush=True)

for R in bond_distance_range:
    t0 = time.perf_counter()
    scf = build_n2_scf(R)
    active_space = range(n_frozen, scf.mol.nao_nr())

    mol_data = ffsim.MolecularData.from_scf(scf, active_space=active_space)
    norb = mol_data.norb
    nelec = mol_data.nelec
    n_qubits = 2 * norb

    fermion_hamiltonian = ffsim.fermion_operator(mol_data.hamiltonian)
    qubit_hamiltonian = jordan_wigner(fermion_hamiltonian, norb=norb)

    circuit, norb, nelec, n_params = build_igcr2_seed_circuit(scf, active_space)
    energy = energy_of_circuit(
        simulator,
        circuit,
        qubit_hamiltonian,
        n_qubits,
        "E_iGCR2",
    )
    seconds = time.perf_counter() - t0

    print(
        f"{R:.10f},"
        f"{energy:.12f},"
        f"{norb},"
        f"{nelec},"
        f"{n_params},"
        f"{seconds:.3f}",
        flush=True,
    )
