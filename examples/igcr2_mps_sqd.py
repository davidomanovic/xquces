import time
import warnings
from itertools import combinations
from functools import partial

import ffsim
import numpy as np
import pyscf.ao2mo
import pyscf.cc
import pyscf.gto
import pyscf.lib
import pyscf.mcscf
import pyscf.scf
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2 as AerSampler
from qiskit_addon_sqd.fermion import (
    diagonalize_fermionic_hamiltonian,
    solve_sci_batch,
)
from qiskit_addon_sqd.counts import bit_array_to_arrays

from xquces.igcr2 import IGCR2SpinBalancedParameterization
from xquces.qiskit.gates.igcr2 import igcr2_stateprep_jw_circuit
from xquces.ucj.init import UCJBalancedDFSeed


warnings.filterwarnings("ignore", category=RuntimeWarning)

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

shots = 100_000
samples_per_batch = 10_000
num_batches = 10
max_iterations = 10
max_dim = 2000
energy_tol = 1e-10
occupancies_tol = 1e-10
carryover_threshold = 1e-10
max_cycle = 1000
seed = 12345

use_initial_hf_occupancies = False
include_excitation_rank = 2
include_max_spin_strings = 800
print_sample_stats = True
print_sqd_iterations = True


pyscf.lib.num_threads(threads)

simulator = AerSimulator(
    method="matrix_product_state",
    matrix_product_state_max_bond_dimension=max_bond_dim,
    matrix_product_state_truncation_threshold=truncation,
    mps_omp_threads=threads,
    mps_parallel_threshold=threads,
    max_parallel_threads=threads,
)
sampler = AerSampler.from_backend(simulator)
sci_solver = partial(solve_sci_batch, spin_sq=0.0, max_cycle=max_cycle)


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


def build_active_space(scf):
    return list(range(n_frozen, scf.mol.nao_nr()))


def build_cas_hamiltonian(scf, active_space, nelec):
    norb = len(active_space)
    cas = pyscf.mcscf.RCASCI(scf, ncas=norb, nelecas=nelec)
    mo = cas.sort_mo(active_space, base=0)
    one_body, core_energy = cas.get_h1cas(mo)
    two_body = pyscf.ao2mo.restore(1, cas.get_h2cas(mo), norb)
    return one_body, two_body, core_energy


def build_initial_occupancies(norb, nelec):
    occ_alpha = np.zeros(norb, dtype=float)
    occ_beta = np.zeros(norb, dtype=float)
    occ_alpha[: nelec[0]] = 1.0
    occ_beta[: nelec[1]] = 1.0
    return occ_alpha, occ_beta


def spin_string(occupied):
    out = 0
    for p in occupied:
        out |= 1 << p
    return out


def excitation_spin_strings(norb, nocc, max_rank):
    occupied = tuple(range(nocc))
    virtual = tuple(range(nocc, norb))
    strings = [spin_string(occupied)]

    for rank in range(1, max_rank + 1):
        for holes in combinations(occupied, rank):
            remaining = [p for p in occupied if p not in holes]
            for particles in combinations(virtual, rank):
                strings.append(spin_string((*remaining, *particles)))

    return strings


def build_include_configurations(norb, nelec):
    max_rank = max(0, int(include_excitation_rank))
    max_strings = max(1, int(include_max_spin_strings))
    alpha = excitation_spin_strings(norb, nelec[0], max_rank)[:max_strings]
    beta = excitation_spin_strings(norb, nelec[1], max_rank)[:max_strings]
    return alpha, beta


def sample_stats(bit_array, norb, nelec):
    bitstrings, probabilities = bit_array_to_arrays(bit_array)
    valid_alpha = np.sum(bitstrings[:, norb:], axis=1) == nelec[0]
    valid_beta = np.sum(bitstrings[:, :norb], axis=1) == nelec[1]
    valid = np.logical_and(valid_alpha, valid_beta)
    valid_weight = float(np.sum(probabilities[valid]))
    return len(bitstrings), int(np.count_nonzero(valid)), valid_weight


def build_igcr2_seed_circuit(scf, active_space, previous_t1=None, previous_t2=None):
    mol_data = ffsim.MolecularData.from_scf(scf, active_space=active_space)
    norb = mol_data.norb
    nelec = mol_data.nelec

    frozen = [i for i in range(scf.mol.nao_nr()) if i not in active_space]
    ccsd = pyscf.cc.RCCSD(scf, frozen=frozen)
    ccsd.kernel(t1=previous_t1, t2=previous_t2)

    ucj_seed = UCJBalancedDFSeed(
        t2=ccsd.t2,
        t1=ccsd.t1,
        n_reps=n_reps,
    ).build_ansatz()
    igcr2_param = IGCR2SpinBalancedParameterization(
        norb=norb,
        nocc=nelec[0],
    )
    x_seed = igcr2_param.parameters_from_ucj_ansatz(ucj_seed)
    ansatz = igcr2_param.ansatz_from_parameters(x_seed)
    circuit = igcr2_stateprep_jw_circuit(ansatz)

    return circuit, norb, nelec, igcr2_param.n_params, ccsd.t1, ccsd.t2


def sample_circuit(circuit):
    measured = circuit.copy()
    measured.measure_all()
    transpiled = transpile(measured, backend=simulator, optimization_level=3)
    job = sampler.run([(transpiled, [])], shots=shots)
    return job.result()[0].data.meas


def run_sqd(bit_array, one_body, two_body, core_energy, norb, nelec):
    include_configurations = build_include_configurations(norb, nelec)
    initial_occupancies = None
    if use_initial_hf_occupancies:
        initial_occupancies = build_initial_occupancies(norb, nelec)

    if print_sqd_iterations:
        print(
            "# SQD include spin strings: "
            f"alpha={len(include_configurations[0])}, "
            f"beta={len(include_configurations[1])}, "
            f"rank<={include_excitation_rank}",
            flush=True,
        )

    result_history = []

    def callback(results):
        best = min(results, key=lambda res: res.energy)
        result_history.append(best)
        if print_sqd_iterations:
            dim_alpha = len(best.sci_state.ci_strs_a)
            dim_beta = len(best.sci_state.ci_strs_b)
            print(
                f"# SQD iter={len(result_history):02d} "
                f"E={best.energy + core_energy:.12f} "
                f"dim={dim_alpha}x{dim_beta}",
                flush=True,
            )

    result = diagonalize_fermionic_hamiltonian(
        one_body,
        two_body,
        bit_array,
        samples_per_batch=samples_per_batch,
        norb=norb,
        nelec=nelec,
        num_batches=num_batches,
        energy_tol=energy_tol,
        occupancies_tol=occupancies_tol,
        max_iterations=max_iterations,
        sci_solver=sci_solver,
        symmetrize_spin=True,
        max_dim=max_dim,
        include_configurations=include_configurations,
        initial_occupancies=initial_occupancies,
        carryover_threshold=carryover_threshold,
        callback=callback,
        seed=seed,
    )
    return float(result.energy + core_energy)


bond_distance_range = np.linspace(
    start,
    stop,
    num=round((stop - start) / step) + 1,
)

print(
    f"{molecule} in {basis}. chi={max_bond_dim}, trunc={truncation}, shots={shots}",
    flush=True,
)
print("R,E_iGCR2_SQD,norb,nelec,n_params,seconds", flush=True)

previous_t1 = None
previous_t2 = None

for R in bond_distance_range:
    t0 = time.perf_counter()
    scf = build_n2_scf(R)
    active_space = build_active_space(scf)

    circuit, norb, nelec, n_params, previous_t1, previous_t2 = build_igcr2_seed_circuit(
        scf,
        active_space,
        previous_t1=previous_t1,
        previous_t2=previous_t2,
    )
    one_body, two_body, core_energy = build_cas_hamiltonian(scf, active_space, nelec)

    bit_array = sample_circuit(circuit)
    if print_sample_stats:
        unique, valid_unique, valid_weight = sample_stats(bit_array, norb, nelec)
        print(
            f"# samples: unique={unique}, "
            f"valid_unique={valid_unique}, "
            f"valid_weight={valid_weight:.6f}",
            flush=True,
        )

    energy = run_sqd(bit_array, one_body, two_body, core_energy, norb, nelec)
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
