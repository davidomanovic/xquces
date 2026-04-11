from __future__ import annotations

import numpy as np
import ffsim
import ffsim.qiskit
import pyscf
import pyscf.cc
import pyscf.gto
import pyscf.scf
from qiskit import QuantumCircuit
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from ffsim.qiskit.gates import PrepareHartreeFockJW, PrepareSlaterDeterminantJW
from ffsim.qiskit.gates.ucj import UCJOpSpinBalancedJW
from qiskit.quantum_info import Statevector

import xquces.qiskit as xq_qiskit
from xquces.basis import occ_rows, reshape_state
from xquces.hamiltonians import MolecularHamiltonianLinearOperator
from xquces.igcr2 import IGCR2SpinBalancedParameterization
from xquces.qiskit.gates.diag_2 import Diag2SpinBalancedJW
from xquces.qiskit.gates.igcr2 import igcr2_jw_circuit, spin_balanced_rzz_circuit_gauge
from xquces.qiskit.gates.orbital_rotations import OrbitalRotationJW
from xquces.states import hartree_fock_state
from xquces.ucj.init import _ucj_ansatz_from_ffsim_stock


start, stop, step = 0.9, 2.4, 0.1
bond_distance_range = np.linspace(start, stop, num=round((stop - start) / step) + 1)
molecule = "N2"
basis = "sto-6g"
state_mismatch_atol = 1e-7
native_basis_gates = ["rz", "sx", "x", "cx"]
transpiler_optimization_level = 3
transpiler_seed = 12345

pyscf.lib.num_threads(48)


def bitstring_index(occ_alpha, occ_beta, norb: int) -> int:
    alpha_bits = sum(1 << int(p) for p in occ_alpha)
    beta_bits = sum(1 << (norb + int(p)) for p in occ_beta)
    return alpha_bits + beta_bits


def sector_to_jw_state(vec: np.ndarray, norb: int, nelec: tuple[int, int]) -> np.ndarray:
    out = np.zeros(2 ** (2 * norb), dtype=np.complex128)
    mat = reshape_state(vec, norb, nelec)
    occ_alpha = occ_rows(norb, nelec[0])
    occ_beta = occ_rows(norb, nelec[1])

    for i_alpha, alpha in enumerate(occ_alpha):
        for i_beta, beta in enumerate(occ_beta):
            out[bitstring_index(alpha, beta, norb)] = mat[i_alpha, i_beta]

    return out


def jw_state_to_sector(vec: np.ndarray, norb: int, nelec: tuple[int, int]) -> np.ndarray:
    occ_alpha = occ_rows(norb, nelec[0])
    occ_beta = occ_rows(norb, nelec[1])
    out = np.zeros((len(occ_alpha), len(occ_beta)), dtype=np.complex128)

    for i_alpha, alpha in enumerate(occ_alpha):
        for i_beta, beta in enumerate(occ_beta):
            out[i_alpha, i_beta] = vec[bitstring_index(alpha, beta, norb)]

    return out.reshape(-1)


def phase_aligned_diff(psi: np.ndarray, phi: np.ndarray) -> float:
    overlap = np.vdot(phi, psi)
    if abs(overlap) < 1e-14:
        return float(np.linalg.norm(psi - phi))
    return float(np.linalg.norm(psi - overlap / abs(overlap) * phi))


def apply_jw_circuit(
    circuit: QuantumCircuit,
    reference: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
) -> np.ndarray:
    full_reference = sector_to_jw_state(reference, norb, nelec)
    full_out = Statevector(full_reference).evolve(circuit).data
    return jw_state_to_sector(full_out, norb, nelec)


def apply_stateprep_jw_circuit(
    circuit: QuantumCircuit,
    norb: int,
    nelec: tuple[int, int],
) -> np.ndarray:
    full_reference = np.zeros(2 ** (2 * norb), dtype=np.complex128)
    full_reference[0] = 1.0
    full_out = Statevector(full_reference).evolve(circuit).data
    return jw_state_to_sector(full_out, norb, nelec)


def ffsim_ucj_jw_circuit(ucj_op: ffsim.UCJOpSpinBalanced) -> QuantumCircuit:
    circuit = QuantumCircuit(2 * ucj_op.norb)
    circuit.append(UCJOpSpinBalancedJW(ucj_op), circuit.qubits)
    return circuit


def ffsim_ucj_stateprep_jw_circuit(
    ucj_op: ffsim.UCJOpSpinBalanced,
    nelec: tuple[int, int],
) -> QuantumCircuit:
    circuit = QuantumCircuit(2 * ucj_op.norb)
    circuit.append(PrepareHartreeFockJW(ucj_op.norb, nelec), circuit.qubits)
    circuit.append(UCJOpSpinBalancedJW(ucj_op), circuit.qubits)
    return circuit


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


def native_backend(n_qubits: int) -> GenericBackendV2:
    coupling_map = [
        [control, target]
        for control in range(n_qubits)
        for target in range(n_qubits)
        if control != target
    ]
    return GenericBackendV2(
        n_qubits,
        basis_gates=native_basis_gates,
        coupling_map=coupling_map,
        seed=transpiler_seed,
    )


def transpile_to_native(circuit: QuantumCircuit, pre_init: PassManager) -> QuantumCircuit:
    pass_manager = generate_preset_pass_manager(
        optimization_level=transpiler_optimization_level,
        backend=native_backend(circuit.num_qubits),
        initial_layout=list(range(circuit.num_qubits)),
    )
    pass_manager.pre_init = pre_init
    return pass_manager.run(circuit)


def total_gate_count(circuit: QuantumCircuit) -> int:
    return sum(circuit.count_ops().values())


def format_count_ops(circuit: QuantumCircuit) -> str:
    return ";".join(
        f"{gate}:{count}" for gate, count in sorted(circuit.count_ops().items())
    )


def build_ffsim_ucj_seed(t2: np.ndarray, t1: np.ndarray) -> ffsim.UCJOpSpinBalanced:
    return ffsim.UCJOpSpinBalanced.from_t_amplitudes(
        np.asarray(t2, dtype=np.float64),
        t1=np.asarray(t1, dtype=np.complex128),
        n_reps=1,
        interaction_pairs=None,
        tol=1e-8,
        optimize=False,
        method="L-BFGS-B",
        callback=None,
        options=None,
        regularization=0.0,
        multi_stage_start=None,
        multi_stage_step=None,
    )


def main() -> None:
    print(
        "R,"
        "E_iGCR2_circuit,"
        "E_iGCR2_statevectorsimulator,"
        "E_ffsim_UCJ_spin_balanced_circuit,"
        "native_depth_iGCR2_stateprep_circuit,"
        "native_depth_ffsim_UCJ_spin_balanced_stateprep_circuit_with_final_orbital_rotation,"
        "native_gate_count_iGCR2_stateprep_circuit,"
        "native_gate_count_ffsim_UCJ_spin_balanced_stateprep_circuit_with_final_orbital_rotation,"
        "native_ops_iGCR2_stateprep_circuit,"
        "native_ops_ffsim_UCJ_spin_balanced_stateprep_circuit_with_final_orbital_rotation",
        flush=True,
    )

    prev_ccsd_t1 = None
    prev_ccsd_t2 = None

    for R in bond_distance_range:
        mol = pyscf.gto.Mole()
        mol.build(
            atom=[("N", (-0.5 * R, 0, 0)), ("N", (0.5 * R, 0, 0))],
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
        ccsd.kernel(t1=prev_ccsd_t1, t2=prev_ccsd_t2)
        prev_ccsd_t1 = np.array(ccsd.t1, copy=True)
        prev_ccsd_t2 = np.array(ccsd.t2, copy=True)

        ham_xq = MolecularHamiltonianLinearOperator.from_scf(scf, active_space=active_space)
        reference = hartree_fock_state(norb, nelec)

        ffsim_ucj_seed = build_ffsim_ucj_seed(ccsd.t2, ccsd.t1)
        if ffsim_ucj_seed.final_orbital_rotation is None:
            raise RuntimeError("Expected ffsim UCJ seed to include a final orbital rotation.")

        ucj_seed = _ucj_ansatz_from_ffsim_stock(ffsim_ucj_seed)
        igcr2_param = IGCR2SpinBalancedParameterization(norb=norb, nocc=n_alpha)
        x_seed = igcr2_param.parameters_from_ucj_ansatz(ucj_seed)
        ansatz = igcr2_param.ansatz_from_parameters(x_seed)

        igcr2_circuit = igcr2_jw_circuit(ansatz)
        igcr2_stateprep_circuit = igcr2_stateprep_jw_circuit(ansatz)
        ffsim_ucj_circuit = ffsim_ucj_jw_circuit(ffsim_ucj_seed)
        ffsim_ucj_stateprep_circuit = ffsim_ucj_stateprep_jw_circuit(ffsim_ucj_seed, nelec)

        psi_circuit = apply_jw_circuit(igcr2_circuit, reference, norb, nelec)
        psi_stateprep_circuit = apply_stateprep_jw_circuit(
            igcr2_stateprep_circuit,
            norb,
            nelec,
        )
        psi_ffsim_ucj_circuit = apply_jw_circuit(ffsim_ucj_circuit, reference, norb, nelec)
        psi_ffsim_ucj_stateprep_circuit = apply_stateprep_jw_circuit(
            ffsim_ucj_stateprep_circuit,
            norb,
            nelec,
        )
        psi_statevector = ansatz.apply(reference, nelec=nelec, copy=True)

        mismatch = phase_aligned_diff(psi_circuit, psi_statevector)
        if mismatch > state_mismatch_atol:
            raise RuntimeError(
                f"iGCR2 circuit/statevector mismatch at R={R:.6f}: {mismatch:.3e}"
            )

        stateprep_mismatch = phase_aligned_diff(psi_stateprep_circuit, psi_statevector)
        if stateprep_mismatch > state_mismatch_atol:
            raise RuntimeError(
                f"iGCR2 stateprep circuit/statevector mismatch at R={R:.6f}: "
                f"{stateprep_mismatch:.3e}"
            )

        ucj_mismatch = phase_aligned_diff(psi_ffsim_ucj_circuit, psi_statevector)
        if ucj_mismatch > state_mismatch_atol:
            raise RuntimeError(
                f"ffsim UCJ circuit/iGCR2 seed mismatch at R={R:.6f}: {ucj_mismatch:.3e}"
            )

        ucj_stateprep_mismatch = phase_aligned_diff(
            psi_ffsim_ucj_stateprep_circuit,
            psi_statevector,
        )
        if ucj_stateprep_mismatch > state_mismatch_atol:
            raise RuntimeError(
                f"ffsim UCJ stateprep circuit/iGCR2 seed mismatch at R={R:.6f}: "
                f"{ucj_stateprep_mismatch:.3e}"
            )

        energy_circuit = ham_xq.expectation(psi_circuit)
        energy_statevector = ham_xq.expectation(psi_statevector)
        energy_ffsim_ucj_circuit = ham_xq.expectation(psi_ffsim_ucj_circuit)

        igcr2_native = transpile_to_native(igcr2_stateprep_circuit, xq_qiskit.PRE_INIT)
        ffsim_ucj_native = transpile_to_native(
            ffsim_ucj_stateprep_circuit,
            ffsim.qiskit.PRE_INIT,
        )

        print(
            f"{R:.6f},"
            f"{energy_circuit:.12f},"
            f"{energy_statevector:.12f},"
            f"{energy_ffsim_ucj_circuit:.12f},"
            f"{igcr2_native.depth()},"
            f"{ffsim_ucj_native.depth()},"
            f"{total_gate_count(igcr2_native)},"
            f"{total_gate_count(ffsim_ucj_native)},"
            f"{format_count_ops(igcr2_native)},"
            f"{format_count_ops(ffsim_ucj_native)}",
            flush=True,
        )


if __name__ == "__main__":
    main()
