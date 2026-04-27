from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import sys
from typing import TextIO

from qiskit import QuantumCircuit
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.transpiler import PassManager
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager


DEFAULT_NATIVE_BASIS_GATES = ("rz", "sx", "x", "cx")
DEFAULT_TRANSPILE_SEED = 12345


@dataclass(frozen=True)
class CircuitStatsJob:
    """A circuit-statistics job with an optional ``pre_init`` pass manager."""

    label: str
    circuit: QuantumCircuit
    pre_init: PassManager | None = None


@dataclass(frozen=True)
class CircuitStats:
    """Basic native-circuit statistics."""

    label: str
    num_qubits: int
    depth: int
    gate_count: int
    two_qubit_gate_count: int
    count_ops: dict[str, int]


def native_backend(
    n_qubits: int,
    *,
    basis_gates: Sequence[str] = DEFAULT_NATIVE_BASIS_GATES,
    seed: int = DEFAULT_TRANSPILE_SEED,
) -> GenericBackendV2:
    """Build a fully connected fake backend with the requested native gates."""
    coupling_map = [
        [control, target]
        for control in range(n_qubits)
        for target in range(n_qubits)
        if control != target
    ]
    return GenericBackendV2(
        n_qubits,
        basis_gates=list(basis_gates),
        coupling_map=coupling_map,
        seed=seed,
    )


def transpile_to_native(
    circuit: QuantumCircuit,
    *,
    pre_init: PassManager | None = None,
    optimization_level: int = 3,
    basis_gates: Sequence[str] = DEFAULT_NATIVE_BASIS_GATES,
    seed: int = DEFAULT_TRANSPILE_SEED,
) -> QuantumCircuit:
    """Transpile a circuit to a simple all-to-all native-gate model."""
    pass_manager = generate_preset_pass_manager(
        optimization_level=optimization_level,
        backend=native_backend(
            circuit.num_qubits,
            basis_gates=basis_gates,
            seed=seed,
        ),
        initial_layout=list(range(circuit.num_qubits)),
    )
    if pre_init is not None:
        pass_manager.pre_init = pre_init
    return pass_manager.run(circuit)


def total_gate_count(circuit: QuantumCircuit) -> int:
    """Return the total number of operations in a circuit."""
    return sum(circuit.count_ops().values())


def two_qubit_gate_count(circuit: QuantumCircuit) -> int:
    """Return the number of two-qubit operations in a circuit."""
    return sum(1 for instruction in circuit.data if len(instruction.qubits) == 2)


def circuit_stats(circuit: QuantumCircuit, label: str) -> CircuitStats:
    """Collect basic statistics from an already-built circuit."""
    return CircuitStats(
        label=label,
        num_qubits=circuit.num_qubits,
        depth=circuit.depth(),
        gate_count=total_gate_count(circuit),
        two_qubit_gate_count=two_qubit_gate_count(circuit),
        count_ops=dict(circuit.count_ops()),
    )


def format_count_ops(count_ops: dict[str, int]) -> str:
    """Format operation counts in a compact, deterministic order."""
    return ", ".join(f"{gate}: {count}" for gate, count in sorted(count_ops.items()))


def pretty_print_circuit_stats(
    jobs: Sequence[CircuitStatsJob],
    *,
    title: str | None = None,
    optimization_level: int = 3,
    basis_gates: Sequence[str] = DEFAULT_NATIVE_BASIS_GATES,
    seed: int = DEFAULT_TRANSPILE_SEED,
    file: TextIO | None = None,
) -> list[CircuitStats]:
    """Transpile circuits to native gates and print readable statistics.

    Returns the collected statistics so callers can also assert on them in
    tests or notebooks.
    """
    out = sys.stdout if file is None else file
    if title:
        print(title, file=out)
        print("=" * len(title), file=out)

    stats: list[CircuitStats] = []
    for index, job in enumerate(jobs):
        native = transpile_to_native(
            job.circuit,
            pre_init=job.pre_init,
            optimization_level=optimization_level,
            basis_gates=basis_gates,
            seed=seed,
        )
        item = circuit_stats(native, job.label)
        stats.append(item)

        if index:
            print(file=out)
        print(job.label, file=out)
        print("-" * len(job.label), file=out)
        print(f"qubits:          {item.num_qubits}", file=out)
        print(f"depth:           {item.depth}", file=out)
        print(f"gates:           {item.gate_count}", file=out)
        print(f"two-qubit gates: {item.two_qubit_gate_count}", file=out)
        print(f"ops:             {format_count_ops(item.count_ops)}", file=out)

    return stats
