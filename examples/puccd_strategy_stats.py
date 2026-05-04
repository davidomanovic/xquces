from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyscf.cc
import pyscf.gto
import pyscf.lib
import pyscf.scf
from qiskit import QuantumCircuit
from qiskit.transpiler import PassManager
from threadpoolctl import threadpool_limits

from xquces.gcr.product_pair_uccd import ProductPairUCCDStateParameterization
from xquces.qiskit import (
    PRE_INIT as XQUCES_PRE_INIT,
    CircuitStats,
    circuit_stats,
    transpile_to_native,
)
from xquces.qiskit.gates import product_pair_uccd_stateprep_jw_circuit
from xquces.qiskit.utils import DEFAULT_NATIVE_BASIS_GATES, DEFAULT_TRANSPILE_SEED


BASIS = "sto-6g"
N_FROZEN = 2
N2_R = 1.1
NATIVE_BASIS_GATES = ("cx", "rz", "sx", "x")
METHODS = ("pUCCD pair-register", "pUCCD spin-orbital")
STRATEGIES = {
    "pUCCD pair-register": "pair_register",
    "pUCCD spin-orbital": "spin_orbital",
}
METHOD_COLORS = {
    "pUCCD pair-register": "#ff7f0e",
    "pUCCD spin-orbital": "#2ca02c",
}
GATE_COLORS = {
    "cx": "#1f77b4",
    "rz": "#ff7f0e",
    "sx": "#2ca02c",
    "x": "#9467bd",
}


@dataclass(frozen=True)
class SystemData:
    label: str
    norb: int
    nelec: tuple[int, int]
    n_frozen: int
    basis: str
    bond_length: float
    pair_params: np.ndarray


@dataclass(frozen=True)
class StrategyResult:
    method: str
    n_params: int
    raw_stats: CircuitStats
    native_stats: CircuitStats


def build_n2(r: float, basis: str) -> pyscf.gto.Mole:
    mol = pyscf.gto.Mole()
    mol.build(
        atom=[("N", (-0.5 * r, 0.0, 0.0)), ("N", (0.5 * r, 0.0, 0.0))],
        basis=basis,
        symmetry="Dooh",
        verbose=0,
    )
    return mol


def build_system_data(
    *,
    r: float,
    basis: str,
    n_frozen: int,
) -> SystemData:
    mol = build_n2(r, basis)
    scf = pyscf.scf.RHF(mol)
    scf.conv_tol = 1e-12
    scf.kernel()
    if not scf.converged:
        raise RuntimeError("RHF failed for N2")

    active_space = list(range(n_frozen, mol.nao_nr()))
    if not active_space:
        raise ValueError("No active orbitals remain after freezing")
    norb = len(active_space)
    nelectron_active = int(round(sum(scf.mo_occ[active_space])))
    n_alpha = (nelectron_active + mol.spin) // 2
    n_beta = (nelectron_active - mol.spin) // 2
    nelec = (n_alpha, n_beta)
    frozen = [i for i in range(mol.nao_nr()) if i not in active_space]

    ccsd = pyscf.cc.RCCSD(scf, frozen=frozen)
    ccsd.conv_tol = 1e-12
    ccsd.conv_tol_normt = 1e-10
    ccsd.max_cycle = 1000
    ccsd.kernel()
    if ccsd.t2 is None:
        raise RuntimeError("RCCSD failed for N2")

    pair_param = ProductPairUCCDStateParameterization(norb, nelec)
    pair_params = pair_param.parameters_from_t2(np.asarray(ccsd.t2), scale=0.5)

    return SystemData(
        label=f"N2 {basis} nf={n_frozen}",
        norb=norb,
        nelec=nelec,
        n_frozen=n_frozen,
        basis=basis,
        bond_length=float(r),
        pair_params=np.asarray(pair_params, dtype=np.float64),
    )


def native_stats(
    circuit: QuantumCircuit,
    label: str,
    *,
    pre_init: PassManager,
    optimization_level: int,
    basis_gates: tuple[str, ...],
    seed: int,
) -> CircuitStats:
    native = transpile_to_native(
        circuit,
        pre_init=pre_init,
        optimization_level=optimization_level,
        basis_gates=basis_gates,
        seed=seed,
    )
    return circuit_stats(native, label)


def build_strategy_results(
    system: SystemData,
    *,
    optimization_level: int,
    basis_gates: tuple[str, ...],
    seed: int,
) -> dict[str, StrategyResult]:
    results: dict[str, StrategyResult] = {}
    for method in METHODS:
        strategy = STRATEGIES[method]
        circuit = product_pair_uccd_stateprep_jw_circuit(
            system.norb,
            system.nelec,
            system.pair_params,
            strategy=strategy,
        )
        results[method] = StrategyResult(
            method=method,
            n_params=system.pair_params.size,
            raw_stats=circuit_stats(circuit, f"{method} raw"),
            native_stats=native_stats(
                circuit,
                method,
                pre_init=XQUCES_PRE_INIT,
                optimization_level=optimization_level,
                basis_gates=basis_gates,
                seed=seed,
            ),
        )
    return results


def print_summary(system: SystemData, results: dict[str, StrategyResult]) -> None:
    columns = [
        "system",
        "method",
        "basis",
        "R",
        "n_frozen",
        "norb",
        "nelec",
        "params",
        "raw_depth",
        "raw_total",
        "raw_two_qubit",
        "native_depth",
        "native_total",
        "native_two_qubit",
        "native_ops",
    ]
    print(",".join(columns))
    for method in METHODS:
        result = results[method]
        ops = ";".join(
            f"{gate}:{result.native_stats.count_ops.get(gate, 0)}"
            for gate in NATIVE_BASIS_GATES
        )
        row = [
            system.label,
            method,
            system.basis,
            f"{system.bond_length:.6f}",
            str(system.n_frozen),
            str(system.norb),
            f"{system.nelec[0]}:{system.nelec[1]}",
            str(result.n_params),
            str(result.raw_stats.depth),
            str(result.raw_stats.gate_count),
            str(result.raw_stats.two_qubit_gate_count),
            str(result.native_stats.depth),
            str(result.native_stats.gate_count),
            str(result.native_stats.two_qubit_gate_count),
            ops,
        ]
        print(",".join(row))


def plot_summary(
    system: SystemData,
    results: dict[str, StrategyResult],
    output: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), constrained_layout=True)

    ax = axes[0]
    metrics = ("Params", "Raw depth", "Native depth", "Native total", "Native 2q")
    x = np.arange(len(metrics))
    width = 0.34
    offsets = np.asarray([-0.5, 0.5]) * width
    for method, offset in zip(METHODS, offsets):
        result = results[method]
        values = [
            result.n_params,
            result.raw_stats.depth,
            result.native_stats.depth,
            result.native_stats.gate_count,
            result.native_stats.two_qubit_gate_count,
        ]
        ax.bar(
            x + offset,
            values,
            width,
            label=method,
            color=METHOD_COLORS[method],
        )
    ax.set_title(
        f"{system.label}, R={system.bond_length:.2f} A\n"
        f"norb={system.norb}, nelec={system.nelec}"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=20, ha="right")
    ax.set_yscale("log")
    ax.set_ylabel("Count")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)

    ax = axes[1]
    method_x = np.arange(len(METHODS))
    bottom = np.zeros(len(METHODS), dtype=np.float64)
    for gate in NATIVE_BASIS_GATES:
        values = np.asarray(
            [results[method].native_stats.count_ops.get(gate, 0) for method in METHODS],
            dtype=np.float64,
        )
        ax.bar(
            method_x,
            values,
            bottom=bottom,
            label=gate.upper(),
            color=GATE_COLORS[gate],
        )
        bottom += values
    ax.set_xticks(method_x)
    ax.set_xticklabels(METHODS, rotation=20, ha="right")
    ax.set_ylabel("Native gate count")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=250)
    print(f"Wrote figure: {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare product-pUCCD pair-register and spin-orbital state-prep "
            "resource counts for N2 sto-6g with nf=2."
        )
    )
    parser.add_argument("--r", type=float, default=N2_R, help="N2 bond length in A.")
    parser.add_argument("--basis", default=BASIS)
    parser.add_argument("--n-frozen", type=int, default=N_FROZEN)
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--optimization-level", type=int, default=3)
    parser.add_argument("--seed", type=int, default=DEFAULT_TRANSPILE_SEED)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figures/puccd_strategy_stats.png"),
        help="Output figure path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with threadpool_limits(limits=args.threads):
        pyscf.lib.num_threads(args.threads)
        basis_gates = tuple(DEFAULT_NATIVE_BASIS_GATES)
        if tuple(NATIVE_BASIS_GATES) != tuple(DEFAULT_NATIVE_BASIS_GATES):
            basis_gates = tuple(NATIVE_BASIS_GATES)

        system = build_system_data(
            r=args.r,
            basis=args.basis,
            n_frozen=args.n_frozen,
        )
        results = build_strategy_results(
            system,
            optimization_level=args.optimization_level,
            basis_gates=basis_gates,
            seed=args.seed,
        )
        print_summary(system, results)
        plot_summary(system, results, args.output)


if __name__ == "__main__":
    main()
