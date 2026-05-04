from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import ffsim
import ffsim.qiskit
import matplotlib.pyplot as plt
import numpy as np
import pyscf
import pyscf.cc
import pyscf.gto
import pyscf.lib
import pyscf.scf
from ffsim.qiskit.gates import PrepareHartreeFockJW
from ffsim.qiskit.gates.ucj import UCJOpSpinBalancedJW
from qiskit import QuantumCircuit
from qiskit.transpiler import PassManager
from threadpoolctl import threadpool_limits


from xquces import utils as xq_utils
from xquces.gcr.igcr import (
    IGCR2SpinBalancedParameterization,
    IGCR3SpinRestrictedParameterization,
    IGCR4SpinRestrictedParameterization,
)
from xquces.qiskit import (
    PRE_INIT as XQUCES_PRE_INIT,
    CircuitStats,
    circuit_stats,
    transpile_to_native,
)
from xquces.qiskit.utils import DEFAULT_NATIVE_BASIS_GATES, DEFAULT_TRANSPILE_SEED
from xquces.qiskit.gates import (
    igcr2_stateprep_jw_circuit,
    igcr3_stateprep_jw_circuit,
    igcr4_stateprep_jw_circuit,
)
from xquces.ucj.init import (
    UCJRestrictedProjectedDFSeed,
    _ucj_ansatz_from_ffsim_stock,
)


N_REPS = 1
HYDROGEN_R = 1.0
_RNG = np.random.default_rng(42)
N2_R = 1.1
NATIVE_BASIS_GATES = ("cx", "rz", "sx", "x")
METHODS = ("UCJ", "iGCR2", "iGCR3", "iGCR4")
METRICS = ("Parameters", "Depth", "Total", "Two-qubit")
METHOD_COLORS = {
    "UCJ": "#1f77b4",
    "iGCR2": "#ff7f0e",
    "iGCR3": "#2ca02c",
    "iGCR4": "#d62728",
}
GATE_COLORS = {
    "cx": "#1f77b4",
    "rz": "#ff7f0e",
    "sx": "#2ca02c",
    "x": "#9467bd",
}


@dataclass(frozen=True)
class SystemSpec:
    key: str
    label: str
    kind: str
    basis: str
    n_atoms: int | None = None
    n_frozen: int = 0
    bond_length: float = HYDROGEN_R


@dataclass(frozen=True)
class MethodResult:
    method: str
    n_params: int
    stats: CircuitStats


@dataclass(frozen=True)
class SystemResult:
    spec: SystemSpec
    norb: int
    nelec: tuple[int, int]
    methods: dict[str, MethodResult]


SYSTEMS = (
    SystemSpec(
        key="h4",
        label="H4 chain sto-3g",
        kind="hydrogen",
        basis="sto-3g",
        n_atoms=4,
    ),
    SystemSpec(
        key="h6",
        label="H6 chain sto-3g",
        kind="hydrogen",
        basis="sto-3g",
        n_atoms=6,
    ),
    SystemSpec(
        key="n2_nf2",
        label="N2 sto-6g nf=2",
        kind="n2",
        basis="sto-6g",
        n_frozen=2,
        bond_length=N2_R,
    ),
)


def _perturb_higher_order_params(params: np.ndarray, param_obj, scale: float = 1e-3) -> np.ndarray:
    """Replace zero higher-order (cubic/quartic) params with small random values.

    The flat parameter layout is [left | pair | extra... | right]. Without this,
    zero-valued cubic/quartic terms get optimized away by the transpiler, making
    iGCR3/iGCR4 appear identical to iGCR2 in gate count.
    """
    params = params.copy()
    start = param_obj.n_left_orbital_rotation_params + param_obj.n_pair_params
    end = len(params) - param_obj.n_right_orbital_rotation_params
    params[start:end] = _RNG.normal(0, scale, size=end - start)
    return params


def build_mol(spec: SystemSpec) -> pyscf.gto.Mole:
    if spec.kind == "hydrogen":
        if spec.n_atoms is None:
            raise ValueError("hydrogen systems require n_atoms")
        return xq_utils.build_hydrogen_chain(
            spec.bond_length,
            spec.n_atoms,
            spec.basis,
        )
    if spec.kind == "n2":
        mol = pyscf.gto.Mole()
        r = float(spec.bond_length)
        mol.build(
            atom=[("N", (-0.5 * r, 0.0, 0.0)), ("N", (0.5 * r, 0.0, 0.0))],
            basis=spec.basis,
            symmetry="Dooh",
            verbose=0,
        )
        return mol
    raise ValueError(f"Unknown system kind: {spec.kind!r}")


def ffsim_ucj_seed(t2: np.ndarray, t1: np.ndarray | None):
    return ffsim.UCJOpSpinBalanced.from_t_amplitudes(
        np.asarray(t2, dtype=np.float64),
        t1=None if t1 is None else np.asarray(t1, dtype=np.complex128),
        n_reps=N_REPS,
    )


def ffsim_ucj_stateprep_jw_circuit(ucj_op, nelec: tuple[int, int]) -> QuantumCircuit:
    circuit = QuantumCircuit(2 * ucj_op.norb)
    circuit.append(PrepareHartreeFockJW(ucj_op.norb, nelec), circuit.qubits)
    circuit.append(UCJOpSpinBalancedJW(ucj_op), circuit.qubits)
    return circuit


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


def build_system_result(
    spec: SystemSpec,
    *,
    optimization_level: int,
    basis_gates: tuple[str, ...],
    seed: int,
) -> SystemResult:
    mol = build_mol(spec)
    scf = pyscf.scf.RHF(mol)
    scf.kernel()
    if not scf.converged:
        raise RuntimeError(f"RHF failed for {spec.label}")

    active_space = list(range(spec.n_frozen, mol.nao_nr()))
    if not active_space:
        raise ValueError(f"{spec.label} has no active orbitals")
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
    if ccsd.t1 is None or ccsd.t2 is None:
        raise RuntimeError(f"RCCSD failed for {spec.label}")

    results: dict[str, MethodResult] = {}

    ucj_op = ffsim_ucj_seed(ccsd.t2, ccsd.t1)
    ucj_circuit = ffsim_ucj_stateprep_jw_circuit(ucj_op, nelec)
    ucj_params = ucj_op.n_params(
        norb=norb,
        n_reps=N_REPS,
        with_final_orbital_rotation=ucj_op.final_orbital_rotation is not None,
    )
    results["UCJ"] = MethodResult(
        "UCJ",
        ucj_params,
        native_stats(
            ucj_circuit,
            "UCJ",
            pre_init=ffsim.qiskit.PRE_INIT,
            optimization_level=optimization_level,
            basis_gates=basis_gates,
            seed=seed,
        ),
    )

    balanced_ucj = _ucj_ansatz_from_ffsim_stock(ucj_op)
    igcr2_param = IGCR2SpinBalancedParameterization(norb=norb, nocc=n_alpha)
    igcr2_params = igcr2_param.parameters_from_ucj_ansatz(balanced_ucj)
    igcr2_circuit = igcr2_stateprep_jw_circuit(
        igcr2_param.ansatz_from_parameters(igcr2_params)
    )
    results["iGCR2"] = MethodResult(
        "iGCR2",
        igcr2_param.n_params,
        native_stats(
            igcr2_circuit,
            "iGCR2",
            pre_init=XQUCES_PRE_INIT,
            optimization_level=optimization_level,
            basis_gates=basis_gates,
            seed=seed,
        ),
    )

    restricted_ucj = UCJRestrictedProjectedDFSeed(
        t2=np.asarray(ccsd.t2, dtype=np.float64),
        t1=np.asarray(ccsd.t1, dtype=np.complex128),
        n_reps=N_REPS,
    ).build_ansatz()

    igcr3_param = IGCR3SpinRestrictedParameterization(norb=norb, nocc=n_alpha)
    igcr3_params = _perturb_higher_order_params(
        igcr3_param.parameters_from_ucj_ansatz(restricted_ucj), igcr3_param
    )
    igcr3_circuit = igcr3_stateprep_jw_circuit(
        igcr3_param.ansatz_from_parameters(igcr3_params)
    )
    results["iGCR3"] = MethodResult(
        "iGCR3",
        igcr3_param.n_params,
        native_stats(
            igcr3_circuit,
            "iGCR3",
            pre_init=XQUCES_PRE_INIT,
            optimization_level=optimization_level,
            basis_gates=basis_gates,
            seed=seed,
        ),
    )

    igcr4_param = IGCR4SpinRestrictedParameterization(norb=norb, nocc=n_alpha)
    igcr4_params = _perturb_higher_order_params(
        igcr4_param.parameters_from_ucj_ansatz(restricted_ucj), igcr4_param
    )
    igcr4_circuit = igcr4_stateprep_jw_circuit(
        igcr4_param.ansatz_from_parameters(igcr4_params)
    )
    results["iGCR4"] = MethodResult(
        "iGCR4",
        igcr4_param.n_params,
        native_stats(
            igcr4_circuit,
            "iGCR4",
            pre_init=XQUCES_PRE_INIT,
            optimization_level=optimization_level,
            basis_gates=basis_gates,
            seed=seed,
        ),
    )

    return SystemResult(spec=spec, norb=norb, nelec=nelec, methods=results)


def plot_summary(results: list[SystemResult], output: Path) -> None:
    fig, axes = plt.subplots(
        2,
        len(results),
        figsize=(5.2 * len(results), 8.4),
        constrained_layout=True,
    )
    if len(results) == 1:
        axes = np.asarray(axes).reshape(2, 1)

    for col, result in enumerate(results):
        ax = axes[0, col]
        x = np.arange(len(METRICS))
        width = 0.18
        offsets = (np.arange(len(METHODS)) - 0.5 * (len(METHODS) - 1)) * width
        for method, offset in zip(METHODS, offsets):
            item = result.methods[method]
            values = [
                item.n_params,
                item.stats.depth,
                item.stats.gate_count,
                item.stats.two_qubit_gate_count,
            ]
            ax.bar(
                x + offset,
                values,
                width,
                label=method,
                color=METHOD_COLORS[method],
            )
        ax.set_title(
            f"{result.spec.label}\n"
            f"norb={result.norb}, nelec={result.nelec}",
            fontsize=12,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(METRICS, rotation=20, ha="right")
        ax.set_yscale("log")
        ax.grid(axis="y", alpha=0.25)
        if col == 0:
            ax.set_ylabel("Count")
            ax.legend(frameon=False, fontsize=10)

        ax = axes[1, col]
        method_x = np.arange(len(METHODS))
        bottom = np.zeros(len(METHODS), dtype=np.float64)
        for gate in NATIVE_BASIS_GATES:
            values = np.asarray(
                [
                    result.methods[method].stats.count_ops.get(gate, 0)
                    for method in METHODS
                ],
                dtype=np.float64,
            )
            ax.bar(
                method_x,
                values,
                bottom=bottom,
                label=gate.upper(),
                color=GATE_COLORS.get(gate),
            )
            bottom += values
        ax.set_xticks(method_x)
        ax.set_xticklabels(METHODS, rotation=20, ha="right")
        ax.grid(axis="y", alpha=0.25)
        if col == 0:
            ax.set_ylabel("Native gate count")
        if col == len(results) - 1:
            ax.legend(frameon=False, fontsize=10)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=250)
    print(f"Wrote figure: {output}")


def print_summary(results: list[SystemResult]) -> None:
    columns = [
        "system",
        "method",
        "norb",
        "nelec",
        "params",
        "depth",
        "total",
        "two_qubit",
        "ops",
    ]
    print(",".join(columns))
    for result in results:
        for method in METHODS:
            item = result.methods[method]
            ops = ";".join(
                f"{gate}:{item.stats.count_ops.get(gate, 0)}"
                for gate in NATIVE_BASIS_GATES
            )
            row = [
                result.spec.key,
                method,
                str(result.norb),
                f"{result.nelec[0]}:{result.nelec[1]}",
                str(item.n_params),
                str(item.stats.depth),
                str(item.stats.gate_count),
                str(item.stats.two_qubit_gate_count),
                ops,
            ]
            print(",".join(row))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare UCJ and iGCR state-preparation circuit resources."
    )
    parser.add_argument(
        "--systems",
        nargs="+",
        choices=[spec.key for spec in SYSTEMS],
        default=[spec.key for spec in SYSTEMS],
        help="Subset of systems to run.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figures/circuit_test.png"),
        help="Output figure path.",
    )
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--optimization-level", type=int, default=3)
    parser.add_argument("--seed", type=int, default=DEFAULT_TRANSPILE_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pyscf.lib.num_threads(args.threads)
    selected = [spec for spec in SYSTEMS if spec.key in set(args.systems)]
    basis_gates = tuple(DEFAULT_NATIVE_BASIS_GATES)
    if tuple(NATIVE_BASIS_GATES) != tuple(DEFAULT_NATIVE_BASIS_GATES):
        basis_gates = tuple(NATIVE_BASIS_GATES)

    results = []
    for spec in selected:
        print(f"Building {spec.label}...", flush=True)
        results.append(
            build_system_result(
                spec,
                optimization_level=args.optimization_level,
                basis_gates=basis_gates,
                seed=args.seed,
            )
        )

    print_summary(results)
    plot_summary(results, args.output)


if __name__ == "__main__":
    with threadpool_limits(limits=12):
        main()
