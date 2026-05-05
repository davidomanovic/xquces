from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import ffsim
import ffsim.qiskit
import matplotlib.pyplot as plt
import numpy as np
import pyscf
import pyscf.ao2mo
import pyscf.cc
import pyscf.gto
import pyscf.lib
import pyscf.mcscf
import pyscf.scf
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.transpiler import PassManager
from threadpoolctl import threadpool_limits

from xquces import utils as xq_utils
from xquces.gcr.igcr import IGCR2SpinRestrictedParameterization
from xquces.gcr.pair_uccd_reference import GCR2ProductPairUCCDParameterization
from xquces.qiskit import (
    PRE_INIT as XQUCES_PRE_INIT,
    CircuitStats,
    circuit_stats,
    transpile_to_native,
)
from xquces.qiskit.gates import (
    gcr_product_pair_uccd_stateprep_jw_circuit,
    igcr2_stateprep_jw_circuit,
)
from xquces.qiskit.utils import DEFAULT_NATIVE_BASIS_GATES, DEFAULT_TRANSPILE_SEED


HYDROGEN_R = 1.0
N2_R = 1.1
NATIVE_BASIS_GATES = ("cx", "rz", "sx", "x")
LAYERS = (1, 2)
METHODS = (
    "GCR2-L1-HF",
    "GCR2-L2-HF",
    "GCR2-L1-pUCCD",
    "GCR2-L2-pUCCD",
)
METRICS = ("Parameters", "Depth", "Total", "Two-qubit")
METHOD_COLORS = {
    "GCR2-L1-HF": "#1f77b4",
    "GCR2-L2-HF": "#2ca02c",
    "GCR2-L1-pUCCD": "#ff7f0e",
    "GCR2-L2-pUCCD": "#d62728",
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
    energy: float
    abs_error: float


@dataclass(frozen=True)
class SystemResult:
    spec: SystemSpec
    norb: int
    nelec: tuple[int, int]
    e_hf: float
    e_ccsd: float
    e_fci: float
    pauli_terms: int
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


def active_space_data(
    scf: pyscf.scf.hf.SCF,
    active_space: list[int],
) -> tuple[int, tuple[int, int], pyscf.mcscf.RCASCI, np.ndarray, np.ndarray, float]:
    norb = len(active_space)
    nelectron_active = int(round(sum(scf.mo_occ[active_space])))
    n_alpha = (nelectron_active + scf.mol.spin) // 2
    n_beta = (nelectron_active - scf.mol.spin) // 2
    nelec = (n_alpha, n_beta)

    cas = pyscf.mcscf.RCASCI(scf, ncas=norb, nelecas=nelec)
    mo_coeff = cas.sort_mo(active_space, base=0)
    h1, ecore = cas.get_h1eff(mo_coeff=mo_coeff)
    h2 = cas.get_h2eff(mo_coeff=mo_coeff)
    eri = pyscf.ao2mo.restore(1, h2, norb)
    cas.fix_spin_(ss=0)
    cas.kernel(mo_coeff=mo_coeff)
    if not cas.converged:
        raise RuntimeError("RCASCI/FCI failed to converge")
    return (
        norb,
        nelec,
        cas,
        np.asarray(h1, dtype=np.float64),
        np.asarray(eri, dtype=np.float64),
        float(ecore),
    )


def sparse_pauli_hamiltonian(h1: np.ndarray, eri: np.ndarray, ecore: float, norb: int):
    ham = ffsim.MolecularHamiltonian(h1, eri, ecore)
    return ffsim.qiskit.jordan_wigner(ffsim.fermion_operator(ham), norb=norb)


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


def circuit_energy(circuit: QuantumCircuit, hamiltonian) -> float:
    state = Statevector.from_label("0" * circuit.num_qubits).evolve(circuit)
    return float(np.real(state.expectation_value(hamiltonian)))


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
    norb, nelec, cas, h1, eri, ecore = active_space_data(scf, active_space)
    n_alpha, _ = nelec
    frozen = [i for i in range(mol.nao_nr()) if i not in active_space]

    ccsd = pyscf.cc.RCCSD(scf, frozen=frozen)
    ccsd.conv_tol = 1e-12
    ccsd.conv_tol_normt = 1e-10
    ccsd.max_cycle = 1000
    ccsd.kernel()
    if ccsd.t1 is None or ccsd.t2 is None:
        raise RuntimeError(f"RCCSD failed for {spec.label}")

    hamiltonian = sparse_pauli_hamiltonian(h1, eri, ecore, norb)
    t2 = np.asarray(ccsd.t2, dtype=np.float64)
    t1 = np.asarray(ccsd.t1, dtype=np.float64)

    results: dict[str, MethodResult] = {}

    for layers in LAYERS:
        hf_label = f"GCR2-L{layers}-HF"
        hf_param = IGCR2SpinRestrictedParameterization(
            norb=norb,
            nocc=n_alpha,
            layers=layers,
        )
        hf_params = hf_param.parameters_from_t_amplitudes(t2, t1=t1)
        hf_circuit = igcr2_stateprep_jw_circuit(
            hf_param.ansatz_from_parameters(hf_params),
        )
        hf_energy = circuit_energy(hf_circuit, hamiltonian)
        results[hf_label] = MethodResult(
            method=hf_label,
            n_params=hf_param.n_params,
            stats=native_stats(
                hf_circuit,
                hf_label,
                pre_init=XQUCES_PRE_INIT,
                optimization_level=optimization_level,
                basis_gates=basis_gates,
                seed=seed,
            ),
            energy=hf_energy,
            abs_error=abs(hf_energy - float(cas.e_tot)),
        )

        puccd_label = f"GCR2-L{layers}-pUCCD"
        puccd_param = GCR2ProductPairUCCDParameterization(
            norb=norb,
            nocc=n_alpha,
            layers=layers,
        )
        puccd_params = puccd_param.parameters_from_t_amplitudes(
            t2,
            t1=t1,
            scale=0.5,
        )
        puccd_circuit = gcr_product_pair_uccd_stateprep_jw_circuit(
            puccd_param,
            puccd_params,
            puccd_strategy="pair_register",
        )
        puccd_energy = circuit_energy(puccd_circuit, hamiltonian)
        results[puccd_label] = MethodResult(
            method=puccd_label,
            n_params=puccd_param.n_params,
            stats=native_stats(
                puccd_circuit,
                puccd_label,
                pre_init=XQUCES_PRE_INIT,
                optimization_level=optimization_level,
                basis_gates=basis_gates,
                seed=seed,
            ),
            energy=puccd_energy,
            abs_error=abs(puccd_energy - float(cas.e_tot)),
        )

    return SystemResult(
        spec=spec,
        norb=norb,
        nelec=nelec,
        e_hf=float(scf.e_tot),
        e_ccsd=float(ccsd.e_tot),
        e_fci=float(cas.e_tot),
        pauli_terms=len(hamiltonian),
        methods=results,
    )


def plot_resource_summary(results: list[SystemResult], output: Path) -> None:
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
        width = 0.22
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
    print(f"Wrote resource figure: {output}")


def plot_depth_error(results: list[SystemResult], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
    markers = {
        "GCR2-L1-HF": "o",
        "GCR2-L2-HF": "^",
        "GCR2-L1-pUCCD": "s",
        "GCR2-L2-pUCCD": "D",
    }

    for result in results:
        xs = [result.methods[method].stats.depth for method in METHODS]
        ys = [max(result.methods[method].abs_error, 1e-16) for method in METHODS]
        ax.plot(xs, ys, color="0.78", linewidth=1.0, zorder=1)
        for method, x, y in zip(METHODS, xs, ys):
            ax.scatter(
                x,
                y,
                s=70,
                marker=markers[method],
                color=METHOD_COLORS[method],
                edgecolor="white",
                linewidth=0.8,
                label=method,
                zorder=2,
            )
        ax.annotate(
            result.spec.key,
            xy=(xs[-1], ys[-1]),
            xytext=(5, 3),
            textcoords="offset points",
            fontsize=9,
        )

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), frameon=False)
    ax.set_xlabel("Native circuit depth")
    ax.set_ylabel(r"$|E_\mathrm{init} - E_\mathrm{FCI}|$ [$E_h$]")
    ax.set_yscale("log")
    ax.grid(alpha=0.25)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=250)
    print(f"Wrote depth-error figure: {output}")


def print_summary(results: list[SystemResult]) -> None:
    columns = [
        "system",
        "method",
        "norb",
        "nelec",
        "E_HF",
        "E_CCSD",
        "E_FCI",
        "E_init",
        "abs_error",
        "params",
        "depth",
        "total",
        "two_qubit",
        "pauli_terms",
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
                f"{result.e_hf:.12f}",
                f"{result.e_ccsd:.12f}",
                f"{result.e_fci:.12f}",
                f"{item.energy:.12f}",
                f"{item.abs_error:.6e}",
                str(item.n_params),
                str(item.stats.depth),
                str(item.stats.gate_count),
                str(item.stats.two_qubit_gate_count),
                str(result.pauli_terms),
                ops,
            ]
            print(",".join(row))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare one- and two-layer iGCR2-on-HF and iGCR2-on-product-pUCCD "
            "state-preparation resources and initialized energies."
        )
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
        default=Path("figures/puccd_test_resources.png"),
        help="Output resource-summary figure path.",
    )
    parser.add_argument(
        "--energy-output",
        type=Path,
        default=None,
        help="Output depth-vs-energy-error figure path.",
    )
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--optimization-level", type=int, default=3)
    parser.add_argument("--seed", type=int, default=DEFAULT_TRANSPILE_SEED)
    return parser.parse_args()


def default_energy_output(output: Path) -> Path:
    return output.with_name(f"{output.stem}_depth_error{output.suffix}")


def main() -> None:
    args = parse_args()
    with threadpool_limits(limits=args.threads):
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
        plot_resource_summary(results, args.output)
        plot_depth_error(
            results,
            args.energy_output or default_energy_output(args.output),
        )


if __name__ == "__main__":
    main()
