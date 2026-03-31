# variational.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Sequence, Tuple, Optional, Dict
from qiskit.circuit.library import XXPlusYYGate
import numpy as np
from qiskit.circuit import QuantumCircuit, QuantumRegister, ParameterVector, Parameter

def _xx_plus_yy(qc: QuantumCircuit, i: int, j: int, theta):
    qc.append(XXPlusYYGate(theta), [i, j])

def _a(p: int) -> int: return p
def _b(norb: int, p: int) -> int: return norb + p

def lucj(
    norb: int,
    interaction_pairs: tuple[Sequence[tuple[int, int]] | None, Sequence[tuple[int, int]] | None],
    *,
    n_reps: int = 1,
    parameter_prefix: str = "lucj",
    with_final_orbital_rotation: bool = False,
) -> QuantumCircuit:

    def _check_pairs(pairs, name):
        if pairs is None:
            return []
        seen = set()
        out = []
        for (i, j) in pairs:
            if not (0 <= i < norb and 0 <= j < norb):
                raise ValueError(f"{name}: pair {(i,j)} out of range")
            if i > j:
                raise ValueError(f"{name}: pair {(i,j)} not upper triangular (require i<=j)")
            if (i, j) in seen:
                raise ValueError(f"{name}: duplicate pair {(i,j)}")
            seen.add((i, j))
            out.append((int(i), int(j)))
        return out

    pairs_aa = _check_pairs(interaction_pairs[0], "pairs_aa")
    pairs_ab = _check_pairs(interaction_pairs[1], "pairs_ab")

    q = QuantumRegister(2 * norb, "q")
    qc = QuantumCircuit(q, name="UCJ_spin_balanced")

    # index sets
    ij_pairs = [(i, j) for i in range(norb) for j in range(i + 1, norb)]  # i<j

    # parameters
    U_A = [ParameterVector(f"{parameter_prefix}_U_A_rep{r}", len(ij_pairs)) for r in range(n_reps)]
    U_B = [ParameterVector(f"{parameter_prefix}_U_B_rep{r}", len(ij_pairs)) for r in range(n_reps)]
    U_Z = [ParameterVector(f"{parameter_prefix}_U_Z_rep{r}", norb) for r in range(n_reps)]
    J_AA = [ParameterVector(f"{parameter_prefix}_Jaa_rep{r}", len(pairs_aa)) for r in range(n_reps)]
    J_AB = [ParameterVector(f"{parameter_prefix}_Jab_rep{r}", len(pairs_ab)) for r in range(n_reps)]
    Ufinal_A = Ufinal_B = Ufinal_Z = None
    if with_final_orbital_rotation:
        Ufinal_A = ParameterVector(f"{parameter_prefix}_U_A_final", len(ij_pairs))
        Ufinal_B = ParameterVector(f"{parameter_prefix}_U_B_final", len(ij_pairs))
        Ufinal_Z = ParameterVector(f"{parameter_prefix}_U_Z_final", norb)

    def apply_U(Avec, Bvec, Zvec, dagger: bool = False):
        if not dagger:
            for k, (i, j) in enumerate(ij_pairs):
                _xx_plus_yy(qc, _a(i), _a(j), Avec[k])
                _xx_plus_yy(qc, _b(norb, i), _b(norb, j), Avec[k])
            for k, (i, j) in reversed(list(enumerate(ij_pairs))):
                _xx_plus_yy(qc, _a(i), _a(j), Bvec[k])
                _xx_plus_yy(qc, _b(norb, i), _b(norb, j), Bvec[k])
            for i in range(norb):
                qc.p(Zvec[i], _a(i))
                qc.p(Zvec[i], _b(norb, i))
        else:
            for i in range(norb):
                qc.p(-Zvec[i], _a(i))
                qc.p(-Zvec[i], _b(norb, i))
            for k, (i, j) in enumerate(ij_pairs):
                _xx_plus_yy(qc, _a(i), _a(j), -Bvec[k])
                _xx_plus_yy(qc, _b(norb, i), _b(norb, j), -Bvec[k])
            for k, (i, j) in reversed(list(enumerate(ij_pairs))):
                _xx_plus_yy(qc, _a(i), _a(j), -Avec[k])
                _xx_plus_yy(qc, _b(norb, i), _b(norb, j), -Avec[k])

    # build repetitions
    for r in range(n_reps):
        apply_U(U_A[r], U_B[r], U_Z[r], dagger=False)

        # Jastrow
        for k, (i, j) in enumerate(pairs_aa):
            phi = J_AA[r][k]
            if i == j:
                qc.p(phi / 2, _a(i))
                qc.p(phi / 2, _b(norb, i))
            else:
                qc.cp(phi, _a(i), _a(j))
                qc.cp(phi, _b(norb, i), _b(norb, j))
        for k, (i, j) in enumerate(pairs_ab):
            phi = J_AB[r][k]
            qc.cp(phi, _a(i), _b(norb, j))

        apply_U(U_A[r], U_B[r], U_Z[r], dagger=True)
        if r != n_reps - 1:
            qc.barrier()

    if with_final_orbital_rotation:
        apply_U(Ufinal_A, Ufinal_B, Ufinal_Z, dagger=False)

    return qc

def lucj_parameter_order(
    circ: QuantumCircuit,
    norb: int,
    *,
    n_reps: int,
    parameter_prefix: str = "lucj",
    with_final_orbital_rotation: bool = False,
    pairs_aa: Sequence[tuple[int,int]] | None = None,
    pairs_ab: Sequence[tuple[int,int]] | None = None,
) -> list[Parameter]:
    params_by_name = {p.name: p for p in circ.parameters}
    ij_pairs = [(i, j) for i in range(norb) for j in range(i + 1, norb)]
    pairs_aa = [] if pairs_aa is None else list(pairs_aa)
    pairs_ab = [] if pairs_ab is None else list(pairs_ab)

    out: list[Parameter] = []
    for r in range(n_reps):
        out += [params_by_name[f"{parameter_prefix}_U_A_rep{r}[{k}]"] for k in range(len(ij_pairs))]
        out += [params_by_name[f"{parameter_prefix}_U_B_rep{r}[{k}]"] for k in range(len(ij_pairs))]
        out += [params_by_name[f"{parameter_prefix}_U_Z_rep{r}[{k}]"] for k in range(norb)]
        out += [params_by_name[f"{parameter_prefix}_Jaa_rep{r}[{k}]"] for k in range(len(pairs_aa))]
        out += [params_by_name[f"{parameter_prefix}_Jab_rep{r}[{k}]"] for k in range(len(pairs_ab))]
    if with_final_orbital_rotation:
        out += [params_by_name[f"{parameter_prefix}_U_A_final[{k}]"] for k in range(len(ij_pairs))]
        out += [params_by_name[f"{parameter_prefix}_U_B_final[{k}]"] for k in range(len(ij_pairs))]
        out += [params_by_name[f"{parameter_prefix}_U_Z_final[{k}]"] for k in range(norb)]
    return out


def bind_from_vector(params: List[Parameter], x: np.ndarray) -> Dict[Parameter, float]:
    if len(x) != len(params):
        raise ValueError(f"Expected vector of length {len(params)}, got {len(x)}")
    return {p: float(v) for p, v in zip(params, x)}