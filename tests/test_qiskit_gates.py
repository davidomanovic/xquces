from __future__ import annotations

import numpy as np
import pytest

qiskit = pytest.importorskip("qiskit")
pytest.importorskip("qiskit.quantum_info")

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from xquces.basis import occ_rows, reshape_state
from xquces.gates import apply_ucj_spin_balanced, apply_ucj_spin_restricted
from xquces.igcr2 import IGCR2SpinBalancedParameterization
from xquces.orbitals import apply_orbital_rotation, unitary_from_generator
from xquces.qiskit.gates.diag_2 import Diag2SpinBalancedJW, Diag2SpinRestrictedJW
from xquces.qiskit.gates.igcr2 import IGCR2JW
from xquces.qiskit.gates.orbital_rotations import OrbitalRotationJW


def _random_state(dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vec = rng.normal(size=dim) + 1j * rng.normal(size=dim)
    return vec / np.linalg.norm(vec)


def _bitstring_index(occ_alpha, occ_beta, norb: int) -> int:
    alpha_bits = sum(1 << int(p) for p in occ_alpha)
    beta_bits = sum(1 << (norb + int(p)) for p in occ_beta)
    return alpha_bits + beta_bits


def _sector_to_jw_state(vec: np.ndarray, norb: int, nelec: tuple[int, int]) -> np.ndarray:
    out = np.zeros(2 ** (2 * norb), dtype=np.complex128)
    mat = reshape_state(vec, norb, nelec)
    occ_alpha = occ_rows(norb, nelec[0])
    occ_beta = occ_rows(norb, nelec[1])
    for i_alpha, alpha in enumerate(occ_alpha):
        for i_beta, beta in enumerate(occ_beta):
            out[_bitstring_index(alpha, beta, norb)] = mat[i_alpha, i_beta]
    return out


def _jw_state_to_sector(vec: np.ndarray, norb: int, nelec: tuple[int, int]) -> np.ndarray:
    occ_alpha = occ_rows(norb, nelec[0])
    occ_beta = occ_rows(norb, nelec[1])
    out = np.zeros((len(occ_alpha), len(occ_beta)), dtype=np.complex128)
    for i_alpha, alpha in enumerate(occ_alpha):
        for i_beta, beta in enumerate(occ_beta):
            out[i_alpha, i_beta] = vec[_bitstring_index(alpha, beta, norb)]
    return out.reshape(-1)


def _evolve_gate_on_sector(gate, vec: np.ndarray, norb: int, nelec: tuple[int, int]) -> np.ndarray:
    circuit = QuantumCircuit(2 * norb)
    circuit.append(gate, circuit.qubits)
    full = _sector_to_jw_state(vec, norb, nelec)
    evolved = Statevector(full).evolve(circuit).data
    return _jw_state_to_sector(evolved, norb, nelec)


def _random_unitary(norb: int, seed: int, scale: float = 0.1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(norb, norb)) + 1j * rng.normal(size=(norb, norb))
    return unitary_from_generator(scale * (a - a.conj().T))


def test_qiskit_diag2_spin_balanced_matches_xquces_apply():
    rng = np.random.default_rng(1001)
    norb = 3
    nelec = (2, 1)
    dim = len(occ_rows(norb, nelec[0])) * len(occ_rows(norb, nelec[1]))
    vec = _random_state(dim, 1002)
    same = rng.normal(size=(norb, norb))
    same = 0.1 * (same + same.T)
    mixed = rng.normal(size=(norb, norb))
    mixed = 0.1 * (mixed + mixed.T)

    out = _evolve_gate_on_sector(Diag2SpinBalancedJW(norb, same, mixed), vec, norb, nelec)
    ref = apply_ucj_spin_balanced(vec, same, mixed, norb, nelec)

    assert np.allclose(out, ref, atol=1e-10)


def test_qiskit_diag2_spin_restricted_matches_xquces_apply():
    rng = np.random.default_rng(1101)
    norb = 3
    nelec = (2, 1)
    dim = len(occ_rows(norb, nelec[0])) * len(occ_rows(norb, nelec[1]))
    vec = _random_state(dim, 1102)
    double = 0.1 * rng.normal(size=norb)
    pair = rng.normal(size=(norb, norb))
    pair = 0.1 * (pair + pair.T)
    np.fill_diagonal(pair, 0.0)

    out = _evolve_gate_on_sector(Diag2SpinRestrictedJW(norb, double, pair), vec, norb, nelec)
    ref = apply_ucj_spin_restricted(vec, double, pair, norb, nelec)

    assert np.allclose(out, ref, atol=1e-10)


def test_qiskit_orbital_rotation_matches_xquces_apply():
    norb = 3
    nelec = (2, 1)
    dim = len(occ_rows(norb, nelec[0])) * len(occ_rows(norb, nelec[1]))
    vec = _random_state(dim, 1201)
    orbital_rotation = _random_unitary(norb, 1202)

    out = _evolve_gate_on_sector(OrbitalRotationJW(norb, orbital_rotation), vec, norb, nelec)
    ref = apply_orbital_rotation(vec, orbital_rotation, norb, nelec)

    assert np.allclose(out, ref, atol=1e-10)


def test_qiskit_igcr2_matches_xquces_ansatz_apply():
    rng = np.random.default_rng(1301)
    norb = 4
    nocc = 2
    nelec = (nocc, nocc)
    param = IGCR2SpinBalancedParameterization(norb=norb, nocc=nocc)
    x = 0.05 * rng.normal(size=param.n_params)
    ansatz = param.ansatz_from_parameters(x)
    dim = len(occ_rows(norb, nelec[0])) * len(occ_rows(norb, nelec[1]))
    vec = _random_state(dim, 1302)

    out = _evolve_gate_on_sector(IGCR2JW(ansatz), vec, norb, nelec)
    ref = ansatz.apply(vec, nelec=nelec, copy=True)

    assert np.allclose(out, ref, atol=1e-10)
