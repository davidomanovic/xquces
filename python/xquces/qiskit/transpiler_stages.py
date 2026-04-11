from __future__ import annotations

from collections.abc import Iterator

from qiskit.transpiler.basepasses import BasePass
from qiskit.transpiler.passes import Decompose


def pre_init_passes() -> Iterator[BasePass]:
    """Yield iGCR-2 passes for Qiskit's ``pre_init`` transpiler stage."""
    yield Decompose(["igcr2_jw"])
    yield Decompose(
        [
            "igcr2_diag2_balanced_jw",
            "igcr2_diag2_restricted_jw",
            "orbital_rotation_jw",
            "orbital_rotation_spinless_jw",
        ]
    )
