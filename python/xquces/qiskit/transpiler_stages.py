from __future__ import annotations

from collections.abc import Iterator

from qiskit.transpiler.basepasses import BasePass
from qiskit.transpiler.passes import Decompose


def pre_init_passes() -> Iterator[BasePass]:
    """Yield iGCR pre-init decompositions for Qiskit's transpiler."""
    yield Decompose(["igcr2_jw", "igcr3_jw"])
    yield Decompose(["igcr3_diag3_restricted_jw"])
    yield Decompose(
        [
            "igcr2_diag2_balanced_jw",
            "igcr2_diag2_restricted_jw",
            "orbital_rotation_jw",
            "orbital_rotation_spinless_jw",
        ]
    )
