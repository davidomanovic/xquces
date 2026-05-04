from __future__ import annotations

from collections.abc import Iterator

from qiskit.transpiler.basepasses import BasePass
from qiskit.transpiler.passes import Decompose


def pre_init_passes() -> Iterator[BasePass]:
    yield Decompose(
        [
            "igcr2_jw",
            "igcr3_jw",
            "igcr4_jw",
            "pair_gcr2_jw",
            "product_pair_uccd_jw",
        ]
    )
    yield Decompose(["igcr4_diag4_restricted_jw"])
    yield Decompose(["igcr3_diag3_restricted_jw"])
    yield Decompose(
        [
            "igcr2_diag2_balanced_jw",
            "igcr2_diag2_restricted_jw",
            "orbital_rotation_jw",
            "orbital_rotation_spinless_jw",
            "pair_register_uccd_givens_jw",
            "pair_uccd_rotation_jw",
        ]
    )
