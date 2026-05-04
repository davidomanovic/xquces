from __future__ import annotations

from xquces.qiskit.gates.diag_2 import Diag2SpinBalancedJW, Diag2SpinRestrictedJW
from xquces.qiskit.gates.diag_3 import Diag3SpinRestrictedJW
from xquces.qiskit.gates.diag_4 import Diag4SpinRestrictedJW
from xquces.qiskit.gates.igcr2 import (
    IGCR2JW,
    igcr2_jw_circuit,
    igcr2_stateprep_jw_circuit,
    spin_balanced_circuit_gauge,
    spin_balanced_rzz_circuit_gauge,
)
from xquces.qiskit.gates.igcr3 import (
    IGCR3JW,
    igcr3_jw_circuit,
    igcr3_stateprep_jw_circuit,
)
from xquces.qiskit.gates.igcr4 import (
    IGCR4JW,
    igcr4_jw_circuit,
    igcr4_stateprep_jw_circuit,
)
from xquces.qiskit.gates.orbital_rotations import (
    OrbitalRotationJW,
    OrbitalRotationSpinlessJW,
)
from xquces.qiskit.gates.pair_gcr2 import (
    PairGCR2JW,
    pair_gcr2_stateprep_jw_circuit,
    product_pair_uccd_pair_gcr2_stateprep_jw_circuit,
)
from xquces.qiskit.gates.product_pair_uccd import (
    PairRegisterUCCDGivensJW,
    PairUCCDRotationJW,
    ProductPairUCCDJW,
    gcr_product_pair_uccd_stateprep_jw_circuit,
    product_pair_uccd_igcr_stateprep_jw_circuit,
    product_pair_uccd_jw_circuit,
    product_pair_uccd_pair_register_stateprep_jw_circuit,
    product_pair_uccd_stateprep_jw_circuit,
)

__all__ = [
    "Diag2SpinBalancedJW",
    "Diag2SpinRestrictedJW",
    "Diag3SpinRestrictedJW",
    "Diag4SpinRestrictedJW",
    "IGCR2JW",
    "IGCR3JW",
    "IGCR4JW",
    "OrbitalRotationJW",
    "OrbitalRotationSpinlessJW",
    "PairGCR2JW",
    "PairRegisterUCCDGivensJW",
    "PairUCCDRotationJW",
    "ProductPairUCCDJW",
    "gcr_product_pair_uccd_stateprep_jw_circuit",
    "igcr2_jw_circuit",
    "igcr2_stateprep_jw_circuit",
    "igcr3_jw_circuit",
    "igcr3_stateprep_jw_circuit",
    "igcr4_jw_circuit",
    "igcr4_stateprep_jw_circuit",
    "product_pair_uccd_igcr_stateprep_jw_circuit",
    "product_pair_uccd_jw_circuit",
    "product_pair_uccd_pair_gcr2_stateprep_jw_circuit",
    "product_pair_uccd_pair_register_stateprep_jw_circuit",
    "product_pair_uccd_stateprep_jw_circuit",
    "pair_gcr2_stateprep_jw_circuit",
    "spin_balanced_circuit_gauge",
    "spin_balanced_rzz_circuit_gauge",
]
