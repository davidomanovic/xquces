from __future__ import annotations

from qiskit.transpiler import PassManager

from xquces.qiskit.transpiler_stages import pre_init_passes

PRE_INIT = PassManager(list(pre_init_passes()))

__all__ = [
    "PRE_INIT",
    "pre_init_passes",
]
