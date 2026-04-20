from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Callable

from xquces.basis import flatten_state, reshape_state
from xquces.gcr.commutator_gcr2 import (
    _apply_pairhop_gate_numpy,
    _diag2_features,
    _edge_coloring_matchings,
    _pair_hop_gate_arrays,
    _validate_pairs,
    apply_gcr2_pairhop_product_middle,
    apply_gcr2_pairhop_product_middle_cached_in_place_num_rep,
)
from xquces.gcr.igcr2 import (
    IGCR2Ansatz,
    IGCR2SpinRestrictedParameterization,
    IGCR2SpinRestrictedSpec,
    _symmetric_matrix_from_values,
    orbital_relabeling_from_overlap,
)
from xquces.orbitals import apply_orbital_rotation
from xquces.ucj.model import UCJAnsatz


def apply_pair_reference_product(
    vec: np.ndarray,
    pair_reference_params: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    pairs: tuple[tuple[int, int], ...],
    copy: bool = True,
) -> np.ndarray:
    zeros = np.zeros(len(pairs), dtype=np.float64)
    return apply_gcr2_pairhop_product_middle(
        vec,
        zeros,
        np.asarray(pair_reference_params, dtype=np.float64),
        norb,
        nelec,
        pairs,
        copy=copy,
    )


def _transfer_pair_reference_params(
    pair_reference_params: np.ndarray,
    previous_pairs: tuple[tuple[int, int], ...],
    new_pairs: tuple[tuple[int, int], ...],
    old_for_new: np.ndarray | None,
    phases: np.ndarray | None,
) -> np.ndarray:
    pair_reference_params = np.asarray(pair_reference_params, dtype=np.float64)
    if pair_reference_params.shape != (len(previous_pairs),):
        raise ValueError("pair_reference_params has the wrong shape")
    if old_for_new is None:
        if previous_pairs == new_pairs:
            return np.array(pair_reference_params, copy=True)
        previous_map = {
            pair: float(pair_reference_params[idx])
            for idx, pair in enumerate(previous_pairs)
        }
        return np.asarray(
            [previous_map.get(pair, 0.0) for pair in new_pairs],
            dtype=np.float64,
        )

    old_for_new = np.asarray(old_for_new, dtype=np.int64)
    if old_for_new.ndim != 1:
        raise ValueError("old_for_new must be one-dimensional")
    norb = old_for_new.size
    if phases is None:
        phase_arr = np.ones(norb, dtype=np.complex128)
    else:
        phase_arr = np.asarray(phases, dtype=np.complex128)
        if phase_arr.shape != (norb,):
            raise ValueError("phases must have shape (norb,)")

    current_for_old = np.empty_like(old_for_new)
    current_for_old[old_for_new] = np.arange(norb)

    new_map: dict[tuple[int, int], float] = {}
    for idx, (p_old, q_old) in enumerate(previous_pairs):
        p_new = int(current_for_old[p_old])
        q_new = int(current_for_old[q_old])
        coeff = float(pair_reference_params[idx])

        gamma = (phase_arr[p_new] ** 2) * np.conjugate(phase_arr[q_new]) ** 2
        if abs(np.imag(gamma)) > 1e-8 or not np.isclose(
            abs(np.real(gamma)), 1.0, atol=1e-8
        ):
            raise ValueError(
                "pair-reference transfer encountered non-real pair phase; "
                "this parameterization only supports real pair-hop angles."
            )
        coeff *= float(np.real(gamma))

        if p_new > q_new:
            p_new, q_new = q_new, p_new
            coeff = -coeff

        key = (p_new, q_new)
        new_map[key] = new_map.get(key, 0.0) + coeff

    return np.asarray([new_map.get(pair, 0.0) for pair in new_pairs], dtype=np.float64)


def _apply_pair_reference_product_batch(
    batch: np.ndarray,
    pair_reference_params: np.ndarray,
    norb: int,
    pairs: tuple[tuple[int, int], ...],
    source: np.ndarray,
    target: np.ndarray,
    sign: np.ndarray,
    starts: np.ndarray,
    order: np.ndarray | None = None,
) -> np.ndarray:
    batch = np.asarray(batch, dtype=np.complex128)
    if batch.ndim != 2:
        raise ValueError("batch must be two-dimensional")
    out = np.array(batch, dtype=np.complex128, copy=True)
    pair_reference_params = np.asarray(pair_reference_params, dtype=np.float64)
    if pair_reference_params.shape != (len(pairs),):
        raise ValueError("pair_reference_params has the wrong shape")
    gate_order = (
        np.asarray(order, dtype=np.uintp)
        if order is not None
        else _edge_coloring_matchings(norb, pairs)
    )
    for pair_index in gate_order:
        pair_index = int(pair_index)
        start = int(starts[pair_index])
        stop = int(starts[pair_index + 1])
        theta = float(pair_reference_params[pair_index])
        if theta == 0.0 or start == stop:
            continue
        for row in range(out.shape[0]):
            _apply_pairhop_gate_numpy(
                out[row],
                theta,
                source[start:stop],
                target[start:stop],
                sign[start:stop],
            )
    return out


def _pair_reference_gate_derivative_on_state(
    vec: np.ndarray,
    theta: float,
    source: np.ndarray,
    target: np.ndarray,
    sign: np.ndarray,
) -> np.ndarray:
    out = np.zeros_like(np.asarray(vec, dtype=np.complex128))
    if source.size == 0:
        return out
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    v_source = np.asarray(vec[source], dtype=np.complex128)
    v_target = np.asarray(vec[target], dtype=np.complex128)
    signed_c = sign * c
    out[source] = -s * v_source - signed_c * v_target
    out[target] = -s * v_target + signed_c * v_source
    return out


def _pair_reference_derivative_vectors(
    vec: np.ndarray,
    pair_reference_params: np.ndarray,
    norb: int,
    pairs: tuple[tuple[int, int], ...],
    source: np.ndarray,
    target: np.ndarray,
    sign: np.ndarray,
    starts: np.ndarray,
    order: np.ndarray | None = None,
) -> np.ndarray:
    pair_reference_params = np.asarray(pair_reference_params, dtype=np.float64)
    if pair_reference_params.shape != (len(pairs),):
        raise ValueError("pair_reference_params has the wrong shape")
    gate_order = list(
        np.asarray(order, dtype=np.uintp)
        if order is not None
        else _edge_coloring_matchings(norb, pairs)
    )
    forward = [np.asarray(vec, dtype=np.complex128)]
    current = np.array(vec, dtype=np.complex128, copy=True)
    for pair_index in gate_order:
        pair_index = int(pair_index)
        start = int(starts[pair_index])
        stop = int(starts[pair_index + 1])
        _apply_pairhop_gate_numpy(
            current,
            float(pair_reference_params[pair_index]),
            source[start:stop],
            target[start:stop],
            sign[start:stop],
        )
        forward.append(current.copy())
    position = {int(pair_index): pos for pos, pair_index in enumerate(gate_order)}
    derivs = np.zeros((len(pairs), current.size), dtype=np.complex128)
    for pair_index in range(len(pairs)):
        pos = position[pair_index]
        start = int(starts[pair_index])
        stop = int(starts[pair_index + 1])
        tmp = _pair_reference_gate_derivative_on_state(
            forward[pos],
            float(pair_reference_params[pair_index]),
            source[start:stop],
            target[start:stop],
            sign[start:stop],
        )
        for later_pos in range(pos + 1, len(gate_order)):
            later_pair = int(gate_order[later_pos])
            later_start = int(starts[later_pair])
            later_stop = int(starts[later_pair + 1])
            _apply_pairhop_gate_numpy(
                tmp,
                float(pair_reference_params[later_pair]),
                source[later_start:later_stop],
                target[later_start:later_stop],
                sign[later_start:later_stop],
            )
        derivs[pair_index] = tmp
    return derivs


@dataclass(frozen=True)
class GCR2PairReferenceAnsatz:
    pair_params: np.ndarray
    pair_reference_params: np.ndarray
    left: np.ndarray
    right: np.ndarray
    norb: int
    nocc: int
    pairs: tuple[tuple[int, int], ...]
    use_rust: bool = True

    def apply(self, vec: np.ndarray, nelec: tuple[int, int], copy: bool = True) -> np.ndarray:
        out = apply_orbital_rotation(
            vec,
            self.right,
            self.norb,
            nelec,
            copy=copy,
        )
        if (
            self.use_rust
            and apply_gcr2_pairhop_product_middle_cached_in_place_num_rep is not None
        ):
            out2 = reshape_state(out, self.norb, nelec)
            source, target, sign, starts = _pair_hop_gate_arrays(
                self.norb, nelec, self.pairs
            )
            apply_gcr2_pairhop_product_middle_cached_in_place_num_rep(
                out2,
                np.zeros(len(self.pairs), dtype=np.float64),
                self.pair_reference_params,
                _diag2_features(self.norb, nelec, self.pairs),
                source,
                target,
                sign,
                starts,
                _edge_coloring_matchings(self.norb, self.pairs),
            )
            out = flatten_state(out2)
        else:
            out = apply_pair_reference_product(
                out,
                self.pair_reference_params,
                self.norb,
                nelec,
                self.pairs,
                copy=False,
            )
        phases = _diag2_features(self.norb, nelec, self.pairs) @ self.pair_params
        out *= np.exp(1j * phases)
        return apply_orbital_rotation(
            out,
            self.left,
            self.norb,
            nelec,
            copy=False,
        )


@dataclass(frozen=True)
class GCR2PairReferenceParameterization:
    norb: int
    nocc: int
    pairs: list[tuple[int, int]] | None = None
    base_parameterization: IGCR2SpinRestrictedParameterization | None = None
    left_right_ov_relative_scale: float | None = 1.0
    real_right_orbital_chart: bool = False
    _pairs: tuple[tuple[int, int], ...] = field(init=False, repr=False)

    def __post_init__(self):
        object.__setattr__(self, "_pairs", _validate_pairs(self.pairs, self.norb))
        if self.base_parameterization is not None:
            if self.base_parameterization.norb != self.norb:
                raise ValueError("base_parameterization.norb does not match")
            if self.base_parameterization.nocc != self.nocc:
                raise ValueError("base_parameterization.nocc does not match")

    @property
    def pair_indices(self) -> tuple[tuple[int, int], ...]:
        return self._pairs

    @property
    def _base(self) -> IGCR2SpinRestrictedParameterization:
        if self.base_parameterization is not None:
            return self.base_parameterization
        return IGCR2SpinRestrictedParameterization(
            self.norb,
            self.nocc,
            interaction_pairs=list(self._pairs),
            real_right_orbital_chart=self.real_right_orbital_chart,
            left_right_ov_relative_scale=self.left_right_ov_relative_scale,
        )

    @property
    def right_orbital_chart(self):
        return self._base.right_orbital_chart

    @property
    def _left_orbital_chart(self):
        return self._base._left_orbital_chart

    @property
    def n_left_orbital_rotation_params(self) -> int:
        return self._base.n_left_orbital_rotation_params

    @property
    def n_pair_params(self) -> int:
        return len(self._pairs)

    @property
    def n_pair_reference_params(self) -> int:
        return len(self._pairs)

    @property
    def n_pair_hop_params(self) -> int:
        return self.n_pair_reference_params

    @property
    def n_right_orbital_rotation_params(self) -> int:
        return self._base.n_right_orbital_rotation_params

    @property
    def _right_orbital_rotation_start(self) -> int:
        return (
            self.n_left_orbital_rotation_params
            + self.n_pair_params
            + self.n_pair_reference_params
        )

    @property
    def _left_right_ov_transform_scale(self):
        return None

    def _native_parameters_from_public(self, params: np.ndarray) -> np.ndarray:
        return np.asarray(params, dtype=np.float64)

    def _public_parameters_from_native(self, params: np.ndarray) -> np.ndarray:
        return np.asarray(params, dtype=np.float64)

    @property
    def n_params(self) -> int:
        return (
            self.n_left_orbital_rotation_params
            + self.n_pair_params
            + self.n_pair_reference_params
            + self.n_right_orbital_rotation_params
        )

    def _split(
        self, params: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        params = np.asarray(params, dtype=np.float64)
        if params.shape != (self.n_params,):
            raise ValueError(f"Expected {(self.n_params,)}, got {params.shape}.")
        idx = 0
        n = self.n_left_orbital_rotation_params
        left = params[idx : idx + n]
        idx += n
        n = self.n_pair_params
        pair = params[idx : idx + n]
        idx += n
        n = self.n_pair_reference_params
        pair_reference = params[idx : idx + n]
        idx += n
        right = params[idx:]
        return left, pair, pair_reference, right

    def _base_params_from_split(
        self,
        left: np.ndarray,
        pair: np.ndarray,
        right: np.ndarray,
    ) -> np.ndarray:
        return np.concatenate([left, pair, right]).astype(np.float64, copy=False)

    def ansatz_from_parameters(self, params: np.ndarray) -> GCR2PairReferenceAnsatz:
        left, pair, pair_reference, right = self._split(params)
        base_ansatz = self._base.ansatz_from_parameters(
            self._base_params_from_split(left, pair, right)
        )
        if not isinstance(base_ansatz, IGCR2Ansatz):
            raise TypeError("base parameterization returned an unexpected ansatz")
        pair_matrix = np.asarray(base_ansatz.diagonal.pair, dtype=np.float64)
        pair_values = np.asarray(
            [pair_matrix[p, q] for p, q in self._pairs],
            dtype=np.float64,
        )
        return GCR2PairReferenceAnsatz(
            pair_params=pair_values,
            pair_reference_params=np.asarray(pair_reference, dtype=np.float64),
            left=np.asarray(base_ansatz.left, dtype=np.complex128),
            right=np.asarray(base_ansatz.right, dtype=np.complex128),
            norb=self.norb,
            nocc=self.nocc,
            pairs=self._pairs,
        )

    def parameters_from_ansatz(
        self, ansatz: GCR2PairReferenceAnsatz | IGCR2Ansatz
    ) -> np.ndarray:
        if isinstance(ansatz, GCR2PairReferenceAnsatz):
            base_ansatz = IGCR2Ansatz(
                diagonal=IGCR2SpinRestrictedSpec(
                    pair=_symmetric_matrix_from_values(
                        ansatz.pair_params,
                        self.norb,
                        list(self._pairs),
                    )
                ),
                left=np.asarray(ansatz.left, dtype=np.complex128),
                right=np.asarray(ansatz.right, dtype=np.complex128),
                nocc=ansatz.nocc,
            )
            base_params = self._base.parameters_from_ansatz(base_ansatz)
            left = base_params[: self.n_left_orbital_rotation_params]
            pair_start = self.n_left_orbital_rotation_params
            pair_stop = pair_start + self.n_pair_params
            pair = base_params[pair_start:pair_stop]
            right = base_params[pair_stop:]
            return np.concatenate(
                [
                    left,
                    pair,
                    np.asarray(ansatz.pair_reference_params, dtype=np.float64),
                    right,
                ]
            )
        if isinstance(ansatz, IGCR2Ansatz):
            base_params = self._base.parameters_from_ansatz(ansatz)
            left = base_params[: self.n_left_orbital_rotation_params]
            pair_start = self.n_left_orbital_rotation_params
            pair_stop = pair_start + self.n_pair_params
            pair = base_params[pair_start:pair_stop]
            right = base_params[pair_stop:]
            return np.concatenate(
                [
                    left,
                    pair,
                    np.zeros(self.n_pair_reference_params, dtype=np.float64),
                    right,
                ]
            )
        raise TypeError(type(ansatz).__name__)

    def parameters_from_igcr2(
        self,
        params: np.ndarray,
        parameterization: IGCR2SpinRestrictedParameterization | None = None,
    ) -> np.ndarray:
        parameterization = self._base if parameterization is None else parameterization
        if parameterization.norb != self.norb or parameterization.nocc != self.nocc:
            raise ValueError("IGCR2 parameterization shape does not match")
        ansatz = parameterization.ansatz_from_parameters(params)
        return self.parameters_from_ansatz(ansatz)

    def parameters_from_ucj_ansatz(self, ansatz: UCJAnsatz) -> np.ndarray:
        base_params = self._base.parameters_from_ucj_ansatz(ansatz)
        left = base_params[: self.n_left_orbital_rotation_params]
        pair_start = self.n_left_orbital_rotation_params
        pair_stop = pair_start + self.n_pair_params
        pair = base_params[pair_start:pair_stop]
        right = base_params[pair_stop:]
        return np.concatenate(
            [
                left,
                pair,
                np.zeros(self.n_pair_reference_params, dtype=np.float64),
                right,
            ]
        )

    def transfer_parameters_from(
        self,
        previous_parameters: np.ndarray,
        previous_parameterization: (
            "GCR2PairReferenceParameterization | IGCR2SpinRestrictedParameterization | None"
        ) = None,
        old_for_new: np.ndarray | None = None,
        phases: np.ndarray | None = None,
        orbital_overlap: np.ndarray | None = None,
        block_diagonal: bool = True,
    ) -> np.ndarray:
        if previous_parameterization is None:
            previous_parameterization = self
        if (
            isinstance(previous_parameterization, GCR2PairReferenceParameterization)
            and previous_parameterization.norb == self.norb
            and previous_parameterization.nocc == self.nocc
            and old_for_new is None
            and phases is None
            and orbital_overlap is None
            and previous_parameterization.pair_indices == self.pair_indices
        ):
            params = np.asarray(previous_parameters, dtype=np.float64)
            if params.shape != (self.n_params,):
                raise ValueError(f"Expected {(self.n_params,)}, got {params.shape}.")
            return np.array(params, copy=True)

        if orbital_overlap is not None:
            if old_for_new is not None or phases is not None:
                raise ValueError(
                    "Pass either orbital_overlap or explicit relabeling, not both."
                )
            old_for_new, phases = orbital_relabeling_from_overlap(
                orbital_overlap, nocc=self.nocc, block_diagonal=block_diagonal
            )

        if isinstance(previous_parameterization, GCR2PairReferenceParameterization):
            prev_params = np.asarray(previous_parameters, dtype=np.float64)
            if prev_params.shape != (previous_parameterization.n_params,):
                raise ValueError(
                    f"Expected {(previous_parameterization.n_params,)}, got {prev_params.shape}."
                )
            prev_left, prev_pair, prev_pair_reference, prev_right = (
                previous_parameterization._split(prev_params)
            )
            base_params = self._base.transfer_parameters_from(
                previous_parameterization._base_params_from_split(
                    prev_left, prev_pair, prev_right
                ),
                previous_parameterization=previous_parameterization._base,
                old_for_new=old_for_new,
                phases=phases,
                orbital_overlap=None,
                block_diagonal=block_diagonal,
            )
            n_left = self.n_left_orbital_rotation_params
            n_pair = self.n_pair_params
            left = base_params[:n_left]
            pair = base_params[n_left : n_left + n_pair]
            right = base_params[n_left + n_pair :]
            pair_reference = _transfer_pair_reference_params(
                prev_pair_reference,
                previous_parameterization.pair_indices,
                self.pair_indices,
                old_for_new,
                phases,
            )
            return np.concatenate([left, pair, pair_reference, right])

        base_params = self._base.transfer_parameters_from(
            previous_parameters,
            previous_parameterization=previous_parameterization,
            old_for_new=old_for_new,
            phases=phases,
            orbital_overlap=None,
            block_diagonal=block_diagonal,
        )
        return self.parameters_from_igcr2(base_params, parameterization=self._base)

    def params_to_vec(
        self,
        reference_vec: np.ndarray,
        nelec: tuple[int, int],
    ) -> Callable[[np.ndarray], np.ndarray]:
        reference_vec = np.asarray(reference_vec, dtype=np.complex128)

        def func(params: np.ndarray) -> np.ndarray:
            return self.ansatz_from_parameters(params).apply(
                reference_vec,
                nelec=nelec,
                copy=True,
            )

        return func
