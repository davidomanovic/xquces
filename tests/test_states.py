import numpy as np
import pytest

from xquces.states import (
    determinant_index,
    determinant_state,
    hartree_fock_state,
    linear_combination_state,
    open_shell_singlet_state,
)
from xquces.utils import spin_square


def test_hartree_fock_state_is_first_determinant():
    state = hartree_fock_state(4, (2, 2))

    assert state.shape == (36,)
    assert state[0] == 1.0
    assert np.count_nonzero(state) == 1


def test_determinant_state_selects_alpha_beta_occupation():
    state = determinant_state(4, (2, 2), (0, 2), (1, 3))
    index = determinant_index(4, (2, 2), (0, 2), (1, 3))

    assert state[index] == 1.0
    assert np.count_nonzero(state) == 1


def test_determinant_state_rejects_bad_occupation_length():
    with pytest.raises(ValueError):
        determinant_state(4, (2, 2), (0,), (1, 3))


def test_linear_combination_state_normalizes_terms():
    state = linear_combination_state(
        4,
        (2, 2),
        [
            (2.0, (0, 1), (0, 2)),
            (-2.0, (0, 2), (0, 1)),
        ],
    )

    assert np.linalg.norm(state) == pytest.approx(1.0)
    assert np.count_nonzero(state) == 2


def test_open_shell_singlet_state_uses_expected_determinants():
    state = open_shell_singlet_state(4, (2, 2), (0,), (1, 2))
    i = determinant_index(4, (2, 2), (0, 1), (0, 2))
    j = determinant_index(4, (2, 2), (0, 2), (0, 1))

    assert state[i] == pytest.approx(1 / np.sqrt(2))
    assert state[j] == pytest.approx(1 / np.sqrt(2))
    assert np.linalg.norm(state) == pytest.approx(1.0)


def test_open_shell_singlet_state_has_zero_spin_square():
    state = open_shell_singlet_state(4, (2, 2), (0,), (1, 2))

    assert spin_square(state, 4, (2, 2)) == pytest.approx(0.0)
