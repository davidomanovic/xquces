import numpy as np

from xquces.basis import flatten_state, occ_rows, reshape_state, sector_dim


def test_occ_rows_shapes():
    rows = occ_rows(4, 2)
    assert rows.shape == (6, 2)
    assert rows.dtype == np.uintp
    expected = np.array(
        [
            [0, 1],
            [0, 2],
            [0, 3],
            [1, 2],
            [1, 3],
            [2, 3],
        ],
        dtype=np.uintp,
    )
    assert np.array_equal(rows, expected)


def test_occ_rows_zero_particles():
    rows = occ_rows(5, 0)
    assert rows.shape == (1, 0)
    assert rows.dtype == np.uintp


def test_sector_dim():
    assert sector_dim(4, 0) == 1
    assert sector_dim(4, 1) == 4
    assert sector_dim(4, 2) == 6
    assert sector_dim(4, 4) == 1


def test_reshape_and_flatten_roundtrip():
    norb = 4
    nelec = (2, 1)
    vec = np.arange(24, dtype=np.complex128)
    mat = reshape_state(vec, norb, nelec)
    assert mat.shape == (6, 4)
    out = flatten_state(mat)
    assert np.array_equal(out, vec)