from math import comb

from xquces.gcr.diagonal_rank import (
    diagonal_rank,
    spin_flip_orbit_count,
    spin_orbital_diagonal_features,
    spin_restricted_igcr_diagonal_features,
)


def test_spin_orbital_four_body_diagonal_spans_fixed_sector():
    norb = 4
    nelec = (2, 2)
    dim = comb(norb, nelec[0]) * comb(norb, nelec[1])

    features = spin_orbital_diagonal_features(norb, nelec, max_body=4)
    rank = diagonal_rank(features)

    assert rank.rank_mod_constant == dim - 1


def test_spin_balanced_four_body_diagonal_spans_spin_flip_orbits():
    norb = 4
    nelec = (2, 2)
    orbit_count = spin_flip_orbit_count(norb, nelec)

    features = spin_orbital_diagonal_features(
        norb,
        nelec,
        max_body=4,
        spin_balanced=True,
    )
    rank = diagonal_rank(features)

    assert rank.rank_mod_constant == orbit_count - 1


def test_current_spin_restricted_igcr4_diagonal_is_not_spin_balanced_complete():
    norb = 4
    nelec = (2, 2)
    orbit_count = spin_flip_orbit_count(norb, nelec)

    restricted = diagonal_rank(
        spin_restricted_igcr_diagonal_features(norb, nelec, max_body=4)
    )

    assert restricted.rank_mod_constant < orbit_count - 1
