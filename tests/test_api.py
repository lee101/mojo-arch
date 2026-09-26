"""Tests for the batch kernels and the Python-facing version objects."""

from __future__ import annotations

import functools
import random

import numpy as np
import pytest

import mojo_arch
from mojo_arch import _lib
from reference import alpm_pkg_vercmp

_RANDOM_VERSIONS = [
    "1.0", "1.0-1", "1.0.1", "1.0a", "1.0~rc1", "2:1.0-1", "1.0-13",
    "0.9.9", "10", "9", "1.10", "1.9", "1:0.1", "3.0.0", "1.0.0alpha.1",
]

# A corpus on which the ALPM order is a genuine total order. Two families of
# real package versions have to stay out of it: multi-character separator
# runs, and mixing a version with and without a pkgrel. vercmp(8) compares the
# pkgrel only when both sides carry one, so "1.0" ties with "1.0-1" while
# "1.0" beats "1.0-13". Both quirks are pinned by their own tests, but a sort
# cannot be compared against Python's sorted on inputs where two different
# sort implementations are free to disagree.
_SORTABLE = [
    "0.9.9", "1.0.1", "1.0a", "1.0rc", "1.9", "1.10",
    "1:0.1", "2:1.0-1", "3.0.0", "9", "10",
]


def test_sortable_corpus_is_a_total_order():
    """Precondition for the sort test: the oracle must order this corpus totally."""
    for a in _SORTABLE:
        for b in _SORTABLE:
            for c in _SORTABLE:
                if alpm_pkg_vercmp(a, b) <= 0 and alpm_pkg_vercmp(b, c) <= 0:
                    assert alpm_pkg_vercmp(a, c) <= 0, (a, b, c)


def test_vercmp_pairs_matches_scalar():
    """The batch kernel must agree with the scalar one element for element."""
    lhs = _RANDOM_VERSIONS * 4
    rhs = list(reversed(_RANDOM_VERSIONS)) * 4
    got = _lib.vercmp_pairs(lhs, rhs)
    assert got.dtype == np.int32
    assert got.shape == (len(lhs),)
    for a, b, value in zip(lhs, rhs, got):
        assert int(value) == mojo_arch.vercmp(a, b), (a, b)


def test_vercmp_pairs_self_comparison_is_zero():
    """Comparing every version with itself must yield all zeros."""
    got = _lib.vercmp_pairs(_RANDOM_VERSIONS, _RANDOM_VERSIONS)
    assert np.array_equal(got, np.zeros(len(_RANDOM_VERSIONS), dtype=np.int32))


def test_vercmp_pairs_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        _lib.vercmp_pairs(["1.0", "2.0"], ["1.0"])


def test_vercmp_pairs_empty():
    assert _lib.vercmp_pairs([], []).shape == (0,)


def test_vercmp_pairs_on_a_long_corpus():
    """Both CSR sides share one blob, so an offset shift error must show up."""
    rng = random.Random(21)
    pool = _RANDOM_VERSIONS * 8
    lhs = [rng.choice(pool) for _ in range(300)]
    rhs = [rng.choice(pool) for _ in range(300)]
    got = _lib.vercmp_pairs(lhs, rhs)
    for a, b, value in zip(lhs, rhs, got):
        assert int(value) == alpm_pkg_vercmp(a, b), (a, b)


def test_sort_indices_matches_python_sort():
    """The compiled sort against Python's stable sort under the oracle."""
    versions = _SORTABLE * 6
    random.Random(3).shuffle(versions)
    expected = sorted(
        range(len(versions)),
        key=functools.cmp_to_key(
            lambda i, j: alpm_pkg_vercmp(versions[i], versions[j])
        ),
    )
    assert _lib.sort_indices(versions).tolist() == expected


def test_sort_indices_is_a_permutation():
    versions = _SORTABLE * 5
    random.Random(13).shuffle(versions)
    order = _lib.sort_indices(versions).tolist()
    assert sorted(order) == list(range(len(versions)))


def test_sort_indices_is_locally_sorted():
    """No adjacent pair in the output may be inverted."""
    versions = _SORTABLE * 5 + _RANDOM_VERSIONS
    order = _lib.sort_indices(versions).tolist()
    for a, b in zip(order, order[1:]):
        assert mojo_arch.vercmp(versions[a], versions[b]) <= 0, (a, b)


def test_sort_indices_is_stable_on_ties():
    """Ties keep their original order, which is what makes the sort total."""
    tied = ["1.0"] * 7
    assert _lib.sort_indices(tied).tolist() == list(range(7))
    mixed = ["1.0", "2.0", "1.0", "1.0-0", "2.0-0", "1.0"]
    assert _lib.sort_indices(mixed).tolist() == [0, 2, 3, 5, 1, 4]


def test_sort_indices_reorders_a_long_list():
    """A longer, deliberately reversed list: the kernel must fully sort it."""
    versions = [f"1.0.{i}" for i in range(60)][::-1]
    assert mojo_arch.sort_versions(versions) == [f"1.0.{i}" for i in range(60)]


def test_max_index_picks_the_newest():
    assert _lib.max_index(["1.0", "1.1", "1.10", "1.2"]) == 2
    assert _lib.max_index(["2:0.1", "1:9.9", "0.1"]) == 0
    assert _lib.max_index([]) == -1


def test_max_index_lowest_index_wins_a_tie():
    assert _lib.max_index(["1.0", "1.0", "1.0"]) == 0
    assert _lib.max_index(["0.1", "1.0", "1.0", "2.0", "2.0"]) == 3


def test_ver_splits_epoch_pkgver_and_pkgrel():
    assert mojo_arch.Ver("2:1.0-3").epoch == 2
    assert mojo_arch.Ver("2:1.0-3").pkgver == "1.0"
    assert mojo_arch.Ver("2:1.0-3").pkgrel == "3"
    assert mojo_arch.Ver("1.0").pkgrel is None
    assert mojo_arch.Ver("1.0").epoch == 0
    assert mojo_arch.Ver("1.0-2-3").pkgver == "1.0-2"
    assert mojo_arch.Ver("1.0-2-3").pkgrel == "3"


def test_ver_ordering_uses_the_kernel():
    order = ["1.0.1", "1.0", "1.0-1", "0.9", "2:0.1", "1.0rc"]
    assert [str(v) for v in sorted(mojo_arch.Ver(v) for v in order)] == [
        "0.9", "1.0rc", "1.0", "1.0-1", "1.0.1", "2:0.1",
    ]


def test_ver_equality_uses_vercmp():
    assert mojo_arch.Ver("1.0") == mojo_arch.Ver("1.0-0")
    assert mojo_arch.Ver("1.0") == "1.0-0"
    assert mojo_arch.Ver("1.0.1") > mojo_arch.Ver("1.0")
    assert mojo_arch.Ver("1.0rc") < mojo_arch.Ver("1.0")
    assert mojo_arch.Ver("1.0") != "1.0.1"
    with pytest.raises(TypeError):
        mojo_arch.Ver(3)


def test_pkgrel_absence_breaks_transitivity():
    """vercmp(8) only compares the pkgrel when both sides have one.

    That makes ``1.0`` tie with both ``1.0-13`` and ``1.0-1``, yet
    ``1.0-13`` loses to ``1.0-1`` and the order is not transitive across that
    boundary. A kernel that defaulted a missing pkgrel to 0 -- which older
    libalpm did -- would fail this.
    """
    assert mojo_arch.vercmp("1.0-13", "1.0") == 0
    assert mojo_arch.vercmp("1.0", "1.0-1") == 0
    assert mojo_arch.vercmp("1.0-13", "1.0-1") == 1
    for pair in (("1.0-13", "1.0"), ("1.0", "1.0-1"), ("1.0-13", "1.0-1")):
        assert mojo_arch.vercmp(*pair) == alpm_pkg_vercmp(*pair)


def test_compare_and_strict_cmp_are_the_kernel():
    assert mojo_arch.compare("1.0", "2.0") == -1
    assert mojo_arch.strict_cmp(mojo_arch.Ver("1.0"), mojo_arch.Ver("2.0")) == -1
    assert mojo_arch.strict_cmp("1.0-1", "1.0-1") == 0


def test_newest_and_sort_versions_on_a_package_list():
    versions = ["1.0-1", "1.0-2", "0.9.9", "1:0.1", "1.0-2-1"]
    assert mojo_arch.newest(versions) == "1:0.1"
    assert mojo_arch.sort_versions(versions) == [
        "0.9.9", "1.0-1", "1.0-2", "1.0-2-1", "1:0.1",
    ]
    assert mojo_arch.newest([]) is None
    assert mojo_arch.sort_versions([]) == []


def test_sort_versions_preserves_the_original_objects():
    wrapped = [mojo_arch.Ver(v) for v in ["2.0", "1.0", "1.5"]]
    ordered = mojo_arch.sort_versions(wrapped)
    assert all(isinstance(v, mojo_arch.Ver) for v in ordered)
    assert [str(v) for v in ordered] == ["1.0", "1.5", "2.0"]


def test_version_split_types():
    assert mojo_arch.version_split("1.0.0") == [1, 0, 0]
    assert mojo_arch.version_split("1.0.0alpha.1") == [1, 0, 0, "alpha", 1]
    assert mojo_arch.version_split("1:2.0rc-3") == [2, 0, "rc"]
    assert mojo_arch.version_split(mojo_arch.Ver("3.1.4")) == [3, 1, 4]


def test_linux_package_id():
    assert mojo_arch.linux_package_id("bash") == "bash"
    assert mojo_arch.linux_package_id("bash", pkgver="5.2") == "bash-5.2"
    assert (
        mojo_arch.linux_package_id("bash", pkgver="5.2", pkgrel="1")
        == "bash-5.2-1"
    )
