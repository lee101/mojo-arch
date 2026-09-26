"""Parity tests for the ALPM version comparison.

The expected values in ``VERCMP_TABLE`` were produced by compiling libalpm's
``src/libalpm/version.c`` and calling ``alpm_pkg_vercmp`` on each pair, so this
is parity against the real C implementation rather than against a restatement
of it. ``test_vercmp.py::test_matches_libalpm_oracle`` then widens the check to
inputs the table does not list by diffing the kernel against a Python
transcription of that same C.
"""

from __future__ import annotations

import random

import pytest

import mojo_arch
from reference import alpm_pkg_vercmp

VERCMP_TABLE = [
    # vercmp(8) examples
    ('1', '2', -1),
    ('2', '1', 1),
    ('2.0-1', '1.7-6', 1),
    ('2.0', '2.0-13', 0),
    ('4.34', '1:001', -1),
    ('1.5-1', '1.5', 0),
    ('1.5-1', '1.5-2', -1),
    ('2:1.0-1', '1:3.6-1', 1),
    # the alphanumeric ordering chain from vercmp(8)
    ('1.0a', '1.0b', -1),
    ('1.0b', '1.0beta', -1),
    ('1.0beta', '1.0p', -1),
    ('1.0p', '1.0pre', -1),
    ('1.0pre', '1.0rc', -1),
    ('1.0rc', '1.0', -1),
    ('1.0', '1.0.a', -1),
    ('1.0.a', '1.0.1', -1),
    # the numeric ordering chain from vercmp(8)
    ('1', '1.0', -1),
    ('1.0', '1.1', -1),
    ('1.1', '1.1.1', -1),
    ('1.1.1', '1.2', -1),
    ('1.2', '2.0', -1),
    ('2.0', '3.0.0', -1),
    # alpm-pkgver(7) specification examples
    ('1.0.0', '1.1.0', -1),
    ('1.2.0', '1.foo.0', 1),
    ('foo.0', 'boo.0', 1),
    ('1.0', '1.0', 0),
    ('alpha0', 'beta0', -1),
    ('alpha1', 'alpha02', -1),
    ('1alpha0', '2alpha0', -1),
    ('alpha1', 'alpha.0', -1),
    ('1...0', '1.2', 1),
    ('1', '1.foo', -1),
    ('1.0', '1.0foo.2', 1),
    ('1.foo', '1.foo2', -1),
    ('0001', '1', 0),
    ('1', 'zeta', 1),
    ('b', 'a', 1),
    ('aab', 'aaa', 1),
    ('2', '1', 1),
    # separator-run length decides before the segments do
    ('1.0.1', '1.0..1', -1),
    ('1..0', '1.0', 1),
    ('1.-.0', '1.0', -1),
    ('1...0', '1....0', -1),
    ('1.0', '1.', 1),
    ('1.', '1', 1),
    # epochs
    ('1:1.0', '1.0', 1),
    ('2:1.0', '10:0.1', -1),
    ('0:1.0', '1.0', 0),
    ('00:1.0', '1.0', 0),
    (':1.0', '1.0', 0),
    ('1:1.0', '1:1.0-1', 0),
    ('1:1.0-1', '1:1.0', 0),
    # pkgrel, which only counts when both sides carry one
    ('1.0-0', '1.0', 0),
    ('1.0-0', '1.0-0', 0),
    ('1.0-01', '1.0-1', 0),
    ('1.0-1.2', '1.0-1.10', -1),
    ('1.0-1', '2.0-0', -1),
    # digit runs long enough to overflow an int, which is why libalpm stopped
    # converting segments to integers
    ('1' * 40, '1' * 41, -1),
    ('9' * 40, '1' + '0' * 40, -1),
    ('1' * 40 + '0', '1' * 40, 1),
    ('1.0alpha', '1.0beta', -1),
    ('1.0foo.2', '1.0', -1),
    ('1.0foo.2', '1.0.1', -1),
    ('1.0.0alpha.1', '1.0.0', -1),
    ('Z', 'a', -1),
    ('aB', 'ab', -1),
    ('a1', '1a', -1),
    ('', '1.0', -1),
    ('1.0', '', 1),
    ('', '', 0),
    ('1', '1', 0),
    # '~' and '^' are ordinary delimiters for libalpm, unlike RPM
    ('1.0~rc1', '1.0', 1),
    ('1.0^git1', '1.0', 1),
    ('1.0~rc1', '1.0~rc2', -1),
    ('1.0.1', '1.0.a', 1),
    ('10', '9', 1),
    ('1.10', '1.9', 1),
    ('1.0.10', '1.0.9', 1),
]


@pytest.mark.parametrize("a,b,expected", VERCMP_TABLE)
def test_libalpm_table(a, b, expected):
    assert mojo_arch.vercmp(a, b) == expected
    # the comparator must be antisymmetric on every case in the table
    assert mojo_arch.vercmp(b, a) == -expected


def _random_versions(seed, count):
    rng = random.Random(seed)
    alpha = "0123456789abcdefABCDEFxyzXYZ.:-+_~^ $@"
    out = []
    for _ in range(count):
        out.append("".join(rng.choice(alpha) for _ in range(rng.randint(0, 10))))
    return out


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_matches_libalpm_oracle(seed):
    """Diff the kernel against a Python transcription of libalpm's C."""
    versions = _random_versions(seed, 700)
    for a in versions:
        for b in versions[::7]:
            assert mojo_arch.vercmp(a, b) == alpm_pkg_vercmp(a, b), (a, b)


@pytest.mark.parametrize("seed", [4, 5])
def test_version_shaped_inputs_match_oracle(seed):
    """The same check on strings that look like real package versions."""
    rng = random.Random(seed)
    num = lambda: str(rng.choice([0, 1, 2, 9, 10, 99, 100, "007", 1234, 10**12]))
    alpha = lambda: rng.choice(["a", "b", "rc", "alpha", "beta", "Z", "zz", "x1", "1a", "a1"])

    def build():
        parts = [rng.choice([num(), alpha()]) for _ in range(rng.randint(1, 4))]
        s = rng.choice([".", "-", "_", "..", "+", ""]).join(parts)
        if rng.random() < 0.3:
            s = rng.choice(["~", "^", " "]) + s
        if rng.random() < 0.35:
            s = str(rng.choice([0, 1, 2, 10, 7])) + ":" + s
        if rng.random() < 0.5:
            s = s + "-" + str(rng.choice([0, 1, 2, 13, 99]))
        return s

    versions = [build() for _ in range(400)]
    for a in versions:
        for b in versions[::5]:
            assert mojo_arch.vercmp(a, b) == alpm_pkg_vercmp(a, b), (a, b)


def test_vercmp_is_antisymmetric_and_reflexive():
    """A comparator that is not antisymmetric is not a comparison at all."""
    versions = _random_versions(11, 160)
    for a in versions:
        assert mojo_arch.vercmp(a, a) == 0
    for a in versions:
        for b in versions:
            assert mojo_arch.vercmp(a, b) == -mojo_arch.vercmp(b, a)
            if mojo_arch.vercmp(a, b) == 0:
                assert mojo_arch.vercmp(b, a) == 0


def test_separator_run_makes_alpm_non_transitive():
    """The ALPM order is not transitive, and the kernel reproduces that.

    alpm-pkgver(7) says the separator-run rule was probably not added
    deliberately. It fires before the segment compare, so
    ``" :._@cb" < "^" < "6Cz410"`` while ``" :._@cb" > "6Cz410"``. A kernel
    that "fixed" this would no longer match libalpm, and one that moved the
    rule after the segment compare would fail the ``1...0`` table cases.
    """
    a, b, c = " :._@cb", "^", "6Cz410"
    assert mojo_arch.vercmp(a, b) == -1
    assert mojo_arch.vercmp(b, c) == -1
    assert mojo_arch.vercmp(a, c) == 1
    for pair in ((a, b), (b, c), (a, c)):
        assert mojo_arch.vercmp(*pair) == alpm_pkg_vercmp(*pair)


def test_zero_is_never_returned_as_something_else():
    """``vercmp`` must return exactly -1, 0 or 1; a raw strcmp would not."""
    versions = _random_versions(12, 120)
    for a in versions:
        for b in versions[::3]:
            assert mojo_arch.vercmp(a, b) in (-1, 0, 1)
