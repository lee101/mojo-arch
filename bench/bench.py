"""Correctness-gated benchmark for mojo-arch.

Every case checks the compiled result against the pure-Python ALPM
``vercmp`` in ``tests/reference.py`` before timing, so a regression in the Mojo
kernels shows up as a correctness failure rather than a suspiciously good
number. The baseline is that same pure-Python implementation, which is the
fair comparison: this port does not beat a different algorithm, it runs the
same one in compiled code.
"""

from __future__ import annotations

import functools
import itertools
import pathlib
import random
import sys
import time

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "python"))
sys.path.insert(0, str(_ROOT / "tests"))

import numpy as np  # noqa: E402

import mojo_arch  # noqa: E402
from mojo_arch import _lib  # noqa: E402
from reference import alpm_pkg_vercmp  # noqa: E402


def _time(fn, repeats=7):
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


_WORDS = ["1.0", "1.0.1", "1.0-1", "1.0-13", "0.9.9", "1.10", "1.9", "10",
          "9", "2:1.0-1", "1.0rc", "1.0a", "1.0.0alpha.1", "3.0.0", "1:0.1"]

# ALPM's order is not a total order over every real package version: a
# separator run decides before the segments do, and a missing pkgrel is not
# compared at all. Mixing a version with and without a pkgrel therefore makes
# "1.0" tie with "1.0-13" while "1.0-13" loses to "1.0-1", and two different
# sort implementations are then free to disagree. The sort case uses a corpus
# where the order really is total, and checks that before timing.
_SORT_WORDS = ["0.9.9", "1.0.1", "1.0a", "1.0rc", "1.9", "1.10",
               "1:0.1", "2:1.0-1", "3.0.0", "9", "10"]


def _corpus(words, seed, count):
    rng = random.Random(seed)
    return [rng.choice(words) for _ in range(count)]


def bench_vercmp_pairs(n=20000):
    """Batch comparison against a Python list comprehension of the same work."""
    lhs = _corpus(_WORDS, 1, n)
    rhs = [v[::-1] for v in lhs]
    got = _lib.vercmp_pairs(lhs, rhs)
    expect = np.array(
        [alpm_pkg_vercmp(a, b) for a, b in zip(lhs, rhs)], dtype=np.int32
    )
    assert np.array_equal(got, expect), "vercmp_pairs mismatch"

    python_time = _time(
        lambda: [alpm_pkg_vercmp(a, b) for a, b in zip(lhs, rhs)], 3
    )
    mojo_time = _time(lambda: _lib.vercmp_pairs(lhs, rhs))
    return f"vercmp_pairs n={n}", python_time, mojo_time


def bench_sort_versions(n=2000):
    """Full sort of a shuffled list, against a stable Python sort by key."""
    for a, b, c in itertools.product(_SORT_WORDS, repeat=3):
        if alpm_pkg_vercmp(a, b) <= 0 and alpm_pkg_vercmp(b, c) <= 0:
            assert alpm_pkg_vercmp(a, c) <= 0, (a, b, c)

    versions = _corpus(_SORT_WORDS, 4, n)
    ordered = mojo_arch.sort_versions(versions)
    expect = sorted(versions, key=functools.cmp_to_key(alpm_pkg_vercmp))
    assert ordered == expect, "sort mismatch"
    return (
        f"sort_versions n={n}",
        _time(lambda: sorted(versions, key=functools.cmp_to_key(alpm_pkg_vercmp)), 3),
        _time(lambda: mojo_arch.sort_versions(versions)),
    )


def bench_newest(n=200000):
    """Newest-of-N reduction, against ``max`` with a comparison key."""
    versions = _corpus(_WORDS, 2, n)
    winner = _lib.max_index(versions)
    expect = functools.reduce(
        lambda acc, i: acc
        if alpm_pkg_vercmp(versions[acc], versions[i]) >= 0
        else i,
        range(n),
        0,
    )
    assert versions[winner] == versions[expect], "newest mismatch"
    assert mojo_arch.newest(versions) == versions[winner]

    python_time = _time(
        lambda: max(versions, key=functools.cmp_to_key(alpm_pkg_vercmp)), 3
    )
    mojo_time = _time(lambda: _lib.max_index(versions))
    return f"newest n={n}", python_time, mojo_time


def bench_scalar_vercmp(n=20000):
    """The single-pair entry point, which pays one ctypes call per comparison."""
    versions = _corpus(_WORDS, 3, n)

    def python_loop():
        acc = 0
        for a in versions:
            acc += alpm_pkg_vercmp(a, a[::-1])
        return acc

    def mojo_loop():
        acc = 0
        for a in versions:
            acc += mojo_arch.vercmp(a, a[::-1])
        return acc

    assert python_loop() == mojo_loop()
    return f"vercmp scalar x{n}", _time(python_loop, 3), _time(mojo_loop, 3)


def main():
    print(f"{'case':<24}{'python reference':>18}{'mojo-arch':>14}{'ratio':>10}")
    print("-" * 66)
    for fn in (
        bench_vercmp_pairs,
        bench_sort_versions,
        bench_newest,
        bench_scalar_vercmp,
    ):
        label, ref, got = fn()
        ratio = ref / got if got else float("nan")
        print(f"{label:<24}{ref * 1e3:>16.2f}ms{got * 1e3:>12.2f}ms{ratio:>9.2f}x")


if __name__ == "__main__":
    main()
