# mojo-arch

`mojo-arch` is a Mojo port of the compute side of `arch`, the small library
that compares Arch Linux package versions. The comparison is the whole point of
that package, and it is the one place with a real inner loop: a byte-wise walk
over two version strings, repeated thousands of times when a tool sorts a
package list or picks the newest candidate.

The algorithm is libalpm's. `src/kernels.mojo` is a port of
`src/libalpm/version.c`: `parseEVR` splits `[epoch:]version[-release]`, and
`rpmvercmp` walks the two strings segment by segment. Nothing about it is
approximated.

The Python package is `mojo_arch`, so it installs alongside the real `arch`.

```python
import mojo_arch

mojo_arch.vercmp("1:2.0-1", "1:1.9-9")     #  1
mojo_arch.vercmp("1.5-1", "1.5")           #  0  (pkgrel only counts on both sides)
mojo_arch.Ver("2:1.0-3").pkgver            # '1.0'
mojo_arch.newest(["1.0-1", "1.0-2", "0.9"]) # '1.0-2'
mojo_arch.sort_versions(["1.0-1", "1.0-2", "0.9"])
```

## Covered subset

| area | implemented API | where |
| --- | --- | --- |
| Comparison | `vercmp`, `compare`, `strict_cmp`, `Ver` / `Version` ordering | `arch_vercmp` |
| Batch comparison | `vercmp_pairs(lhs, rhs)` -> `int32[n]` | `arch_vercmp_batch` |
| Ordering a list | `sort_versions(versions)` | `arch_sort_indices` |
| Newest of many | `newest(versions)` | `arch_max_index` |
| Version parts | `Ver.epoch` / `.pkgver` / `.pkgrel`, `version_split` | Python |
| Package ids | `linux_package_id(uid, pkgname, pkgver, pkgrel)` | Python |

Not implemented, and not ported: anything that is not comparison arithmetic.
There is no dependency resolution, no version constraint parsing, no package
database, and no `linux_distribution` probing. This package has no compute core
beyond the comparator, and none was invented to pad the coverage.

The numerical contract is integer-only. Every result is exactly `-1`, `0` or
`1`; nothing is floating point, so nothing is approximate.

## Parity

The real `arch` is not importable from this toolchain's test venv: the name on
PyPI belongs to ARCH 8.0.0, an econometrics package, and that is what is
installed. Parity is therefore established against the specification the
upstream library implements, not against an installed copy of it:

* `tests/test_vercmp.py::test_libalpm_table` checks 78 cases whose expected
  values were produced by **compiling libalpm's `version.c` and calling
  `alpm_pkg_vercmp`** on each pair. The cases include every example in
  `vercmp(8)` and `alpm-pkgver(7)`, the separator-run rule, epoch handling,
  pkgrel handling, 40-digit numeric runs, and the `~` / `^` behaviour.
* `tests/reference.py` is a Python transcription of the same C file, and the
  randomised tests diff the kernel against it over thousands of inputs drawn
  from a byte alphabet and from a generator shaped like real package versions.
* Properties that a plausible bug would break are asserted directly:
  antisymmetry and reflexivity, that the result is always in `{-1, 0, 1}`, and
  that the batch kernel agrees with the scalar one element for element.

Two ALPM behaviours are genuinely surprising, so both are pinned by their own
tests rather than smoothed over. Neither is a bug in this port:

* the separator-run rule fires *before* the segment compare, which makes the
  order non-transitive (`" :._@cb" < "^" < "6Cz410"` but
  `" :._@cb" > "6Cz410"`). `alpm-pkgver(7)` says the rule's motivation is
  unclear and it may not have been added deliberately.
* a pkgrel is only compared when **both** versions carry one, so `"1.0"` ties
  with `"1.0-13"` and with `"1.0-1"`, while `"1.0-13"` loses to `"1.0-1"`.

The second consequence is that `sort_versions` cannot be compared against
Python's `sorted` on a corpus that mixes versions with and without a pkgrel:
on such input the two sort implementations are free to disagree, and both are
correct. The sort tests use a corpus on which the order really is total, and
assert that precondition before comparing.

## Install

```bash
pixi install
pixi run build
pixi run test
```

`pixi run build` produces `dist/libmojo-arch.so`. Set `PYTHONPATH=python` when
using the package outside a Pixi task. The tests need the shared toolchain, not
a private `.pixi` environment:

```bash
source /nvme0n1-disk/mojo-toolchain/activate.sh
bash build/build.sh
PYTHONPATH=python /nvme0n1-disk/mojo-toolchain/testvenv/bin/python -m pytest tests -q
```

## Performance

Best-of-seven wall clock in one process, against the pure-Python ALPM
`vercmp` in `tests/reference.py` -- the same algorithm, so this measures the
compiled inner loop and nothing else. Every case checks its result against
that reference before timing.

| case | python reference | mojo-arch | result |
| --- | ---: | ---: | ---: |
| `vercmp_pairs` n=20000 | 82.95 ms | 29.55 ms | 2.81x faster |
| `sort_versions` n=2000 | 176.68 ms | 11.42 ms | 15.47x faster |
| `newest` n=200000 | 1140.66 ms | 188.86 ms | 6.04x faster |
| `vercmp` scalar x20000 | 88.75 ms | 458.11 ms | 0.19x, a loss |

Reproduce with:

```bash
pixi run bench
```

The scalar case is a genuine loss and is reported as one. A single comparison
is a few dozen byte operations; the ctypes call costs about a microsecond, so
one call per comparison is pure overhead. The batch entry point amortises
that over 20000 comparisons and is where the port pays off. If you only ever
compare one pair at a time in a loop, the API will not help you.

The box this was measured on is shared, so repeated runs vary by two to three
times in either direction. The ordering of the cases is stable.

## How it works

All kernels live in `src/kernels.mojo`, one compilation unit, because shared
library build cost is largely fixed. `build/build.sh` compiles it with
`mojo build --emit shared-lib` into `dist/libmojo-arch.so`.

The `python/mojo_arch` layer owns every array. It packs the version strings
into a single byte blob with an `int64` CSR offset table -- the layout a
`str` would use -- and passes the blob's address across the C ABI as a 64-bit
integer. The Mojo side rebuilds the pointer inside the function body
(`Pointer[UInt8, AnyOrigin[mut=True]](unsafe_from_address=addr)`), because
`@export` rejects parametric functions and an inferred pointer origin would
make the symbol parametric.

There is no SIMD in these kernels and there should not be: the comparison is
branch-heavy byte scanning, not arithmetic, so the vector units have nothing to
do. The three exported entry points are a scalar comparison, a batch
comparison loop, and a stable merge sort over an index permutation whose
comparator is the same `vercmp`. The sort takes its scratch buffer from the
caller, which is why `sort_indices` allocates two `int32` arrays instead of one.

## License

MIT
