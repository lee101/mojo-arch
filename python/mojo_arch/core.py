"""Arch Linux version objects built on the Mojo ``vercmp`` kernel.

The comparison itself -- the ``[epoch:]pkgver[-pkgrel]`` split and the
segment walk -- runs in ``src/kernels.mojo``. What stays here is the string
handling around it: splitting a version into its parts for display, and
assembling a distribution package id.
"""

from __future__ import annotations

import re
from functools import total_ordering

from ._lib import max_index, sort_indices, vercmp, vercmp_pairs

__all__ = [
    "Ver",
    "Version",
    "compare",
    "linux_package_id",
    "newest",
    "sort_versions",
    "strict_cmp",
    "vercmp",
    "vercmp_pairs",
    "version_split",
]

_EPOCH_RE = re.compile(r"^(\d+):(.*)$", re.DOTALL)
_SPLIT_RE = re.compile(r"(\d+|[A-Za-z]+)")


def _parse(full: str) -> tuple[int, str, str | None]:
    """Split ``[epoch:]pkgver[-pkgrel]`` the way libalpm's parseEVR does."""
    m = _EPOCH_RE.match(full)
    if m:
        epoch = int(m.group(1))
        rest = m.group(2)
    else:
        epoch = 0
        rest = full
    # libalpm takes the LAST dash of the version part as the pkgrel separator.
    dash = rest.rfind("-")
    if dash < 0:
        return epoch, rest, None
    return epoch, rest[:dash], rest[dash + 1 :]


@total_ordering
class Ver:
    """One ``epoch:pkgver-pkgrel`` version string.

    Ordering is total and delegates to the Mojo kernel, so ``Ver`` instances
    sort exactly the way ``vercmp`` says. Because ``vercmp`` treats ``1.0`` and
    ``1.0-0`` as equal, instances are deliberately unhashable: no hash can
    respect that equivalence.
    """

    __slots__ = ("fullver", "epoch", "pkgver", "pkgrel")

    def __init__(self, fullver):
        if isinstance(fullver, Ver):
            fullver = fullver.fullver
        if isinstance(fullver, bytes):
            fullver = fullver.decode("utf-8")
        if not isinstance(fullver, str):
            raise TypeError(f"version must be a string, got {type(fullver).__name__}")
        self.fullver = fullver
        self.epoch, self.pkgver, self.pkgrel = _parse(fullver)

    def _cmp(self, other) -> int:
        if isinstance(other, Ver):
            return vercmp(self.fullver, other.fullver)
        if isinstance(other, str):
            return vercmp(self.fullver, other)
        return NotImplemented

    def __eq__(self, other):
        return self._cmp(other) == 0

    def __lt__(self, other):
        return self._cmp(other) < 0

    def __str__(self):
        return self.fullver

    def __repr__(self):
        return f"Ver({self.fullver!r})"


Version = Ver


def compare(a, b) -> int:
    """``-1`` / ``0`` / ``1`` comparing two versions, ``Ver`` or ``str``."""
    return vercmp(str(a), str(b))


def strict_cmp(a, b) -> int:
    """Alias of :func:`compare`; the name the upstream module exports."""
    return compare(a, b)


def version_split(version) -> list:
    """Split a version's ``pkgver`` into its numeric and alphabetic tokens.

    Numeric runs become ``int`` and alphabetic runs stay ``str``, which is the
    representation a caller wants for display; ordering decisions must go
    through :func:`vercmp`, never through this list.
    """
    tokens = _SPLIT_RE.findall(Ver(version).pkgver)
    return [int(t) if t.isdigit() else t for t in tokens]


def sort_versions(versions) -> list:
    """Sort versions oldest to newest, using the compiled sort."""
    items = list(versions)
    if not items:
        return items
    return [items[i] for i in sort_indices([str(v) for v in items])]


def newest(versions):
    """The newest of ``versions``, or ``None`` for an empty input."""
    items = list(versions)
    if not items:
        return None
    return items[max_index([str(v) for v in items])]


def linux_package_id(uid, pkgname=None, pkgver=None, pkgrel=None) -> str:
    """Build a ``name-pkgver-pkgrel`` package id from the pieces of one.

    A bare ``uid`` string is returned unchanged. When any part is missing the
    id is assembled from what was given, and an absent pkgrel is left off
    rather than defaulted to ``0``.
    """
    if pkgname is None and pkgver is None and pkgrel is None:
        return str(uid)
    parts = [str(uid) if pkgname is None else str(pkgname)]
    if pkgver is not None:
        parts.append(str(pkgver))
    if pkgrel is not None:
        parts.append(str(pkgrel))
    return "-".join(parts)
