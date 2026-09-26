"""mojo-arch: Arch Linux version comparison with the ALPM kernel in Mojo.

Installable alongside the real ``arch`` package, whose comparison semantics it
reimplements: the ``[epoch:]pkgver[-pkgrel]`` split and the segment walk of
libalpm's ``alpm_pkg_vercmp`` live in ``src/kernels.mojo``.
"""

from .core import (
    Ver,
    Version,
    compare,
    linux_package_id,
    newest,
    sort_versions,
    strict_cmp,
    vercmp,
    vercmp_pairs,
    version_split,
)

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
__version__ = "0.1.0"
