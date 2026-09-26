"""ctypes bridge to the compiled Mojo kernels.

The shared library owns no memory. Every buffer crosses the C ABI as a 64-bit
address, so the argtypes below must stay ``c_int64`` for addresses; ``c_int``
truncates them and segfaults.
"""

from __future__ import annotations

import ctypes
import pathlib

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
_LIB_PATH = _ROOT / "dist" / "libmojo-arch.so"

_ADDR = ctypes.c_int64


def _load():
    if not _LIB_PATH.exists():
        raise RuntimeError(
            f"{_LIB_PATH} not found; run `bash build/build.sh` first"
        )
    lib = ctypes.CDLL(str(_LIB_PATH))
    lib.arch_vercmp.restype = ctypes.c_int32
    lib.arch_vercmp.argtypes = [_ADDR, _ADDR, _ADDR, _ADDR]
    lib.arch_vercmp_batch.restype = None
    lib.arch_vercmp_batch.argtypes = [_ADDR, _ADDR, _ADDR, _ADDR, _ADDR]
    lib.arch_sort_indices.restype = None
    lib.arch_sort_indices.argtypes = [_ADDR, _ADDR, _ADDR, _ADDR, _ADDR]
    lib.arch_max_index.restype = ctypes.c_int32
    lib.arch_max_index.argtypes = [_ADDR, _ADDR, _ADDR]
    return lib


lib = _load()


def _as_bytes(version) -> bytes:
    if isinstance(version, bytes):
        return version
    return str(version).encode("utf-8")


def _zero_copy_address(payload: bytes) -> int:
    """Address of a bytes object's own buffer, without copying it.

    ``bytes`` is contiguous and NUL terminated, so the kernel may read one
    past the declared length without running off the allocation. The
    ``c_char_p`` keeps the bytes object alive for as long as the pointer
    exists, which is why the callers hold on to their local names.
    """
    return ctypes.cast(ctypes.c_char_p(payload), ctypes.c_void_p).value


def vercmp(a, b) -> int:
    """1 if ``a`` is newer than ``b``, -1 if older, 0 if equal."""
    ab = _as_bytes(a)
    bb = _as_bytes(b)
    return int(
        lib.arch_vercmp(
            ctypes.c_int64(_zero_copy_address(ab)),
            len(ab),
            ctypes.c_int64(_zero_copy_address(bb)),
            len(bb),
        )
    )


def _csr(strings):
    """Pack version strings into one blob plus an Int64 CSR offset table."""
    encoded = [_as_bytes(s) for s in strings]
    offsets = np.empty(len(encoded) + 1, dtype=np.int64)
    pos = 0
    for i, item in enumerate(encoded):
        offsets[i] = pos
        pos += len(item)
    offsets[len(encoded)] = pos
    return b"".join(encoded), offsets


def _blob_buffer(blob: bytes):
    return ctypes.create_string_buffer(blob, max(len(blob), 1))


def vercmp_pairs(lhs, rhs) -> np.ndarray:
    """Compare ``n`` version pairs in one call; returns an int32 array.

    Both sides share one blob because the kernel indexes a single address
    space, so the right-hand offsets are shifted past the left-hand bytes.
    """
    if len(lhs) != len(rhs):
        raise ValueError("lhs and rhs must be the same length")
    n = len(lhs)
    if n == 0:
        return np.empty(0, dtype=np.int32)
    left_blob, loff = _csr(lhs)
    right_blob, roff = _csr(rhs)
    blob = left_blob + right_blob
    roff = roff + len(left_blob)
    dst = np.empty(n, dtype=np.int32)
    buf = _blob_buffer(blob)
    lib.arch_vercmp_batch(
        ctypes.c_int64(ctypes.addressof(buf)),
        loff.ctypes.data,
        roff.ctypes.data,
        n,
        dst.ctypes.data,
    )
    return dst


def sort_indices(versions) -> np.ndarray:
    """Indices that put ``versions`` in ascending ALPM order (stable)."""
    n = len(versions)
    if n == 0:
        return np.empty(0, dtype=np.int32)
    blob, off = _csr(versions)
    idx = np.empty(n, dtype=np.int32)
    scratch = np.empty(n, dtype=np.int32)
    buf = _blob_buffer(blob)
    lib.arch_sort_indices(
        ctypes.c_int64(ctypes.addressof(buf)),
        off.ctypes.data,
        n,
        idx.ctypes.data,
        scratch.ctypes.data,
    )
    return idx


def max_index(versions) -> int:
    """Index of the newest of ``versions``; the lowest index wins a tie."""
    n = len(versions)
    if n == 0:
        return -1
    blob, off = _csr(versions)
    buf = _blob_buffer(blob)
    return int(lib.arch_max_index(ctypes.c_int64(ctypes.addressof(buf)), off.ctypes.data, n))
