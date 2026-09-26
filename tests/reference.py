"""A direct transcription of libalpm's ``version.c`` used as a test oracle.

This is the same algorithm the Mojo kernel implements, written out in Python
straight from the C so the two can be cross-checked over inputs the fixed
table in ``test_vercmp.py`` does not enumerate. It is a test oracle only --
nothing in ``mojo_arch`` imports it.
"""

from __future__ import annotations

__all__ = ["alpm_pkg_vercmp"]


def _isalnum(c: str) -> bool:
    return c.isascii() and c.isalnum()


def _isdigit(c: str) -> bool:
    return "0" <= c <= "9"


def _isalpha(c: str) -> bool:
    return ("a" <= c <= "z") or ("A" <= c <= "Z")


def _rpmvercmp(a: str, b: str) -> int:
    if a == b:
        return 0

    one = ptr1 = 0
    two = ptr2 = 0
    n1, n2 = len(a), len(b)

    while one < n1 and two < n2:
        while one < n1 and not _isalnum(a[one]):
            one += 1
        while two < n2 and not _isalnum(b[two]):
            two += 1
        if not (one < n1 and two < n2):
            break

        if (one - ptr1) != (two - ptr2):
            return -1 if (one - ptr1) < (two - ptr2) else 1

        ptr1, ptr2 = one, two
        if _isdigit(a[ptr1]):
            while ptr1 < n1 and _isdigit(a[ptr1]):
                ptr1 += 1
            while ptr2 < n2 and _isdigit(b[ptr2]):
                ptr2 += 1
            isnum = True
        else:
            while ptr1 < n1 and _isalpha(a[ptr1]):
                ptr1 += 1
            while ptr2 < n2 and _isalpha(b[ptr2]):
                ptr2 += 1
            isnum = False

        if one == ptr1:
            return -1
        if two == ptr2:
            return 1 if isnum else -1

        if isnum:
            while one < ptr1 and a[one] == "0":
                one += 1
            while two < ptr2 and b[two] == "0":
                two += 1
            la, lb = ptr1 - one, ptr2 - two
            if la > lb:
                return 1
            if lb > la:
                return -1

        sa, sb = a[one:ptr1], b[two:ptr2]
        if sa != sb:
            return -1 if sa < sb else 1

        one, two = ptr1, ptr2

    if one >= n1 and two >= n2:
        return 0

    one_alpha = one < n1 and _isalpha(a[one])
    two_alpha = two < n2 and _isalpha(b[two])
    if (one >= n1 and not two_alpha) or one_alpha:
        return -1
    return 1


def _parse_evr(evr: str):
    i = 0
    while i < len(evr) and evr[i].isdigit():
        i += 1
    dash = evr.rfind("-", i)
    if i < len(evr) and evr[i] == ":":
        epoch = evr[:i] or "0"
        version = evr[i + 1 : dash] if dash >= 0 else evr[i + 1 :]
    else:
        epoch = "0"
        version = evr[:dash] if dash >= 0 else evr
    return epoch, version, (evr[dash + 1 :] if dash >= 0 else None)


def alpm_pkg_vercmp(a: str, b: str) -> int:
    """1 if ``a`` is newer than ``b``, -1 if older, 0 if equal."""
    if a == b:
        return 0
    epoch_a, ver_a, rel_a = _parse_evr(a)
    epoch_b, ver_b, rel_b = _parse_evr(b)
    ret = _rpmvercmp(epoch_a, epoch_b)
    if ret == 0:
        ret = _rpmvercmp(ver_a, ver_b)
        if ret == 0 and rel_a is not None and rel_b is not None:
            ret = _rpmvercmp(rel_a, rel_b)
    return ret
