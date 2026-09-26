"""ALPM ``alpm_pkg_vercmp`` for Arch Linux version comparison.

A direct port of libalpm's ``version.c``: ``parseEVR`` splits
``[epoch:]version[-release]``, and ``rpmvercmp`` walks the two version strings
segment by segment. The whole algorithm is here -- the epoch/pkgver/pkgrel
order, the separator-run length rule, leading-zero stripping, the
numeric-beats-alpha tie break and the final showdown -- along with the batch
comparison, the stable sort and the newest-of-N reduction built on top of it.
Buffers cross the C ABI as 64-bit addresses because ``@export`` rejects
parametric functions and an inferred pointer origin would make a symbol
parametric.
"""

comptime UPtr = Pointer[UInt8, AnyOrigin[mut=True]]
comptime I64Ptr = Pointer[Int64, AnyOrigin[mut=True]]
comptime I32Ptr = Pointer[Int32, AnyOrigin[mut=True]]

comptime CH_ZERO = UInt8(0x30)
comptime CH_NINE = UInt8(0x39)
comptime CH_UPPER_A = UInt8(0x41)
comptime CH_UPPER_Z = UInt8(0x5A)
comptime CH_LOWER_A = UInt8(0x61)
comptime CH_LOWER_Z = UInt8(0x7A)
comptime CH_COLON = UInt8(0x3A)
comptime CH_DASH = UInt8(0x2D)


def uptr(addr: Int) -> UPtr:
    return UPtr(unsafe_from_address=addr)


def i64p(addr: Int) -> I64Ptr:
    return I64Ptr(unsafe_from_address=addr)


def i32p(addr: Int) -> I32Ptr:
    return I32Ptr(unsafe_from_address=addr)


def is_digit(c: UInt8) -> Bool:
    return c >= CH_ZERO and c <= CH_NINE


def is_alpha(c: UInt8) -> Bool:
    return (c >= CH_UPPER_A and c <= CH_UPPER_Z) or (
        c >= CH_LOWER_A and c <= CH_LOWER_Z
    )


def is_alnum(c: UInt8) -> Bool:
    return is_digit(c) or is_alpha(c)


def skip_seps(p: UPtr, n: Int, mut i: Int) -> Int:
    while i < n and not is_alnum(p[unsafe_offset=i]):
        i += 1
    return i


def cmp_bytes(a: UPtr, ao: Int, an: Int, b: UPtr, bo: Int, bn: Int) -> Int32:
    """memcmp-style compare of two byte ranges; shorter loses on a tie."""
    var n = an
    if bn < n:
        n = bn
    for k in range(n):
        var x = a[unsafe_offset=ao + k]
        var y = b[unsafe_offset=bo + k]
        if x != y:
            return Int32(-1) if x < y else Int32(1)
    if an < bn:
        return Int32(-1)
    if bn < an:
        return Int32(1)
    return Int32(0)


def rpmvercmp_at(
    a: UPtr, ao: Int, a_end: Int, b: UPtr, bo: Int, b_end: Int
) -> Int32:
    """libalpm's ``rpmvercmp``: 1 if a is newer, -1 if b is newer, 0 if equal.

    ``one``/``two`` first hold the start of the current segment and end up
    holding its end, which is where the separator (or the end of the string)
    lives; ``ptr1``/``ptr2`` walk to the end of the current segment. That is
    the pointer juggling the C code does, and the separator-run rule needs it:
    the run is measured from where the previous segment ended.
    """
    var one = ao
    var two = bo
    var ptr1 = ao
    var ptr2 = bo

    while one < a_end and two < b_end:
        one = skip_seps(a, a_end, one)
        two = skip_seps(b, b_end, two)
        if not (one < a_end and two < b_end):
            break

        var sep1 = one - ptr1
        var sep2 = two - ptr2
        if sep1 != sep2:
            return Int32(-1) if sep1 < sep2 else Int32(1)

        ptr1 = one
        ptr2 = two

        var isnum = is_digit(a[unsafe_offset=ptr1])
        if isnum:
            while ptr1 < a_end and is_digit(a[unsafe_offset=ptr1]):
                ptr1 += 1
            while ptr2 < b_end and is_digit(b[unsafe_offset=ptr2]):
                ptr2 += 1
        else:
            while ptr1 < a_end and is_alpha(a[unsafe_offset=ptr1]):
                ptr1 += 1
            while ptr2 < b_end and is_alpha(b[unsafe_offset=ptr2]):
                ptr2 += 1

        if one == ptr1:
            return Int32(-1)
        if two == ptr2:
            return Int32(1) if isnum else Int32(-1)

        var sa = one
        var sb = two
        if isnum:
            while sa < ptr1 and a[unsafe_offset=sa] == CH_ZERO:
                sa += 1
            while sb < ptr2 and b[unsafe_offset=sb] == CH_ZERO:
                sb += 1
            var la = ptr1 - sa
            var lb = ptr2 - sb
            if la > lb:
                return Int32(1)
            if lb > la:
                return Int32(-1)

        var rc = cmp_bytes(a, sa, ptr1 - sa, b, sb, ptr2 - sb)
        if rc != 0:
            return rc

        one = ptr1
        two = ptr2

    if one >= a_end and two >= b_end:
        return Int32(0)

    # Final showdown. A remaining alpha segment never beats an empty string:
    # one exhausted against a non-alpha remainder means two wins, and a
    # remaining alpha on either side always loses.
    var one_alpha = (one < a_end) and is_alpha(a[unsafe_offset=one])
    var two_alpha = (two < b_end) and is_alpha(b[unsafe_offset=two])
    if (one >= a_end and not two_alpha) or one_alpha:
        return Int32(-1)
    return Int32(1)


def vercmp(a_addr: Int, a_len: Int, b_addr: Int, b_len: Int) -> Int32:
    """alpm_pkg_vercmp: 1 if a is newer, -1 if b is newer, 0 if equal."""
    var a = uptr(a_addr)
    var b = uptr(b_addr)
    var a_end = a_len
    var b_end = b_len

    # parseEVR: s walks past the leading digits; a ':' at s means the prefix
    # was an epoch. The pkgrel is the last '-' at or after s.
    var sa = 0
    while sa < a_len and is_digit(a[unsafe_offset=sa]):
        sa += 1
    var sb = 0
    while sb < b_len and is_digit(b[unsafe_offset=sb]):
        sb += 1

    var a_has_epoch = (sa < a_len) and a[unsafe_offset=sa] == CH_COLON
    var b_has_epoch = (sb < b_len) and b[unsafe_offset=sb] == CH_COLON

    var a_dash = -1
    for k in range(sa, a_len):
        if a[unsafe_offset=k] == CH_DASH:
            a_dash = k
    var b_dash = -1
    for k in range(sb, b_len):
        if b[unsafe_offset=k] == CH_DASH:
            b_dash = k

    # Epochs are pure digit strings, so rpmvercmp on them reduces to a
    # zero-stripped length-then-lexicographic compare. An absent epoch is the
    # literal "0", which strips to the empty run, so it needs no case.
    var ea = 0
    if a_has_epoch:
        while ea < sa and a[unsafe_offset=ea] == CH_ZERO:
            ea += 1
    var eb = 0
    if b_has_epoch:
        while eb < sb and b[unsafe_offset=eb] == CH_ZERO:
            eb += 1
    var la = sa - ea if a_has_epoch else 0
    var lb = sb - eb if b_has_epoch else 0
    if la != lb:
        return Int32(-1) if la < lb else Int32(1)
    if la > 0:
        var rc_epoch = cmp_bytes(a, ea, la, b, eb, lb)
        if rc_epoch != 0:
            return rc_epoch

    var av = sa + 1 if a_has_epoch else 0
    var bv = sb + 1 if b_has_epoch else 0

    var a_ver_end = a_end
    var a_rel = -1
    if a_dash >= 0:
        a_ver_end = a_dash
        a_rel = a_dash + 1
    var b_ver_end = b_end
    var b_rel = -1
    if b_dash >= 0:
        b_ver_end = b_dash
        b_rel = b_dash + 1

    var rc = rpmvercmp_at(a, av, a_ver_end, b, bv, b_ver_end)
    if rc != 0:
        return rc
    # The pkgrel is only compared when BOTH versions carry one.
    if a_rel < 0 or b_rel < 0:
        return Int32(0)
    return rpmvercmp_at(a, a_rel, a_end, b, b_rel, b_end)


@export("arch_vercmp")
def arch_vercmp_export(
    a_addr: Int, a_len: Int, b_addr: Int, b_len: Int
) abi("C") -> Int32:
    return vercmp(a_addr, a_len, b_addr, b_len)


def csr_cmp(blob_addr: Int, a0: Int, a1: Int, b0: Int, b1: Int) -> Int32:
    return vercmp(blob_addr + a0, a1 - a0, blob_addr + b0, b1 - b0)


@export("arch_vercmp_batch")
def arch_vercmp_batch(
    blob_addr: Int,
    lhs_off_addr: Int,
    rhs_off_addr: Int,
    n: Int,
    dst_addr: Int
) abi("C"):
    """Compare ``n`` version pairs from one blob with two CSR offset tables.

    ``lhs_off`` and ``rhs_off`` are ``n + 1`` element Int64 arrays of byte
    offsets into ``blob``; ``dst`` receives ``n`` Int32 results. Both offset
    tables may be the same, which is what the self-comparison tests use.
    """
    if n <= 0:
        return
    var lo = i64p(lhs_off_addr)
    var ro = i64p(rhs_off_addr)
    var dst = i32p(dst_addr)
    for k in range(n):
        dst[unsafe_offset=k] = csr_cmp(
            blob_addr,
            Int(lo[unsafe_offset=k]),
            Int(lo[unsafe_offset=k + 1]),
            Int(ro[unsafe_offset=k]),
            Int(ro[unsafe_offset=k + 1]),
        )


def merge_ranges(
    blob_addr: Int, off: I64Ptr, idx: I32Ptr, scratch: I32Ptr, lo: Int, hi: Int
):
    """Merge the two sorted halves idx[lo:mid] and idx[mid:hi] in place."""
    var mid = (lo + hi) // 2
    var k = lo
    while k < hi:
        scratch[unsafe_offset=k] = idx[unsafe_offset=k]
        k += 1
    var i = lo
    var j = mid
    var m = lo
    while i < mid and j < hi:
        var left = scratch[unsafe_offset=i]
        var right = scratch[unsafe_offset=j]
        var rc = csr_cmp(
            blob_addr,
            Int(off[unsafe_offset=left]),
            Int(off[unsafe_offset=left + 1]),
            Int(off[unsafe_offset=right]),
            Int(off[unsafe_offset=right + 1]),
        )
        # Take from the left half on a tie: that is what makes the sort stable.
        if rc <= 0:
            idx[unsafe_offset=m] = left
            i += 1
        else:
            idx[unsafe_offset=m] = right
            j += 1
        m += 1
    while i < mid:
        idx[unsafe_offset=m] = scratch[unsafe_offset=i]
        i += 1
        m += 1
    while j < hi:
        idx[unsafe_offset=m] = scratch[unsafe_offset=j]
        j += 1
        m += 1


def merge_sort(
    blob_addr: Int, off: I64Ptr, idx: I32Ptr, scratch: I32Ptr, lo: Int, hi: Int
):
    if hi - lo <= 1:
        return
    var mid = (lo + hi) // 2
    merge_sort(blob_addr, off, idx, scratch, lo, mid)
    merge_sort(blob_addr, off, idx, scratch, mid, hi)
    merge_ranges(blob_addr, off, idx, scratch, lo, hi)


@export("arch_sort_indices")
def arch_sort_indices(
    blob_addr: Int, off_addr: Int, n: Int, idx_addr: Int, scratch_addr: Int
) abi("C"):
    """Sort ``n`` versions ascending, writing their original indices to ``idx``.

    A stable merge over an index permutation, so equal versions keep their
    input order and the result matches Python's ``sorted`` exactly on any input
    where the comparator is a total order. ``scratch`` is an ``n``-element
    Int32 array that the caller owns.
    """
    if n <= 0:
        return
    var off = i64p(off_addr)
    var idx = i32p(idx_addr)
    var scratch = i32p(scratch_addr)
    for k in range(n):
        idx[unsafe_offset=k] = Int32(k)
    merge_sort(blob_addr, off, idx, scratch, 0, n)


@export("arch_max_index")
def arch_max_index(blob_addr: Int, off_addr: Int, n: Int) abi("C") -> Int32:
    """Index of the newest of ``n`` versions; the lowest index wins a tie."""
    if n <= 0:
        return Int32(-1)
    var off = i64p(off_addr)
    var best = Int32(0)
    for k in range(1, n):
        var rc = csr_cmp(
            blob_addr,
            Int(off[unsafe_offset=best]),
            Int(off[unsafe_offset=best + 1]),
            Int(off[unsafe_offset=k]),
            Int(off[unsafe_offset=k + 1]),
        )
        if rc < 0:
            best = Int32(k)
    return best
