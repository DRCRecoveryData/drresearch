#!/usr/bin/env python3
"""One-pass anchor-coverage audit over the descrambled dump.

Reproduces dumpextractrelevant.pl's own Found/Missing accounting (exact 7-byte
'|Block#' magic OR the 'xxxxx\n\0' tail, 3-field LBA majority, grid step 9),
then asks the question the tool cannot: how many of the missing grid points come
back if the magic is allowed a few bit errors.  Missing because the magic rotted
is recoverable in software; missing because nothing was written there is not.

POOL=n spreads blocks over n worker processes; results merge exactly as the
serial pass would (union of found points, minimum bad-bit count per point).
"""
import os, sys
import numpy as np

BLK = int(os.environ.get('BLKSZ', 21233664))
ECCSTART = int(os.environ.get('ECCSTART', 3145728))
ECCEND = int(os.environ.get('ECCEND', 21725184))
INC = int(os.environ.get('INC', 9))
TAIL_BITS = int(os.environ.get('TOL', 4))
POOL = int(os.environ.get('POOL', 1))
MAXBLK = int(os.environ.get('MAXBLK', 0))
START = int(os.environ.get('START', 0))
MAGIC = np.frombuffer(b'|Block#', dtype=np.uint8)
TAIL = np.frombuffer(b'xxxxx\n\x00', dtype=np.uint8)
PC = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(1).astype(np.uint8)
DUMP = sys.argv[1] if len(sys.argv) > 1 else 'C:/Users/drclab/Documents/LDPC/01_01_plain.dump'
KEYFILE = os.environ.get('XORKEY', 'C:/Users/drclab/Documents/Qoder/2026-11-15/7246e382/run/01_01_plain.dump.xor')
OUT = os.environ.get('MISSOUT', 'missing_anchors.txt')
OUT2 = os.environ.get('HARDOUT', 'hard_missing.txt')

NPOINTS = (ECCEND - ECCSTART) // INC + 1


def digits_to_int(arr):
    """Vectorised strict decimal parse; -1 where any byte is not 0-9."""
    d = arr.astype(np.int16) - 48
    bad = (d < 0) | (d > 9)
    v = np.zeros(arr.shape[0], dtype=np.int64)
    for col in range(arr.shape[1]):
        v = v * 10 + np.where(bad[:, col], 0, d[:, col])
    return np.where(bad.any(1), -1, v)


def hex_to_int(arr):
    b = arr.astype(np.int16)
    ok = ((b >= 48) & (b <= 57)) | ((b >= 97) & (b <= 102)) | ((b >= 65) & (b <= 70))
    d = np.where((b >= 48) & (b <= 57), b - 48,
        np.where((b >= 97) & (b <= 102), b - 87,
        np.where((b >= 65) & (b <= 70), b - 55, 0)))
    v = np.zeros(arr.shape[0], dtype=np.int64)
    for col in range(arr.shape[1]):
        v = v * 16 + np.where(ok[:, col], d[:, col], 0)
    return np.where(ok.all(1), v, -1)


def fixdecimal(arr):
    """The tool's rescue: force low nibble | 0x30 when at most 2 bytes are not digits."""
    d = arr.astype(np.int16) - 48
    bad = (d < 0) | (d > 9)
    nb = bad.sum(1)
    low = arr.astype(np.int16) & 15                      # value after '|0x30'
    cols = np.where(bad, low, d)
    v = np.zeros(arr.shape[0], dtype=np.int64)
    for col in range(arr.shape[1]):
        v = v * 10 + np.clip(cols[:, col], 0, 9)
    valid = (low <= 9).all(1) & (nb <= 2)
    return np.where(valid, v, -1)


def score_positions(data, pos, bits=None):
    """Return (grid_idx, bits) pairs for in-range LBA fields at these magic positions.

    dumpextractrelevant.pl increments lbafound per in-range field on its own, so a
    grid point counts as found if any single field declares it -- no majority here.
    """
    n = len(pos)
    if n == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    keep = pos + 59 <= len(data)
    pos = pos[keep]
    if len(pos) == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    if bits is not None:
        bits = bits[keep]
    lbad = digits_to_int(data[pos[:, None] + np.arange(7, 19)])
    lbab_ = digits_to_int(data[pos[:, None] + np.arange(39, 59)])
    lbah = hex_to_int(data[pos[:, None] + np.arange(23, 31)])
    lbad = np.where(lbad < 0, fixdecimal(data[pos[:, None] + np.arange(7, 19)]), lbad)
    lbab_ = np.where(lbab_ < 0, fixdecimal(data[pos[:, None] + np.arange(39, 59)]), lbab_)
    lbaB = np.where(lbab_ >= 0, lbab_ // 512, -1)
    idx, bts = [], []
    for v in (lbad, lbaB, lbah):
        inr = (v >= ECCSTART) & (v <= ECCEND) & ((v - ECCSTART) % INC == 0)
        i = ((v - ECCSTART) // INC)[inr]
        idx.append(i)
        bts.append(i if bits is None else bits[inr])
    if not idx:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    return np.concatenate(idx), np.concatenate(bts)


def structural_positions(data):
    """Positions that look like a record start with the magic destroyed.

    initpattern's layout has fixed bytes at +20 '(' +21 '0' +22 'x' +31 ')' +33 'B'
    +37 ':', which is 48 bits of structure that survive independently of '|Block#'.
    The caller then requires the 12-digit sector and 20-digit byte fields to agree
    (sector*512 == byte), which is what makes a coincidence impossible.
    """
    n = len(data) - 60
    c = np.ones(n, dtype=bool)          # starts true: the loop only ever narrows it
    for off, ch in ((20, 40), (21, 48), (22, 120), (31, 41), (33, 66), (37, 58)):
        c &= data[off:off + n] == ch
    return np.flatnonzero(c)


def score_structural(data, pos):
    """Score magic-free candidates, but only when the record is self-consistent.

    Only the sector->byte relation is required.  The 8 hex digits are a physical
    block id (0x00300878 next to LBA 3151992 on this dump), not a second copy of
    the LBA, so insisting they agree would reject every real record.
    """
    pos = pos[pos + 59 < len(data)]
    if len(pos) == 0:
        return np.zeros(0, np.int64)
    d12 = digits_to_int(data[pos[:, None] + np.arange(7, 19)])
    b20 = digits_to_int(data[pos[:, None] + np.arange(39, 59)])
    good = (d12 >= 0) & (b20 == d12 * 512)
    v = d12[good]
    inr = (v >= ECCSTART) & (v <= ECCEND) & ((v - ECCSTART) % INC == 0)
    return ((v - ECCSTART) // INC)[inr]


def scan_block(args):
    b, path, keyfn = args
    key = np.fromfile(keyfn, dtype=np.uint8)
    with open(path, 'rb') as f:
        f.seek(b * BLK)
        raw = f.read(BLK + 64)
    if len(raw) < BLK + 64:
        raw = raw + bytes(64)
    data = np.frombuffer(raw, dtype=np.uint8) ^ np.concatenate([key, key[:64]])
    m = np.zeros(len(data) - 6, dtype=np.int16)
    for i in range(7):
        m += PC[data[i:i + len(m)] ^ MAGIC[i]]
    ex, exb = score_positions(data, np.flatnonzero(m == 0))
    # char2 is 7 bytes ("xxxxx\n\0"); comparing fewer lets any 'xxxxx' run pose as an
    # anchor, which is why the corrupted-anchor control is part of the KAT.
    t = np.zeros(len(data) - 6, dtype=np.int16)
    for i in range(7):
        t += PC[data[i:i + len(t)] ^ TAIL[i]]
    tailpos = np.flatnonzero(t == 0) - 512 + 7
    tx, tb = score_positions(data, tailpos[tailpos >= 0])
    cand = np.flatnonzero((m > 0) & (m <= TAIL_BITS))
    fx, fb = score_positions(data, cand, m[cand])
    sx = score_structural(data, structural_positions(data))
    return b, (np.concatenate([ex, tx]).astype(np.int32),
               np.concatenate([exb, tb]),
               fx.astype(np.int32), fb,
               sx.astype(np.int32))


def main():
    key = np.fromfile(KEYFILE, dtype=np.uint8)
    assert len(key) == BLK, 'key must be exactly one block'
    size = os.path.getsize(DUMP)
    nblocks = size // BLK
    if MAXBLK:
        nblocks = min(nblocks, MAXBLK)
    blocks = list(range(START, nblocks))
    print('key: %d bytes, block %d, scanning %d blocks' % (len(key), BLK, len(blocks)), flush=True)
    print('%s: %d bytes, %d blocks; grid %d..%d step %d = %d points, magic tolerance %d bits, pool %d'
          % (DUMP, size, nblocks, ECCSTART, ECCEND, INC, NPOINTS, TAIL_BITS, POOL), flush=True)
    if len(blocks) != size // BLK:
        print('SUBSET RUN: blocks %d..%d only -- "missing" here means missing in this subset'
              % (blocks[0], blocks[-1]), flush=True)

    found = np.zeros(NPOINTS, dtype=np.uint8)
    fuzzy_bits = np.full(NPOINTS, 255, dtype=np.int16)
    struct = np.zeros(NPOINTS, dtype=np.uint8)
    args = [(b, DUMP, KEYFILE) for b in blocks]
    if POOL > 1:
        import multiprocessing as mp
        ctx = mp.get_context('spawn')
        it = ctx.Pool(POOL).imap(scan_block, args, chunksize=4)
    else:
        it = map(scan_block, args)
    tot_exact_hits = 0
    for done, (b, (ei, eb, fi, fb, si)) in enumerate(it, 1):
        tot_exact_hits += len(ei)
        found[ei] = 1
        np.minimum.at(fuzzy_bits, fi, fb)
        struct[si] = 1
        if done % 100 == 0:
            print('  %d/%d blocks (last %d)  magic hits %d  grid points %d  magic-free %d'
                  % (done, len(blocks), b, tot_exact_hits, int(found.sum()), int(struct.sum())), flush=True)

    exact_pts = int(found.sum())
    miss = np.flatnonzero(found == 0)
    print('\ngrid points           : %d' % NPOINTS)
    print('found (exact magic)   : %d  = %.4f %%  [tool reported 2062902 = 99 %%]' % (exact_pts, 100.0 * exact_pts / NPOINTS))
    print('missing (exact magic) : %d' % (NPOINTS - exact_pts))
    for L in range(1, TAIL_BITS + 1):
        extra = int(((found == 0) & (fuzzy_bits <= L)).sum())
        tot = exact_pts + extra
        print('  with magic allowed %d bad bit(s): +%d recovered -> %.4f %%  (still missing %d)'
              % (L, extra, 100.0 * tot / NPOINTS, NPOINTS - tot))
    after_fuzzy = exact_pts + int(((found == 0) & (fuzzy_bits <= TAIL_BITS)).sum())
    only_struct = int(((found == 0) & (fuzzy_bits > TAIL_BITS) & (struct == 1)).sum())
    final = after_fuzzy + only_struct
    print('\nmagic-free structural scan adds : %d  (self-consistent record, magic destroyed)' % only_struct)
    print('total recoverable in software   : %d / %d = %.4f %%' % (final, NPOINTS, 100.0 * final / NPOINTS))
    print('hard-missing (no record at all) : %d' % (NPOINTS - final))
    hard = np.flatnonzero((found == 0) & (fuzzy_bits > TAIL_BITS) & (struct == 0))
    np.savetxt(OUT2, hard * INC + ECCSTART, fmt='%d')
    print('hard-missing LBA list           : %s (%d entries)' % (OUT2, len(hard)))
    np.savetxt(OUT, miss * INC + ECCSTART, fmt='%d')
    print('missing LBA list      : %s (%d entries)' % (OUT, len(miss)))
    if len(miss):
        print('missing span in grid  : %d .. %d  (LBA %d .. %d)'
              % (miss.min(), miss.max(), miss.min() * INC + ECCSTART, miss.max() * INC + ECCSTART))
        gaps = np.diff(miss)
        runs, cur = [], 1
        for g in gaps:
            if g == 1:
                cur += 1
            else:
                runs.append(cur); cur = 1
        runs.append(cur)
        hist = np.bincount(runs)
        print('missing run lengths   : %d runs, longest %d consecutive, singletons %d'
              % (len(runs), max(runs), int(hist[1]) if len(hist) > 1 else 0))
        print('first 20 missing LBAs :', [int(x) for x in (miss[:20] * INC + ECCSTART)])


if __name__ == '__main__':
    main()
