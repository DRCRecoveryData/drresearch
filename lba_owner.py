#!/usr/bin/env python3
"""Which NAND block holds each still-missing anchor, decided by neighbours.

lba_block_map.py interpolated LBA ranges per block from min/max anchors.  That is
invalid: one block carries several interleaved LBA streams, so the ranges overlap
and a missing LBA fell inside dozens of blocks.

The anchors themselves give the answer without any interpolation.  In one block the
found anchors step by 9 LBAs (the grid), and a step of 18 or 27 is exactly a missing
point.  So a missing grid point is bracketed by the nearest found anchor below and
above it; if both neighbours sit in the same block, that block holds the missing
record, and its page range is known from the neighbours' page positions.

Classes:
  A  both neighbours found, same block            -> block known exactly
  B  both neighbours found, different blocks      -> 2 candidate blocks
  C  one or both neighbours missing, but the gap to the found bracket is <= 4 grid
     steps and the bracket blocks agree           -> probable, short run to re-read
  D  otherwise                                    -> not attributable
"""
import os, sys
import numpy as np
from multiprocessing import Pool

BLK = int(os.environ.get('BLKSZ', 21233664))
PGSZ = int(os.environ.get('PGSZ', 18432))
ECCSTART = int(os.environ.get('ECCSTART', 3145728))
ECCEND = int(os.environ.get('ECCEND', 21725184))
INC = int(os.environ.get('INC', 9))
POOL = int(os.environ.get('POOL', 8))
MAGIC = np.frombuffer(b'|Block#', dtype=np.uint8)
TAIL = np.frombuffer(b'xxxxx\n\x00', dtype=np.uint8)
DUMP = sys.argv[1] if len(sys.argv) > 1 else 'C:/Users/drclab/Documents/LDPC/01_01_plain.dump'
KEYFILE = os.environ.get('XORKEY', 'C:/Users/drclab/Documents/Qoder/2026-11-15/7246e382/run/01_01_plain.dump.xor')
MISSING = os.environ.get('MISSING', 'real_hard_missing.txt')
OUT = os.environ.get('OUT', 'reread_blocks_verified.txt')
SUM = os.environ.get('SUMMARY', 'reread_blocks_summary.txt')
NPOINTS = (ECCEND - ECCSTART) // INC + 1


def digits_to_int(arr):
    d = arr.astype(np.int16) - 48
    bad = (d < 0) | (d > 9)
    v = np.zeros(arr.shape[0], dtype=np.int64)
    for col in range(arr.shape[1]):
        v = v * 10 + np.where(bad[:, col], 0, d[:, col])
    return np.where(bad.any(1), -1, v)


def scan_block(args):
    """Return (block, grididx array, page-in-block array) for exact-magic anchors."""
    b, path, keyfn = args
    key = np.fromfile(keyfn, dtype=np.uint8)
    with open(path, 'rb') as f:
        f.seek(b * BLK)
        raw = f.read(BLK)
    data = np.frombuffer(raw, dtype=np.uint8) ^ key
    posall, vals = [], []
    for pat, shift in ((MAGIC, 0), (TAIL, -512 + 7)):
        p = np.zeros(len(data) - 6, dtype=np.int16)
        for i in range(7):
            p += (pat[i] == data[i:i + len(p)]).astype(np.int16)
        pos = np.flatnonzero(p == 7) + shift
        pos = pos[(pos >= 0) & (pos + 19 < len(data))]
        if len(pos):
            posall.append(pos)
            vals.append(digits_to_int(data[pos[:, None] + np.arange(7, 19)]))
    if not vals:
        return b, np.zeros(0, np.int32), np.zeros(0, np.int16)
    pos = np.concatenate(posall)
    v = np.concatenate(vals)
    v = v[(v >= ECCSTART) & (v <= ECCEND) & ((v - ECCSTART) % INC == 0)]
    pos = pos[:len(v)]
    if len(v) == 0:
        return b, np.zeros(0, np.int32), np.zeros(0, np.int16)
    idx = ((v - ECCSTART) // INC).astype(np.int32)
    keep = np.unique(idx, return_index=True)[1]
    return b, idx[keep], (pos[keep] // PGSZ).astype(np.int16)


def main():
    key = np.fromfile(KEYFILE, dtype=np.uint8)
    assert len(key) == BLK, 'key must be exactly one block'
    nblocks = os.path.getsize(DUMP) // BLK
    print('%s: %d blocks, grid %d..%d step %d -> %d points'
          % (DUMP, nblocks, ECCSTART, ECCEND, INC, NPOINTS), flush=True)

    owner = [None] * NPOINTS           # list of (block, page) per found grid index
    with Pool(POOL) as pool:
        for b, idx, pg in pool.imap_unordered(scan_block, ((x, DUMP, KEYFILE) for x in range(nblocks)),
                                              chunksize=1):
            for i, p in zip(idx.tolist(), pg.tolist()):
                owner[i] = owner[i] or []
                owner[i].append((b, p))
    found = np.array([i for i in range(NPOINTS) if owner[i]], dtype=np.int64)
    print('grid points with an exact anchor : %d / %d' % (len(found), NPOINTS), flush=True)

    miss = np.loadtxt(MISSING, dtype=np.int64)
    miss = np.atleast_1d(miss)
    midx = (miss - ECCSTART) // INC
    midx = midx[(midx >= 0) & (midx < NPOINTS)]
    midx = np.unique(midx.astype(np.int64))
    print('missing LBAs to attribute        : %d' % len(midx), flush=True)

    FSET = np.zeros(NPOINTS, dtype=bool)
    FSET[found] = True
    lines = []
    counts = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
    blocks_needed = {}
    for m in midx.tolist():
        lo = found[found < m]
        hi = found[found > m]
        if len(lo) == 0 or len(hi) == 0:
            counts['D'] += 1
            lines.append('%d %d D -1 -1 no bracket inside the searched range' % (ECCSTART + m * INC, m))
            continue
        blo, bhi = lo[-1], hi[0]
        pb = sorted({b for b, _ in owner[blo]})
        ph = sorted({b for b, _ in owner[bhi]})
        gap = bhi - blo
        cls = 'D'
        det = ''
        if pb and ph and pb == ph and gap <= 2:
            cls = 'A'
            det = 'block %d, between pages %d and %d' % (
                pb[0], min(p for _, p in owner[blo]), max(p for _, p in owner[bhi]))
        elif gap <= 2 and (set(pb) & set(ph)):
            cls = 'B'
            det = 'candidates %s (lo in %s, hi in %s)' % (sorted(set(pb) | set(ph)), pb, ph)
        elif gap <= 5 and pb and ph and pb == ph:
            cls = 'C'
            det = 'block %d, gap %d grid steps' % (pb[0], gap)
        elif gap <= 5 and pb and ph:
            cls = 'C'
            det = 'candidates %s, gap %d' % (sorted(set(pb) | set(ph)), gap)
        counts[cls] += 1
        for b in (pb if cls in ('A', 'C') and pb == ph else sorted(set(pb) | set(ph))):
            blocks_needed.setdefault(b, []).append(ECCSTART + m * INC)
        lines.append('%d %d %s %d %d %s' % (ECCSTART + m * INC, m, cls, blo, bhi, det))

    with open(OUT, 'w') as f:
        f.write('# missing LBA, grid idx, class, lo idx, hi idx, block/page attribution\n')
        f.write('\n'.join(lines) + '\n')
    with open(SUM, 'w') as f:
        f.write('# blocks to re-read, ranked by how many missing LBAs they own\n')
        for b, lst in sorted(blocks_needed.items(), key=lambda kv: -len(kv[1])):
            f.write('%d %d %s\n' % (b, len(lst), ','.join(str(x) for x in sorted(lst))))
    print('A exact block            : %d' % counts['A'])
    print('B two candidate blocks   : %d' % counts['B'])
    print('C probable (wider gap)   : %d' % counts['C'])
    print('D not attributable       : %d' % counts['D'])
    print('blocks to re-read        : %d  (see %s)' % (len(blocks_needed), SUM))
    print('per-LBA table            : %s' % OUT)


if __name__ == '__main__':
    main()
