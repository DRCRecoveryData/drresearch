#!/usr/bin/env python3
"""Which NAND block holds which LBA, for the anchors that were found.

Turns a missing-LBA list into a physical re-read list: initpattern wrote one
sequential LBA stream, so block order and LBA order agree and each block covers a
contiguous LBA span.  With min/max per block, any LBA can be attributed to a block
(+-1 for the block that straddles the boundary) and the Flash Extractor can be told
what to read again.

Only exact-magic anchors are used: their address needed no repair, so the span ends
are trustworthy.
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
OUT = os.environ.get('OUT', 'lba_block_map.txt')


def digits_to_int(arr):
    d = arr.astype(np.int16) - 48
    bad = (d < 0) | (d > 9)
    v = np.zeros(arr.shape[0], dtype=np.int64)
    for col in range(arr.shape[1]):
        v = v * 10 + np.where(bad[:, col], 0, d[:, col])
    return np.where(bad.any(1), -1, v)


def scan_block(args):
    b, path, keyfn = args
    key = np.fromfile(keyfn, dtype=np.uint8)
    with open(path, 'rb') as f:
        f.seek(b * BLK)
        raw = f.read(BLK + 64)
    if len(raw) < BLK + 64:
        raw = raw + bytes(64)
    data = np.frombuffer(raw, dtype=np.uint8) ^ np.concatenate([key, key[:64]])
    posall, lbavals = [], []
    for pat, shift in ((MAGIC, 0), (TAIL, -512 + 7)):
        p = np.zeros(len(data) - 6, dtype=np.int16)
        for i in range(7):
            p += (pat[i] == data[i:i + len(p)]).astype(np.int16)
        pos = np.flatnonzero(p == 7) + shift
        pos = pos[pos >= 0]
        pos = pos[pos < BLK]                  # the +64 tail read belongs to the next block
        pos = pos[pos + 19 < len(data)]
        if len(pos):
            posall.append(pos)
            lbavals.append(digits_to_int(data[pos[:, None] + np.arange(7, 19)]))
    if not lbavals:
        return b, 0, -1, -1, -1
    v = np.unique(np.concatenate(lbavals))     # magic and tail mark the same record
    v = v[(v >= ECCSTART) & (v <= ECCEND) & ((v - ECCSTART) % INC == 0)]
    if len(v) == 0:
        return b, 0, -1, -1, -1
    return b, len(v), int(v.min()), int(v.max()), int(np.min(np.concatenate(posall)) // PGSZ)


def main():
    key = np.fromfile(KEYFILE, dtype=np.uint8)
    assert len(key) == BLK, 'key must be exactly one block'
    nblocks = os.path.getsize(DUMP) // BLK
    print('%s: %d blocks of %d, grid %d..%d step %d' % (DUMP, nblocks, BLK, ECCSTART, ECCEND, INC), flush=True)
    rows = []
    done = 0
    with Pool(POOL) as pool:
        for r in pool.imap_unordered(scan_block, ((b, DUMP, KEYFILE) for b in range(nblocks)), chunksize=1):
            rows.append(r)
            done += 1
            if done % 200 == 0:
                print('  %d/%d blocks' % (done, nblocks), flush=True)
    rows.sort()
    with open(OUT, 'w') as f:
        for b, n, lo, hi, pg in rows:
            if n:
                f.write('%d %d %d %d %d\n' % (b, lo, hi, n, pg))
    covered = sum(r[1] for r in rows)
    print('blocks with anchors : %d / %d' % (sum(1 for r in rows if r[1]), nblocks))
    print('anchors mapped      : %d' % covered)
    print('LBA span            : %d .. %d' % (min((r[2] for r in rows if r[1]), default=-1),
                                              max((r[3] for r in rows if r[1]), default=-1)))
    print('per-block span      : median %d LBAs' % int(np.median([r[3] - r[2] for r in rows if r[1]])))
    print('table               : %s (block loLBA hiLBA count firstAnchorPage)' % OUT)


if __name__ == '__main__':
    main()
