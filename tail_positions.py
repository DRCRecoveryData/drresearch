"""Which byte positions inside the 128 B page tail are nonzero?

tail_classes.py split the pages into populations of exactly 0, 1, 2-5, 16 and 32
nonzero tail bytes, and nothing between 6-15 or above 64. Random bit errors do not
produce exact counts like 16 and 32 - a field with a fixed set of live positions does.
Sixteen is also the chunk count per page, which is the interesting coincidence: one
byte per chunk somewhere in that tail.

This gives, per tail byte position, the nonzero rate and the value histogram, so the
tail can be described as what it is instead of asserted away as padding.

Usage: python tail_positions.py [raw.dump] [residual_key]
Env: NBLOCKS (40), BSTRIDE (47)
"""
import os
import sys

import numpy as np

DUMP = sys.argv[1] if len(sys.argv) > 1 else 'C:/Users/drclab/Documents/LDPC/01_01_plain.dump'
KEY = sys.argv[2] if len(sys.argv) > 2 else '../submit/residual_xor_key.bin'
BLK, PG, CH = 21233664, 18432, 1144
TAIL = PG - (PG // CH) * CH
PPB = BLK // PG
NBLOCKS = int(os.environ.get('NBLOCKS', 40))
BSTRIDE = int(os.environ.get('BSTRIDE', 47))


def main():
    key = np.fromfile(KEY, dtype=np.uint8)
    assert key.size == BLK, f'residual key is {key.size} B, expected {BLK}'
    blocks_avail = os.path.getsize(DUMP) // BLK
    blocks = list(range(0, blocks_avail, BSTRIDE))[:NBLOCKS]
    nz = np.zeros(TAIL, np.int64)
    hist = np.zeros((TAIL, 256), np.int64)
    n = 0
    with open(DUMP, 'rb') as fh:
        for b in blocks:
            fh.seek(b * BLK)
            raw = fh.read(BLK)
            if len(raw) != BLK:
                break
            tail = (np.frombuffer(raw, np.uint8) ^ key)[:PPB * PG].reshape(PPB, PG)[:, -TAIL:]
            nz += (tail != 0).sum(axis=0)
            for p in range(TAIL):
                hist[p] += np.bincount(tail[:, p], minlength=256)
            n += PPB
    print(f'{n:,} pages, tail {TAIL} B at page offset {PG - TAIL}')
    rates = nz / float(n)
    live = np.flatnonzero(rates > 0.5)
    print(f'\npositions nonzero in >50 % of pages : {len(live)}')
    print(f'  indices : {list(int(x) for x in live[:40])}'
          f'{" ..." if len(live) > 40 else ""}')
    diffs = np.diff(live)
    if len(live) > 1:
        print(f'  spacing : min {int(diffs.min())} max {int(diffs.max())} '
              f'mode {int(np.bincount(diffs).argmax())} -> '
              f'{"regular, one entry per chunk" if len(live) in (16, 32) else "irregular"}')
    print('\nper-position detail (first 32 B of tail): rate, most common nonzero value')
    for p in range(min(32, TAIL)):
        rest = hist[p].copy()
        rest[0] = 0
        v = int(rest.argmax())
        print(f'  +{p:<3} nonzero {rates[p] * 100:6.2f} %   common nonzero 0x{v:02X} x{int(rest[v])}')
    print('\nvalue uniformity over the live positions:')
    for p in live[:20]:
        rest = hist[p].copy()
        rest[0] = 0
        tot = int(rest.sum())
        v = int(rest.argmax())
        print(f'  +{p:<3} {int(rest[v]):>7,}/{tot:<7,} = {100.0 * rest[v] / max(tot, 1):6.2f} % one value 0x{v:02X}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
