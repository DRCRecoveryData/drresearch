"""Is the 128 B page tail one field or two?

audit_tail.py found the tail is zero in 54.1 % of pages, and the nonzero byte values
are dominated by powers of two - 0x80, 0x10, 0x20, 0x08, 0x04, 0x40, 0x02 at ~3,000
counts each. A constant field plus raw bit errors gives exactly that signature. A data
field would give a flat histogram.

This separates the two stories by counting nonzero bytes per page:
  bit-error pad    -> peak at 0, a few pages with 1-3, almost none at 128
  second data      -> a real population with the whole 128 B nonzero

Usage: python tail_classes.py [raw.dump] [residual_key]
Env: NBLOCKS (40), BSTRIDE (47)
"""
import os
import sys

import numpy as np

DUMP = sys.argv[1] if len(sys.argv) > 1 else 'C:/Users/drclab/Documents/LDPC/01_01_plain.dump'
KEY = sys.argv[2] if len(sys.argv) > 2 else '../submit/residual_xor_key.bin'
BLK, PG, CH = 21233664, 18432, 1144
TAIL = PG - (PG // CH) * CH                  # 128 B
PPB = BLK // PG
NBLOCKS = int(os.environ.get('NBLOCKS', 40))
BSTRIDE = int(os.environ.get('BSTRIDE', 47))
POPCOUNT = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(1).astype(np.int32)


def main():
    key = np.fromfile(KEY, dtype=np.uint8)
    assert key.size == BLK, f'residual key is {key.size} B, expected {BLK}'
    blocks_avail = os.path.getsize(DUMP) // BLK
    blocks = list(range(0, blocks_avail, BSTRIDE))[:NBLOCKS]
    counts = []
    bits = []
    with open(DUMP, 'rb') as fh:
        for b in blocks:
            fh.seek(b * BLK)
            raw = fh.read(BLK)
            if len(raw) != BLK:
                break
            tail = (np.frombuffer(raw, np.uint8) ^ key)[:PPB * PG].reshape(PPB, PG)[:, -TAIL:]
            counts.append((tail != 0).sum(axis=1))
            bits.append(POPCOUNT[tail].sum(axis=1))     # set bits, to test multi-bit flips
    counts = np.concatenate(counts)
    bits = np.concatenate(bits)
    n = len(counts)
    print(f'{n:,} pages ({len(blocks)} blocks x {PPB}), tail {TAIL} B')
    print('\nnonzero tail bytes per page')
    for lo, hi in ((0, 0), (1, 1), (2, 5), (6, 15), (16, 31), (32, 63), (64, 127), (128, 128)):
        c = int(((counts >= lo) & (counts <= hi)).sum())
        print(f'  {lo:>3}-{hi:<3} : {c:>7,}  ({100.0 * c / n:6.3f} %)')
    nz = counts[counts > 0]
    print(f'\nset bits per nonzero byte, pages with nonzero tail : {float(bits[counts > 0].sum()) / int(nz.sum()):.3f}')
    print('  1.00 = every nonzero byte is one flipped bit; >1.3 = the tail carries content')
    print(f'\npages with the whole 128 B nonzero : {int((counts == TAIL).sum()):,} '
          f'({100.0 * int((counts == TAIL).sum()) / n:.3f} %)')
    verdict = ('one class: a constant pad field with raw bit errors'
               if (counts == TAIL).sum() < 0.02 * n else
               'two classes: a pad field AND a page tail that carries data')
    print(f'\nverdict: {verdict}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
