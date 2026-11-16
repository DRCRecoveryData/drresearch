"""Per-block character of the 128 B page tail.

audit_tail.py averaged over 40 blocks and reported the tail all-zero in 54.1 % of
pages; tail_classes.py then produced a histogram (exact populations of 1, 16, 32
nonzero bytes and nothing between 6 and 15) that a direct check on block 82
contradicts - there the tail is nonzero in ~128 of 128 bytes on essentially every
page. Two scripts, one dump, so the aggregate is hiding a per-block switch: the tail
is one thing in some blocks and another thing in the rest.

This prints it block by block, with the cross-check that the per-page and per-position
nonzero totals agree.

Usage: python tail_by_block.py [raw.dump] [residual_key]
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
POPCOUNT = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(1)


def main():
    key = np.fromfile(KEY, dtype=np.uint8)
    assert key.size == BLK, f'residual key is {key.size} B, expected {BLK}'
    blocks_avail = os.path.getsize(DUMP) // BLK
    blocks = list(range(0, blocks_avail, BSTRIDE))[:NBLOCKS]
    print(f'blocks sampled: {len(blocks)} (stride {BSTRIDE}), {TAIL} B tail, {PPB} pages/block\n')
    print('block  pages  tailAllZero  nzBytes/page  medianNz  setBitsPerNzByte  distinctValues')
    rows = []
    with open(DUMP, 'rb') as fh:
        for b in blocks:
            fh.seek(b * BLK)
            raw = fh.read(BLK)
            if len(raw) != BLK:
                break
            tail = (np.frombuffer(raw, np.uint8) ^ key)[:PPB * PG].reshape(PPB, PG)[:, -TAIL:]
            nz = tail != 0
            per_page = nz.sum(axis=1)
            per_pos = nz.sum(axis=0)
            assert int(per_page.sum()) == int(per_pos.sum()), 'page and position totals disagree'
            allzero = int((per_page == 0).sum())
            bits_per_nz = float(POPCOUNT[tail[nz]].sum()) / max(int(per_pos.sum()), 1)
            vals = int(np.unique(tail[tail != 0]).size) if nz.any() else 0
            rows.append((b, allzero, float(per_page.mean()), int(np.median(per_page)),
                         bits_per_nz, vals))
            print(f'{b:>5}  {PPB:>5}  {allzero:>6} ({100.0 * allzero / PPB:5.1f}%)  '
                  f'{per_page.mean():>12.2f}  {int(np.median(per_page)):>8}  '
                  f'{bits_per_nz:>15.3f}  {vals:>6}')
    arr = np.array([[r[2], r[4]] for r in rows])
    groups = {}
    for r in rows:
        groups['zero-tail block' if r[1] > PPB * 0.5 else 'live-tail block'] = \
            groups.get('zero-tail block' if r[1] > PPB * 0.5 else 'live-tail block', 0) + 1
    print(f'\nblock classes: {groups}')
    print(f'nz/page range      : {arr[:, 0].min():.2f} .. {arr[:, 0].max():.2f}')
    print(f'set-bits/nz-byte   : {arr[:, 1].min():.3f} .. {arr[:, 1].max():.3f}  '
          f'(1.0 = single-bit flips in a constant field; >2 = real content)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
