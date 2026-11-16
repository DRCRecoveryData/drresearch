#!/usr/bin/env python3
"""Write a list of chunk offsets that provably carry pattern content, for recover_h.py.

recover_h.py samples every STRIDE-th chunk of a block, which mixes in free, erased and
constant-window chunks. Those are not codewords of the LDPC code, so the span of the sample fills
the whole 9,152-bit ambient space and the dual code comes back empty - which is exactly what the
`BSTRIDE` spread run measured (rank 9152, nullity 0). Restricting the sample to chunks that hold a
tier-3 anchor record (digits parsed, so the chunk really is written pattern data) makes the premise
true again.

Chunk start = blk*BLK + page*PG + (sect//2)*CH, because a 1,144 B chunk is 1,024 DATA + 6 SA +
114 ECC and the two 512 B records of a chunk sit at +0 and +512 inside its DATA field.

Spread rule: at most PERBLOCK chunks per block, every STRIDE-th block, so the sample still covers
the whole address range rather than one dense corner.

Usage:  python pattern_chunk_offsets.py [OUT.txt]
Env:    MAP BLKSZ PGSZ CHUNK PERBLOCK BSTRIDE TIER
"""
import os
import sys
import numpy as np

BLK = int(os.environ.get('BLKSZ', 21233664))
PG = int(os.environ.get('PGSZ', 18432))
CH = int(os.environ.get('CHUNK', 1144))
PERBLOCK = int(os.environ.get('PERBLOCK', 64))
BSTRIDE = int(os.environ.get('BSTRIDE', 1))
TIER = int(os.environ.get('TIER', 3))
HERE = os.path.dirname(os.path.abspath(__file__))
MAP = os.environ.get('MAP', os.path.join(HERE, 'anchor_map_full.txt'))

COLS = ('LBA', 'off', 'tier', 'base_src', 'labelbits', 'digitbits', 'totalbits',
        'rivalbits', 'blk', 'page', 'sect')


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'pattern_chunk_offsets.txt')
    a = np.atleast_2d(np.loadtxt(MAP, dtype=np.int64, comments='#'))
    if a.shape[1] != len(COLS):
        raise SystemExit('%s has %d columns, expected %d' % (MAP, a.shape[1], len(COLS)))
    i = {c: k for k, c in enumerate(COLS)}
    rows = a[a[:, i['tier']] >= TIER]
    if not len(rows):
        raise SystemExit('no rows at tier >= %d - refusing to write an empty list' % TIER)

    blk = rows[:, i['blk']]
    if BSTRIDE > 1:
        rows = rows[(blk % BSTRIDE) == 0]
        blk = rows[:, i['blk']]
    # one chunk per (block, page, chunk-in-page) - two anchors share a chunk, keep it once
    cinp = rows[:, i['sect']] // 2
    key = blk.astype(np.int64) * 100000 + rows[:, i['page']] * 20 + cinp
    order = np.lexsort((key, blk))
    rows, key, cinp, blk = rows[order], key[order], cinp[order], blk[order]

    keep = []
    seen_blk = {}
    for r, b in zip(rows, blk):
        n = seen_blk.get(b, 0)
        if n >= PERBLOCK:
            continue
        seen_blk[b] = n + 1
        keep.append(int(b) * BLK + int(r[i['page']]) * PG + int(r[i['sect']] // 2) * CH)
    offs = np.unique(np.array(keep, dtype=np.int64))
    if not len(offs):
        raise SystemExit('nothing kept - check PERBLOCK/BSTRIDE')

    with open(out, 'w') as f:
        f.write('# absolute byte offsets of 1144 B chunks holding a tier>=%d anchor record\n' % TIER)
        f.write('# source %s, %d anchor rows, %d blocks, %d chunks\n'
                 % (os.path.basename(MAP), len(rows), len(seen_blk), len(offs)))
        for o in offs:
            f.write('%d\n' % o)
    print('wrote %s: %d chunk offsets over %d blocks (cap %d per block, block stride %d)'
          % (out, len(offs), len(seen_blk), PERBLOCK, BSTRIDE))
    return 0


if __name__ == '__main__':
    sys.exit(main())
