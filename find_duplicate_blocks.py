"""Find byte-identical blocks in the raw dump.

Blocks 82, 581 and 718 turned out to be byte-identical across all 21,233,664 B, and
100 with 200 as well. That reframes two things at once: the "no readable address"
blocks may simply be copies of blocks whose records ARE readable, and the linear
sample used by recover_h.py is not made of independent codewords.

Method: cheap fingerprint (16 x 2 KB windows spread over the block) to group
candidates, then byte-exact verification of every candidate pair against the whole
block. Sampling can only miss a duplicate, never invent one, so the verified list is
sound.

Usage: python find_duplicate_blocks.py [raw.dump] [out.json]
Env: NWINDOWS (16), WIN (2048)
"""
import hashlib
import json
import os
import sys

BLK = 21233664
DUMP = sys.argv[1] if len(sys.argv) > 1 else 'C:/Users/drclab/Documents/LDPC/01_01_plain.dump'
OUT = sys.argv[2] if len(sys.argv) > 2 else 'duplicate_blocks.json'
NWINDOWS = int(os.environ.get('NWINDOWS', 16))
WIN = int(os.environ.get('WIN', 2048))
SPECIAL = [207, 218, 335, 581, 718, 1005, 1507, 1641, 1695, 1765, 1860]


def verify(fh, a, b):
    """Byte-exact whole-block comparison, in 4 MB slices so memory stays flat."""
    step = 4 * 1024 * 1024
    for off in range(0, BLK, step):
        fh.seek(a * BLK + off)
        x = fh.read(min(step, BLK - off))
        fh.seek(b * BLK + off)
        y = fh.read(min(step, BLK - off))
        if x != y:
            return False
    return True


def main():
    nblk = os.path.getsize(DUMP) // BLK
    windows = [(o, WIN) for o in window_offsets(NWINDOWS)]
    fp = {}
    with open(DUMP, 'rb') as fh:
        for b in range(nblk):
            h = hashlib.sha1()
            for off, ln in windows:
                fh.seek(b * BLK + off)
                h.update(fh.read(ln))
            fp.setdefault(h.hexdigest(), []).append(b)
    cand = [g for g in fp.values() if len(g) > 1]
    print(f'{nblk} blocks, {len(windows)} x {WIN} B sampled per block '
          f'({len(windows) * WIN * nblk / 1e6:.0f} MB read)')
    print(f'fingerprint candidate groups: {len(cand)}, {sum(len(g) for g in cand)} blocks')

    groups = []
    mismatches = 0
    with open(DUMP, 'rb') as fh:
        for g in cand:
            keep = [g[0]]
            for other in g[1:]:
                if verify(fh, keep[0], other):
                    keep.append(other)
                else:
                    mismatches += 1
            if len(keep) > 1:
                groups.append(keep)
    dup = sum(len(g) for g in groups)
    print(f'\nverified duplicate groups : {len(groups)} covering {dup} blocks '
          f'({100.0 * dup / nblk:.2f} %)  sample false-positives: {mismatches}')
    print(f'distinct physical content  : {nblk - dup + len(groups)} of {nblk} blocks')
    sizes = sorted((len(g) for g in groups), reverse=True)
    print(f'group sizes                : {sizes[:20]}{" ..." if len(sizes) > 20 else ""}')
    print(f'largest group              : {max(sizes) if sizes else 0} blocks')
    print('\ngroups (first 12):')
    for g in sorted(groups, key=lambda x: x[0])[:12]:
        print(f'  {g}')
    inv = {b: g for g in groups for b in g}
    print(f'\nthe {len(SPECIAL)} blocks with no tier-3 anchors, and their duplicate group:')
    for s in SPECIAL:
        g = inv.get(s)
        if g is None:
            print(f'  block {s:>5}: not a duplicate')
        else:
            others = [x for x in g if x != s]
            print(f'  block {s:>5}: {len(g) - 1} twin(s) -> {others[:8]}')
    json.dump({'n_blocks': nblk, 'groups': groups, 'blocks_in_groups': dup,
               'special': {str(s): inv.get(s) for s in SPECIAL}},
              open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT}  ({os.path.getsize(OUT)} B)')
    return 0


def window_offsets(n):
    """n sample windows, evenly spread, each WIN bytes, all inside one block."""
    step = BLK // n
    return [i * step + 4096 for i in range(n) if i * step + 4096 + WIN <= BLK]


if __name__ == '__main__':
    sys.exit(main())
