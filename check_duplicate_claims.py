"""Re-derive, from the artifacts, the two claims SUBMISSION.md makes about duplicate blocks:

  1. no duplicate-group block appears in the harvest offsets (pattern_chunk_offsets.txt);
  2. the blocks that carry no tier-3 anchor and how many of them sit in a duplicate group.

Prints the measured sets so the wording in the submission can be checked against them rather
than against a note from an earlier session.

Usage: python check_duplicate_claims.py
"""
import json
import os
import sys

BLK = 21233664
PPB = 1152
PG = 18432
MAP = 'anchor_map_full.txt'
OFFS = 'pattern_chunk_offsets.txt'
DUPS = 'duplicate_blocks.json'


def tier3_blocks():
    seen = set()
    mapped = set()
    with open(MAP) as fh:
        for line in fh:
            if line.startswith('#'):
                continue
            f = line.split()
            if len(f) < 10:
                continue
            mapped.add(int(f[8]))
            if int(f[2]) == 3:
                seen.add(int(f[8]))
    return mapped, seen


def harvest_blocks():
    out = set()
    with open(OFFS) as fh:
        for line in fh:
            if line.startswith('#') or not line.strip():
                continue
            for tok in line.replace(',', ' ').split():
                try:
                    v = int(tok)
                except ValueError:
                    continue
                if v >= BLK:
                    out.add(v // BLK)
    return out


def main():
    groups = json.load(open(DUPS))['groups']
    dup = sorted({b for g in groups for b in g})
    mapped, tier3 = tier3_blocks()
    harv = harvest_blocks()
    in_harvest = [b for b in dup if b in harv]
    no_t3 = sorted(mapped - tier3)
    print(f'duplicate blocks            : {len(dup)} in {len(groups)} groups '
          f'{[len(g) for g in groups]}')
    print(f'blocks in the anchor map    : {len(mapped)}')
    print(f'blocks with >=1 tier-3 row  : {len(tier3)}')
    print(f'mapped but no tier-3 anchor : {len(no_t3)} -> {no_t3[:20]}')
    print(f'  of those, in a dup group  : {len(set(no_t3) & set(dup))} '
          f'{sorted(set(no_t3) & set(dup))[:20]}')
    print(f'harvest-offset blocks seen  : {len(harv)}')
    print(f'duplicate blocks in harvest : {len(in_harvest)} {in_harvest[:20]}')
    never = sorted(set(dup) - harv)
    print(f'duplicate blocks absent from harvest offsets: {len(never)} of {len(dup)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
