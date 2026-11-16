"""Does the FE's 01_01_logical*.img line up with the physical pages it should have come from?

Stage 11 of the pipeline is "rebuild the logical image from decoded bytes", and right now it rests
on an unchecked assumption: that the existing 27,263,877,120 B image is the plain page->DATA walk
the decoder output can be dropped straight into. The sizes already say the LBA story is not the
assembly rule (the credited LBA span is only 9,512,681,984 B), so the surviving hypothesis is a
concatenation of the 16 x 1,024 B DATA fields of physical pages in dump order:

    logical page index = block * 1,152 + page,   16,384 B each

27,263,877,120 / 16,384 = 1,664,055 logical pages against 1,980 x 1,152 = 2,280,960 physical ones,
so if that rule holds the FE walked 73 % of the stick and stopped, and if it does not hold then the
image skips blocks and the index is not arithmetic - which is the case that actually matters,
because a decode run feeding an unwalkable assembly is 5.2-57.8 h wasted.

Discriminator, given a measured raw BER of 1.576e-03 (mean 12.9 wrong bits per 8,192-bit DATA
field): a CORRECT page pairing differs in ~26 of 16,384 bytes, a WRONG one differs in essentially
all of them. So there is no fuzzy-matching ambiguity here, and this script can be read as a
yes/no per probe rather than a score.

Variants probed per logical page index: raw plain dump, plain^residual key, and the still-scrambled
01_01.dump - that also settles which source the FE assembled from.

Usage: python verify_logical_image.py
Env: IMG, PLAIN, SCRAM, KEY, IDX (comma separated logical page indices)
"""
import os
import sys

import numpy as np

WS = 'C:/Users/drclab/Documents/Qoder/2026-11-15/7246e382'
LD = 'C:/Users/drclab/Documents/LDPC'
IMG = os.environ.get('IMG', f'{LD}/01_01_logical_v2.img')
PLAIN = os.environ.get('PLAIN', f'{LD}/01_01_plain.dump')
SCRAM = os.environ.get('SCRAM', f'{LD}/01_01.dump')
KEY = os.environ.get('KEY', f'{WS}/submit/residual_xor_key.bin')
BLK, PG, CH, PPB = 21233664, 18432, 1144, 1152
LPG = 16 * 1024                                   # logical page = 16 DATA fields
IDX = [int(x) for x in os.environ.get('IDX', '0,1,2,64,1000,262144,800000,1200000,1664053,1664054')
       .split(',') if x.strip()]


def data_area(page_bytes):
    """The 16 x 1,024 B DATA fields of one physical page, concatenated."""
    return b''.join(page_bytes[c * CH:c * CH + 1024] for c in range(16))


def main():
    isize = os.path.getsize(IMG)
    npages_logical = isize // LPG
    print(f'image      : {os.path.basename(IMG)} {isize:,} B = {npages_logical:,} logical '
          f'{LPG} B pages (whole stick would be {1980 * PPB:,})')
    key = np.fromfile(KEY, dtype=np.uint8)
    assert key.size == BLK, f'key is {key.size} B, expected one block {BLK}'
    print(f'walk rule  : physical page = block*{PPB} + page  (image index -> dump offset '
          f'index*{PG} + (index%{PPB})*0 ... i.e. block = index//{PPB})')
    hits = 0
    with open(IMG, 'rb') as fi, open(PLAIN, 'rb') as fp, open(SCRAM, 'rb') as fs:
        for li in IDX:
            if li >= npages_logical:
                print(f'  logical page {li}: beyond image, skipped')
                continue
            fi.seek(li * LPG)
            want = np.frombuffer(fi.read(LPG), np.uint8)
            phys = li                                        # straight-walk hypothesis
            blk, off = divmod(phys, PPB)
            src_off = blk * BLK + (off + 1) * PG - PG        # block*BLK + page*PG
            row = [f'logical {li:>8} -> blk {blk:>4} page {off:>4}']
            best = None
            for name, fh, use_key in (('plain', fp, False), ('plain^key', fp, True),
                                      ('scram', fs, False)):
                if src_off + PG > os.path.getsize(PLAIN if name != 'scram' else SCRAM):
                    row.append(f'{name}: out of range')
                    continue
                fh.seek(src_off)
                page = np.frombuffer(fh.read(PG), np.uint8)
                if use_key:
                    # the residual key is exactly one block long and tiles every block, so the
                    # slice for page `off` is key[off*PG : (off+1)*PG] regardless of block number
                    page = page ^ key[off * PG:(off + 1) * PG]
                got = np.frombuffer(data_area(page), np.uint8)
                nd = int((got != want).sum())
                row.append(f'{name} {nd:>6,} diff/16384')
                if best is None or nd < best[1]:
                    best = (name, nd)
            ok = best[1] <= 120                           # ~5x the expected 26 raw-error bytes
            hits += int(ok)
            print('  ' + '  '.join(row) + f'   -> {"MATCH" if ok else "no"}')
    print(f'\nprobes matched under the straight-walk rule: {hits} of {len(IDX)}')
    print(f'expected byte-differences for a correct pairing at BER 1.576e-03: '
          f'{int(LPG * 8 * 1.576e-3 / 8):,} bytes (mean 12.9 bits per 8,192-bit field x 2 fields)')
    if hits == 0:
        print('verdict: the image is NOT a straight physical-page walk, so the FE skipped or '
              'reordered blocks - stage 11 must derive the page order instead of assuming it.')
    elif hits == len(IDX):
        print('verdict: straight walk confirmed end to end; stage 11 is an input swap onto the '
              'decoded dump, nothing else.')
    else:
        print('verdict: partial - the rule holds only over part of the image, so the walk is '
              'broken at some block boundary and needs the skip list before stage 11 runs.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
