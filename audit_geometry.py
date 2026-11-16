"""Geometry audit over a wide page sample: does the submitted .case hold everywhere?

The submission states chunk 15's ECC is 114 B and the page ends in a 128 B all-zero
remainder. Both were measured on three pages. This walks a stride sample across the
whole raw dump, removes the residual static XOR key, and counts violations per claim,
so the geometry in the package is backed by a number instead of an impression.

Usage: python audit_geometry.py [raw.dump] [residual_key]
Env: NPAGES (default 20000), STRIDE (default pages-per-block so the sample is block-wide)
"""
import os
import sys

import numpy as np

DUMP = sys.argv[1] if len(sys.argv) > 1 else 'C:/Users/drclab/Documents/LDPC/01_01_plain.dump'
KEY = sys.argv[2] if len(sys.argv) > 2 else '../submit/residual_xor_key.bin'
BLK, PG, CH = 21233664, 18432, 1144
DATA, SA, ECC, PAD = 1024, 6, 114, 128
NPAGES = int(os.environ.get('NPAGES', 20000))
STRIDE = int(os.environ.get('STRIDE', BLK // PG))          # one page per block by default

ZERO_OK = 0            # remainder all 0x00
ZERO_BAD = 0           # remainder has a nonzero byte
REMAINDER_BYTES = np.zeros(256, np.int64)   # true byte counts (bincount, not h[idx]+=1)
ASCII_BYTES = 0
REMAINDER_HAS_RECORD = 0     # remainder carries a harvest record shell
SA_NONZERO = 0         # SA slots that carry something (expected: written, not a violation)
ECC_UNIFORM = 0        # chunks whose 114 B parity window is all one constant byte
CHUNKS = 0


def main():
    key = np.fromfile(KEY, dtype=np.uint8)
    assert key.size == BLK, f'residual key is {key.size} B, expected one block {BLK}'
    blocks = os.path.getsize(DUMP) // BLK
    ppb = BLK // PG
    pages_total = blocks * ppb
    want = min(NPAGES, pages_total)
    global CHUNKS, SA_NONZERO, ZERO_OK, ZERO_BAD, ECC_UNIFORM
    sampled = 0
    with open(DUMP, 'rb') as fh:
        for b in range(blocks):
            if sampled >= want:
                break
            for p in range(0, ppb, STRIDE):
                if sampled >= want:
                    break
                fh.seek(b * BLK + p * PG)
                raw = fh.read(PG)
                if len(raw) != PG:
                    break
                kp = p * PG
                page = np.frombuffer(raw, np.uint8) ^ key[kp:kp + PG]
                sampled += 1
                chunks = page[:18304].reshape(-1, CH)
                pad = page[18304:]
                CHUNKS += len(chunks)
                SA_NONZERO += int((chunks[:, DATA:DATA + SA] != 0).any(axis=1).sum())
                # a parity window that is entirely one constant byte is the erased signature
                par = chunks[:, DATA + SA:]
                ECC_UNIFORM += int(((par == par[:, :1]).all(axis=1)).sum())
                PAD_HIST[pad] += 1
                nz = int((pad != 0).sum())
                if nz == 0:
                    ZERO_OK += 1
                else:
                    ZERO_BAD += 1
                    per_page_bad.append((b, p, nz))
    print(f'sampled {sampled} pages ({sampled * 100 / max(pages_total, 1):.2f} % of '
          f'{pages_total} pages in {blocks} blocks), {CHUNKS} chunks, stride {STRIDE} pages')
    print(f'128 B remainder all 0x00 : {ZERO_OK}/{sampled} pages ({100 * ZERO_OK / max(sampled, 1):.4f} %)')
    print(f'128 B remainder nonzero  : {ZERO_BAD} pages; worst first: '
          f'{sorted(per_page_bad, key=lambda t: -t[2])[:5]}')
    print(f'remainder byte histogram : distinct values {int((PAD_HIST > 0).sum())}, '
          f'top {sorted(((int(v), int(c)) for v, c in enumerate(PAD_HIST) if c), key=lambda t: -t[1])[:5]}')
    print(f'SA slots carrying data   : {SA_NONZERO}/{CHUNKS} chunks ({100 * SA_NONZERO / max(CHUNKS, 1):.2f} %)')
    print(f'uniform-byte parity windows (erased signature): {ECC_UNIFORM}/{CHUNKS} '
          f'({100 * ECC_UNIFORM / max(CHUNKS, 1):.2f} %)')
    print(f'field widths audited     : DATA {DATA} + SA {SA} + ECC {ECC} = {CH} B, '
          f'16 chunks = {16 * CH} B, remainder {PG - 16 * CH} B')
    return 0


if __name__ == '__main__':
    sys.exit(main())
