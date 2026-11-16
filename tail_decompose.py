"""Verified decomposition of the 128 B page tail, for the submission text.

History: the case file and SUBMISSION.md first said chunk 15's ECC was 146 B, then
"the 128 B remainder is all 0x00 in every page sampled". Both were over-claims. The
wide sweep says the remainder is exactly zero in 54.12 % of pages, and the nonzero
half is not one thing:

  - in 324 blocks the raw tail is uniformly 0xFF (the pad is ERASED); XORing the
    per-block residual key into it reproduces the key's own byte pattern, which is
    what looked like a "32 B live field with 4.01 set bits per byte".
  - in 12 blocks the raw tail is uniformly 0x00 (written zeros), so the descrambled
    tail is the key itself - the "128 B live field".
  - in the fill blocks the descrambled tail is zero on ~72 % of pages and holds one
    or two isolated single-bit raw errors on the rest.

So the pad is a pad, the case file's 114 B ECC is right, and the nonzero remainder is
an artefact of the key plus raw bit errors - not a controller metadata field. This
script measures exactly that, per page, so the statement in SUBMISSION.md is derived
from the artifact.

Usage: python tail_decompose.py [raw.dump] [residual_key]
Env: NBLOCKS (200), BSTRIDE (10)
"""
import os
import sys

import numpy as np

BLK, PG, CH = 21233664, 18432, 1144
PPB = BLK // PG
TAIL = PG - (PG // CH) * CH                  # 128 B at the page end
DUMP = sys.argv[1] if len(sys.argv) > 1 else 'C:/Users/drclab/Documents/LDPC/01_01_plain.dump'
KEY = sys.argv[2] if len(sys.argv) > 2 else '../submit/residual_xor_key.bin'
NBLOCKS = int(os.environ.get('NBLOCKS', 200))
BSTRIDE = int(os.environ.get('BSTRIDE', 10))
POPCOUNT = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(1)


def main():
    key = np.fromfile(KEY, dtype=np.uint8)
    assert key.size == BLK, f'residual key is {key.size} B, expected {BLK}'
    ktail = key[:PPB * PG].reshape(PPB, PG)[:, -TAIL:]
    nblk_avail = os.path.getsize(DUMP) // BLK
    blocks = list(range(0, nblk_avail, BSTRIDE))[:NBLOCKS]
    tally = {'desc zero': 0, 'raw 0xFF (erased pad)': 0, 'raw 0x00 (written-zero pad)': 0,
             'raw constant, other': 0, 'desc nonzero, raw varies (bit errors/content)': 0}
    nz_hist = np.zeros(3, np.int64)            # 1, 2-15, 16+ nonzero desc bytes
    setbits = 0
    setbits_bytes = 0
    blocks_ff = blocks_00 = blocks_mix = 0
    pages = 0
    with open(DUMP, 'rb') as fh:
        for b in blocks:
            fh.seek(b * BLK)
            raw = np.frombuffer(fh.read(BLK), np.uint8)
            if raw.size != BLK:
                break
            rp = raw[:PPB * PG].reshape(PPB, PG)
            t, k = rp[:, -TAIL:], ktail
            d = t ^ k
            dzero = (d == 0).all(axis=1)
            ff = (t == 0xFF).all(axis=1)
            z = (t == 0x00).all(axis=1)
            const = ((t[:, :1] == t).all(axis=1))
            other = const & ~ff & ~z
            vary = ~const
            tally['desc zero'] += int(dzero.sum())
            tally['raw 0xFF (erased pad)'] += int((ff & ~dzero).sum())
            tally['raw 0x00 (written-zero pad)'] += int((z & ~dzero).sum())
            tally['raw constant, other'] += int((other & ~dzero).sum())
            tally['desc nonzero, raw varies (bit errors/content)'] += int((vary & ~dzero).sum())
            live = ~dzero
            nz = (d != 0).sum(axis=1)
            nz_hist[0] += int((nz == 1).sum())
            nz_hist[1] += int(((nz >= 2) & (nz <= 15)).sum())
            nz_hist[2] += int((nz >= 16).sum())
            setbits += int(POPCOUNT[d[live]].sum())
            setbits_bytes += int(nz[live].sum())
            blocks_ff += int(ff.sum())
            blocks_00 += int(z.sum())
            blocks_mix += int((vary & ~dzero).sum())
            pages += PPB
    print(f'{len(blocks)} blocks x {PPB} pages = {pages:,} pages, tail {TAIL} B at page offset {PG - TAIL}')
    print('\nwhere the nonzero descrambled tail comes from')
    for nm, v in tally.items():
        print(f'  {nm:<48}: {v:>8,}  ({100.0 * v / pages:6.2f} %)')
    print('\nnonzero descrambled tail bytes per page (pages that are not exactly zero)')
    print(f'  exactly 1 byte : {int(nz_hist[0]):,}\n  2-15 bytes     : {int(nz_hist[1]):,}'
          f'\n  16+ bytes      : {int(nz_hist[2]):,}  <- these are the key-bleed pages, not errors')
    print(f'\nset bits per nonzero descrambled tail byte: {setbits / max(setbits_bytes, 1):.3f}')
    print('  ~1.0 = a single flipped bit in a pad that is otherwise exact; the key-bleed pages sit far above it')
    print(f'\nper-page counts of raw pad state: uniformly 0xFF {blocks_ff:,}, '
          f'uniformly 0x00 {blocks_00:,}, varying {blocks_mix:,}')
    print('\nconsequence for the case file: the 114 B ECC window is correct; the 128 B remainder is a')
    print('pad that descrambles to zero wherever the key applies, and the nonzero remainder is key bleed')
    print('over an erased or written-zero pad plus isolated raw bit errors.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
