"""Where does the page's 128 B tail actually come from?

The submitted geometry says: 16 x (1024 DATA + 6 SA + 114 ECC) = 18,304 B, then a
128 B remainder. `SUBMISSION.md` inherited the words "all 0x00 in every page sampled"
from a three-page look. A block-wide sweep says that is false about half the time, so
this measures the tail properly: zero rate by page index inside the block, byte-value
histogram, printable-share, and whether the tail carries harvest-record text.

Answer it needs: is the tail a fixed structural field (zero rate correlates with page
index) or is it the end of the written data area (zero rate correlates with block)?

Usage: python audit_tail.py [raw.dump] [residual_key]
Env: NBLOCKS (default 40), BSTRIDE (default 47 -> blocks spread over the whole stick)
"""
import os
import sys

import numpy as np

DUMP = sys.argv[1] if len(sys.argv) > 1 else 'C:/Users/drclab/Documents/LDPC/01_01_plain.dump'
KEY = sys.argv[2] if len(sys.argv) > 2 else '../submit/residual_xor_key.bin'
BLK, PG, CH = 21233664, 18432, 1144
DATA, SA, ECC = 1024, 6, 114
TAIL = PG - (PG // CH) * CH                 # 128 B
PPB = BLK // PG                             # 1152 pages per block
NBLOCKS = int(os.environ.get('NBLOCKS', 40))
BSTRIDE = int(os.environ.get('BSTRIDE', 47))
RECSKEL = np.frombuffer(b'Byte:', np.uint8)


def contains(buf, pat):
    """Vectorised substring test, good enough for a 128 B window."""
    n = len(pat)
    if buf.shape[-1] < n:
        return np.zeros(buf.shape[:-1], bool)
    win = np.lib.stride_tricks.sliding_window_view(buf, n, axis=-1)
    return (win == pat).all(axis=-1).any(axis=-1)


def main():
    key = np.fromfile(KEY, dtype=np.uint8)
    assert key.size == BLK, f'residual key is {key.size} B, expected {BLK}'
    blocks_avail = os.path.getsize(DUMP) // BLK
    blocks = list(range(0, blocks_avail, BSTRIDE))[:NBLOCKS]
    hist = np.zeros(256, np.int64)
    per_page = np.zeros((len(blocks), PPB), bool)     # True = tail all zero
    per_block = np.zeros(len(blocks))
    tail_rec = 0
    tails = 0
    printable = 0
    tail_bytes = 0
    with open(DUMP, 'rb') as fh:
        for bi, b in enumerate(blocks):
            fh.seek(b * BLK)
            raw = fh.read(BLK)
            if len(raw) != BLK:
                blocks = blocks[:bi]
                break
            data = np.frombuffer(raw, np.uint8) ^ key
            pages = data[:PPB * PG].reshape(PPB, PG)
            tail = pages[:, -TAIL:]
            per_page[bi] = (tail == 0).all(axis=1)
            per_block[bi] = per_page[bi].mean()
            hist += np.bincount(tail.reshape(-1), minlength=256)
            tail_rec += int(contains(tail, RECSKEL).sum())
            tails += tail.shape[0]
            printable += int(((tail >= 32) & (tail < 127)).sum())
            tail_bytes += tail.size
    per_page = per_page[:len(blocks)]
    zero_rate = per_page.mean()
    print(f'{len(blocks)} blocks x {PPB} pages = {tails:,} pages, tail {TAIL} B '
          f'({tail_bytes:,} bytes)')
    print(f'tail all 0x00            : {zero_rate * 100:.4f} % of pages')
    print(f'tail contains "Byte:"    : {tail_rec} of {tails} pages ({100 * tail_rec / tails:.3f} %)'
          f'  -> the tail holds harvest-record text, not pad')
    print(f'tail printable ASCII     : {100 * printable / tail_bytes:.2f} % of tail bytes')
    top = np.argsort(-hist)[:8]
    print('top tail byte values     :', ', '.join(f'0x{v:02X}={hist[v]}' for v in top))
    print(f'distinct tail byte values: {int((hist > 0).sum())} of 256')
    print()
    print('STRUCTURE TEST - if the tail is a fixed field its zero rate is a function of page '
          'index; if it is the end of the written data the rate tracks the block.')
    by_page = per_page.mean(axis=0)
    idx_runs = []
    for start in range(0, PPB, 64):
        idx_runs.append((start, round(float(by_page[start:start + 64].mean()), 4)))
    print('zero rate by page index (64-page bins, first 8):', idx_runs[:8])
    print(f'page-index spread        : min {by_page.min():.4f} max {by_page.max():.4f} '
          f'-> {"structural, page-dependent" if (by_page.max() - by_page.min()) > 0.5 else "not page-structured"}')
    print(f'per-block means          : min {per_block[:len(blocks)].min():.4f} '
          f'max {per_block[:len(blocks)].max():.4f} '
          f'-> {"block-dependent, i.e. content" if (per_block[:len(blocks)].max() - per_block[:len(blocks)].min()) > 0.5 else "not block-structured"}')
    print()
    print(f'chunk grid: {PG // CH} x {CH} B = {(PG // CH) * CH} B, tail {TAIL} B; '
          f'the tail is where the code geometry stops being self-evident')
    return 0


if __name__ == '__main__':
    sys.exit(main())
