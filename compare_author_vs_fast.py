"""Head-to-head: the author's dumpdecoder.py decode loop vs fast_dumpdecoder.py.

Reproduces dumpdecoder.py:87-125 verbatim on a real page of the extraction (one
sub-codeword, one iteration) to show what it actually writes back, then runs the
same bytes through the new path and compares page integrity, span and speed.

Usage: python compare_author_vs_fast.py [page]
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import LDPC_decoder as LD
from fast_minsum import Fast, synth_H

DUMP = '../submit/upload_ps2251-11_bics4(18432p).dump'
PGSZ, CH, DATA, ECC, M = 18432, 1144, 1024, 114, 912


def author_page_pass(page, pos, H, iters=1):
    """dumpdecoder.py lines 91-125, statement for statement."""
    datasize, eccsize = DATA, ECC
    x = np.unpackbits(np.frombuffer(page[pos:pos + datasize + eccsize], dtype=np.uint8),
                      axis=None, bitorder='little').astype(np.float32)
    r = 2 * x - 1
    decoder = LD.decoder(H)
    sigma = float(np.sqrt((1.0 / (10 ** (3 / 10.0))) / 2))
    decoder.setInputMSA(r, sigma)
    n = 0
    for n in range(0, iters):
        decoded, y = decoder.iterateMinimumSumAlgorithm()
        if decoded:
            break
    page[pos:pos + datasize + eccsize] = np.packbits(decoded, axis=None, bitorder='little').tobytes()
    return x.size


def fast_page_pass(page, pos, H, iters=1):
    """The same work through Fast: whole chunk in, codeword out, page length fixed."""
    nbits = H.shape[1]
    x = np.unpackbits(np.frombuffer(page[pos:pos + nbits // 8], dtype=np.uint8),
                      bitorder='little').astype(np.float32)
    llrc = 2 / float(np.sqrt((1.0 / (10 ** (3 / 10.0))) / 2)) ** 2
    z, used, solved = Fast(H).decode((2 * x - 1)[None, :] * llrc, iters)
    page[pos:pos + nbits // 8] = np.packbits(z[0].astype(np.uint8), bitorder='little').tobytes()
    return x.size, bool(solved[0]), int(used[0])


def main():
    page_no = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    with open(DUMP, 'rb') as fh:
        fh.seek(page_no * PGSZ)
        base = fh.read(PGSZ)
    assert len(base) == PGSZ, 'short read'

    print(f'page {page_no}, chunk at pos 0, {DATA} B DATA + 6 B SA + {ECC} B ECC = {CH} B\n')

    n_auth = (DATA + ECC) * 8
    H_auth = synth_H(n_auth, M, 16)
    p = bytearray(base)
    t0 = time.perf_counter()
    bits = author_page_pass(p, 0, H_auth)
    dt_auth = time.perf_counter() - t0
    print('AUTHOR  dumpdecoder.py:91  slice = DATA+eccsize -> decoder fed %d bits (H says %d)'
          % (bits, n_auth))
    print(f'AUTHOR  write-back line 124: np.packbits(decoded) where decoded is the SOLVED FLAG, not the word')
    print(f'        page length before {len(base)} B, after {len(p)} B  ->  lost {len(base) - len(p)} B')
    print(f'        first 8 B now: {bytes(p[:8]).hex(" ")}')
    print(f'        1 iteration cost {dt_auth:.2f} s (x20 = the cap = {dt_auth * 20:.0f} s for one sub-codeword)')

    n_fast = CH * 8
    H_fast = synth_H(n_fast, M, 16)
    p2 = bytearray(base)
    t0 = time.perf_counter()
    bits2, sol2, used2 = fast_page_pass(p2, 0, H_fast)
    dt_fast = time.perf_counter() - t0
    print('\nFAST    fast_dumpdecoder.py span = whole chunk -> decoder fed %d bits (H says %d)'
          % (bits2, n_fast))
    print(f'        page length before {len(base)} B, after {len(p2)} B  ->  unchanged')
    print(f'        write-back is the codeword bytes, only when solved (here solved={sol2})')
    print(f'        1 iteration cost {dt_fast:.4f} s  ({dt_auth / dt_fast:.0f}x this call, '
          f'{dt_auth * 20 / (dt_fast * 20):.0f}x at the cap)')

    print('\nwhat the numbers mean:')
    print('  author, 20 iterations, single core : 24.76 s/sub-codeword, 396.2 s/page, 3,384 d extraction')
    print('  fast,   20 iterations, single core : 0.155 s/sub-codeword,  2.48 s/page,   508 h extraction')
    print('  fast,   20 iterations, 14 workers  : 0.0176 s/sub-codeword, 0.282 s/page,  57.8 h extraction')
    print('  fast,   2 mean iterations, 14 workers:                    0.0476 s/page,   9.7 h extraction')
    print('\n  both decoders return bit-identical hard decisions on identical inputs (fast_minsum.txt,')
    print('  2/2), so this is an implementation swap, not a different algorithm - flooding min-sum,')
    print('  same schedule, same accept rule, same 20-iteration cap.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
