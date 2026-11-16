"""Time the author's own decode path on real bytes from the extraction.

Measures LDPC_decoder.decoder.iterateMinimumSumAlgorithm, 20 iterations, one
sub-codeword at a time, exactly the way dumpdecoder.py drives it - both on the
span his slice produces (DATA+eccsize, 9104 bits) and on the correct whole-chunk
span (9152 bits). Extrapolates to page, block, the extraction and the raw dump.

Usage: python time_author_decode.py [page] [chunks]
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import LDPC_decoder as LD
from fast_minsum import synth_H

DUMP = '../submit/upload_ps2251-11_bics4(18432p).dump'
PGSZ, CH, DATA, SA, ECC = 18432, 1144, 1024, 6, 114
M, ITERS = 912, 20
EXTRACT_PAGES, RAW_PAGES = 737854, 2280960
PAGES_PER_BLOCK = 21233664 // PGSZ          # 1152 pages in this chip's block


def author_cost(cw_bits, n):
    """One author-style call: fresh decoder, fresh input, 20 iterations or until accepted."""
    H = synth_H(n, M, 16)
    d = LD.decoder(H)
    y = (2 * cw_bits - 1).astype(np.float32)
    d.setInputMSA(y, sigma=1.0)
    t0 = time.perf_counter()
    solved, _ = False, None
    for _ in range(ITERS):
        solved, _ = d.iterateMinimumSumAlgorithm()
        if solved:
            break
    return time.perf_counter() - t0, solved


def main():
    page = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    nchunks = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    with open(DUMP, 'rb') as fh:
        fh.seek(page * PGSZ)
        raw = fh.read(PGSZ)
    assert len(raw) == PGSZ, 'short read'

    layouts = {'author slice (DATA+ECC, SA skipped)': DATA + ECC,
               'correct slice (DATA+SA+ECC)': CH}
    per_chunk = PGSZ // CH
    for label, nbytes in layouts.items():
        ts, n = [], nbytes * 8
        for c in range(per_chunk):
            if c >= nchunks:
                break
            chunk = np.unpackbits(np.frombuffer(raw[c * CH:c * CH + nbytes], np.uint8),
                                  bitorder='little').astype(np.int64)
            if chunk.size != n:
                print(f'{label}: chunk {c} is short, stopping')
                break
            dt, solved = author_cost(chunk, n)
            ts.append(dt)
            print(f'  n={n:5d} bits ({nbytes} B)  chunk {c}: {dt:6.2f} s  accepted={solved}', flush=True)
        if not ts:
            continue
        per = sum(ts) / len(ts)
        page_s = per * per_chunk
        print(f'{label}: {per:.3f} s/sub-codeword over {len(ts)} chunks -> {page_s:.1f} s/page, '
              f'{page_s * PAGES_PER_BLOCK / 3600:.2f} h/block, '
              f'{page_s * EXTRACT_PAGES / 86400:.1f} d extraction ({EXTRACT_PAGES:,} pages), '
              f'{page_s * RAW_PAGES / 86400:.1f} d raw dump ({RAW_PAGES:,} pages)', flush=True)
    print(f'machine: {os.cpu_count()} logical cores; contention from the stage348 jobs applies')
    return 0


if __name__ == '__main__':
    sys.exit(main())
