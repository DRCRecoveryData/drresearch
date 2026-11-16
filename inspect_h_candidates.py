#!/usr/bin/env python3
"""Are the candidate checks from recover_h.py real parity relations or degenerate ones?

A vector that is supported mostly on bytes which never vary in the sample is orthogonal to every
codeword for a boring reason - there is nothing there to vary - and it is useless as a row of H.
This scores each candidate on the thing that distinguishes the two: how many of its nonzero
positions land on columns that actually change across held-out chunks.

Usage:  python inspect_h_candidates.py [h_pattern.npy]
Env:    CHUNKOFFS DUMP XORKEY ROWS VAL
"""
import os
import sys
import numpy as np

BLK = int(os.environ.get('BLKSZ', 21233664))
CH = int(os.environ.get('CHUNK', 1144))
NROW = int(os.environ.get('ROWS', 24000))
NVAL = int(os.environ.get('VAL', 8000))
HERE = os.path.dirname(os.path.abspath(__file__))
OFFS = os.environ.get('CHUNKOFFS', os.path.join(HERE, 'pattern_chunk_offsets.txt'))
DUMP = os.environ.get('DUMP', 'C:/Users/drclab/Documents/LDPC/01_01_plain.dump')
KEYFILE = os.environ.get('XORKEY', os.path.join(HERE, '..', 'run', '01_01_plain.dump.xor'))
NCOL = CH * 8
NWORD = NCOL // 64

sys.path.insert(0, HERE)
import recover_h as R


def main():
    cand_file = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'h_pattern.npy')
    if not os.path.exists(cand_file):
        raise SystemExit('no candidate file: %s' % cand_file)
    H = np.load(cand_file)
    key = np.frombuffer(open(KEYFILE, 'rb').read(), np.uint8)
    with open(DUMP, 'rb') as fh:
        raw = R.codewords_from_offsets(fh, key, OFFS, NROW + NVAL)
    if len(raw) < NROW + 100:
        raise SystemExit('only %d chunks read - not enough to score anything' % len(raw))
    val = R.pack(raw[NROW:NROW + NVAL])
    # axis is mandatory: np.unpackbits flattens by default, which silently turns a per-column
    # statistic into one scalar and makes every weight below read as 1.
    bits = np.unpackbits(np.ascontiguousarray(val).view(np.uint8), axis=1, bitorder='little')
    assert bits.shape == (len(val), NCOL), 'bit matrix is %s, expected (%d, %d)' % (
        bits.shape, len(val), NCOL)

    vary = bits.mean(0)                                     # fraction of chunks with the bit set
    changing = (vary > 0.01) & (vary < 0.99)                # a bit that is neither stuck near 0/1
    print('held-out chunks : %d of %d bits' % (len(bits), NCOL))
    print('bit positions that change (>1 %% and <99 %% set): %d / %d (%.2f%%)'
          % (int(changing.sum()), NCOL, 100.0 * changing.sum() / NCOL))
    print('')
    print('  idx  weight  on-changing  fraction  verdict')
    real = degen = 0
    for i, row in enumerate(H):
        b = np.unpackbits(np.ascontiguousarray(row).view(np.uint8), axis=0,
                          bitorder='little').astype(bool)
        w = int(b.sum())
        on = int((b & changing).sum())
        frac = on / w if w else 0.0
        verdict = 'parity-like' if frac >= 0.5 and w >= 16 else 'degenerate (stuck bits)'
        degen += verdict.startswith('degenerate')
        real += not verdict.startswith('degenerate')
        print('  %3d  %6d  %10d   %6.3f   %s' % (i, w, on, frac, verdict))
    print('')
    print('verdict: %d of %d candidates sit on changing bits, %d are degenerate'
          % (real, len(H), degen))
    print('  rank of the pattern-only sample was 9117 of 9152, so a [9152, 8240] single-chunk')
    print('  codeword is refuted either way: 9117 > 8240 means these chunks are not all codewords')
    print('  of that code. The test that decides the next question is the codeword span.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
