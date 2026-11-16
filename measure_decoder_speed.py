#!/usr/bin/env python3
"""Measure the author's decoder cost per iteration, then project it onto this chip's geometry.

dumpdecoder.py calls LDPCdecoder.decoder(H) once per sub-codeword per page and then
iterateMinimumSumAlgorithm() up to 20 times. That inner loop is python-per-edge:

    for i,(j,l) in enumerate(np.transpose(self.connections)):
        con_index = np.argwhere(np.logical_and(self.connections[0]==j, self.connections[1]!=l))

so one iteration is O(nnz^2) element work plus O(nnz) numpy calls, and the decoder constructor adds
another O(n x nnz) `np.where` scan per column. This measures the constants instead of estimating
them, on a synthetic matrix matching the geometry we recovered for the PS2251-11 / BiCS4 page:
n = 9152 bits per 1144 B chunk, m = 912 checks, row weight 16.

The matrix is synthetic - it exists only to time the loops, so nothing here says anything about
whether the real H decodes. Timing depends on n, m and nnz, not on the code's structure.

Usage:  python measure_decoder_speed.py [--iters N] [--maxsec S]
Writes: decoder_speed.txt
"""
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import LDPC_decoder as LDPCdecoder

N = int(os.environ.get('N', 9152))
M = int(os.environ.get('M', 912))
RW = int(os.environ.get('RW', 16))
ITERS = int(os.environ.get('ITERS', '3'))
MAXSEC = float(os.environ.get('MAXSEC', '900'))
OUT = os.environ.get('OUT', 'decoder_speed.txt')

PGSZ = 18432
CHUNK = 1144
CHUNKS_PER_PAGE = PGSZ // CHUNK            # 16
PAGES_DUMP = 42042654720 // PGSZ           # 2,280,960 whole dump
PAGES_EXTRACT = 13600124928 // PGSZ        # 737,854 submitted extraction


def synth_H(n, m, rw, seed=6767):
    """Column-regular-ish binary matrix with the requested row weight; only sizes matter here."""
    rng = np.random.default_rng(seed)
    H = np.zeros((m, n), np.int64)
    for j in range(m):
        cols = rng.choice(n, size=rw, replace=False)
        H[j, cols] = 1
    return H


def main():
    t_start = time.time()
    H = synth_H(N, M, RW)
    nnz = int(H.sum())
    lines = []
    w = lines.append
    print('geometry n=%d m=%d rowweight=%d nnz=%d' % (N, M, RW, nnz), flush=True)

    t0 = time.time()
    d = LDPCdecoder.decoder(H)
    t_build = time.time() - t0
    print('decoder construction: %.3f s' % t_build, flush=True)

    r = np.ones(N, np.float32)
    t0 = time.time()
    d.setInputMSA(r, 0.1)
    t_set = time.time() - t0
    print('setInputMSA: %.3f s' % t_set, flush=True)

    per_iter = []
    for i in range(ITERS):
        t0 = time.time()
        d.iterateMinimumSumAlgorithm()
        dt = time.time() - t0
        per_iter.append(dt)
        print('iteration %d: %.3f s (elapsed %.0f s of %.0f s cap)'
              % (i + 1, dt, time.time() - t_start, MAXSEC), flush=True)
        if time.time() - t_start > MAXSEC:
            print('stopping early at the time cap after %d iterations' % (i + 1), flush=True)
            break

    med = float(np.median(per_iter))
    one = t_build + t_set + med                    # cost per sub-codeword, 1 iteration
    worst = t_build + t_set + 20 * med             # cost per sub-codeword, the tool's 20-iteration cap
    units = [('sub-codeword (1144 B chunk)', 1), ('page (16 chunks, 18432 B)', CHUNKS_PER_PAGE),
             ('block (1152 pages, 21233664 B)', CHUNKS_PER_PAGE * 1152),
             ('submitted extraction (%d pages)' % PAGES_EXTRACT, CHUNKS_PER_PAGE * PAGES_EXTRACT),
             ('whole dump (%d pages)' % PAGES_DUMP, CHUNKS_PER_PAGE * PAGES_DUMP)]

    def human(sec):
        for scale, name in ((3600 * 24 * 365, 'years'), (3600 * 24, 'days'), (3600, 'h'), (60, 'min')):
            if sec >= scale:
                return '%.2f %s' % (sec / scale, name)
        return '%.3f s' % sec

    w('author decoder cost - measured, python reference implementation')
    w('  matrix (synthetic, sized to the recovered geometry): n=%d m=%d row weight %d, nnz=%d'
      % (N, M, RW, nnz))
    w('  decoder(H) construction : %.3f s   (%d columns x np.where over nnz)' % (t_build, N))
    w('  setInputMSA             : %.3f s' % t_set)
    w('  iterateMinimumSum       : %s   median %.3f s' % (', '.join('%.3f' % x for x in per_iter), med))
    w('  cost is O(nnz^2) element work plus ~%d python numpy calls per iteration'
      % (2 * nnz + N))
    w('  dumpdecoder.py imports multiprocessing and never uses it - single core, no batching')
    w('')
    w('  %-42s %14s %14s' % ('unit', '1 iteration', '20 iterations'))
    for name, chunks in units:
        w('  %-42s %14s %14s' % (name, human(one * chunks), human(worst * chunks)))
    w('')
    w('  iterations stop early on a clean syndrome, so the real figure sits between the two columns;')
    w('  the 1-iteration column is the floor this code cannot beat even with zero bit errors.')
    w('  measured for %d iterations in %.1f s of wall time under whatever else this box was doing'
      % (len(per_iter), time.time() - t_start))
    text = '\n'.join(lines) + '\n'
    with open(OUT, 'w') as f:
        f.write(text)
    print('\n' + text)
    return 0


if __name__ == '__main__':
    sys.exit(main())
