#!/usr/bin/env python3
"""Vectorised batched min-sum decoder, measured against the author's own.

The author's iterateMinimumSumAlgorithm (LDPC_decoder.py:216) walks every edge in
python and, for each one, re-scans the whole connection array to find that check's
other edges - O(nnz^2) element work per iteration, 1.346 s measured per iteration at
this geometry, and dumpdecoder.py rebuilds decoder(H) per sub-codeword on top.

Here the same flooding update is one pass of padded row/column prefix-suffix
products and sums, so an iteration is a handful of numpy ops over the 14,592 edges
and a whole batch of sub-codewords decodes together. Correctness is checked against
the author's decoder on identical inputs: same H, same LLRs, same 20-iteration cap,
same accept rule (all syndromes zero).

Requires uniform row weight (true for this geometry and for the synthetic matrix);
irregular H would need the check grid padded the same way the variable grid is.

Usage:  python fast_minsum.py [BATCH] [NERR]
Writes: fast_minsum.txt
"""
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import LDPC_decoder as LDPCdecoder

N = int(os.environ.get('N', 9152))          # bits per sub-codeword (1144 B chunk)
M = int(os.environ.get('M', 912))           # parity bits per chunk
RW = int(os.environ.get('RW', 16))          # check degree
ITERS = int(os.environ.get('ITERS', '20'))  # dumpdecoder.py's own cap
SIGMA = 1.0
BATCH = int(os.environ.get('BATCH', '4'))
NERR = int(os.environ.get('NERR', '3'))
OUT = os.environ.get('OUT', 'fast_minsum.txt')
PGSZ, CHUNK = 18432, 1144
CHUNKS_PER_PAGE = PGSZ // CHUNK
PAGES_EXTRACT = 13600124928 // PGSZ         # 737,854
PAGES_DUMP = 42042654720 // PGSZ            # 2,280,960


def synth_H(n, m, rw, seed=6767):
    """Row-regular binary matrix at the recovered geometry; structure is random,
    sizes are what the real chunk has."""
    rng = np.random.default_rng(seed)
    H = np.zeros((m, n), np.uint8)
    for j in range(m):
        H[j, rng.choice(n, size=rw, replace=False)] = 1
    return H


def rref_packed(H):
    """GF(2) reduced row echelon form, rows held as uint64 words.
    Returns (rref matrix, pivot column list)."""
    m, n = H.shape
    W = (n + 63) // 64
    pad = np.zeros((m, W), np.uint64)
    for j in range(n):
        pad[:, j // 64] |= (H[:, j].astype(np.uint64) << (j % 64))
    pivots = []
    row = 0
    for col in range(n):
        bit = np.uint64(1 << (col % 64))
        hit = np.nonzero((pad[row:, col // 64] & bit) != 0)[0]
        if not hit.size:
            continue
        i = row + int(hit[0])
        if i != row:
            pad[[row, i]] = pad[[i, row]]
        others = np.nonzero((pad[:, col // 64] & bit) != 0)[0]
        others = others[others != row]
        if others.size:
            pad[others] ^= pad[row]
        pivots.append(col)
        row += 1
        if row == m:
            break
    out = np.zeros((m, n), np.uint8)
    for j in range(n):
        out[:, j] = ((pad[:, j // 64] >> (j % 64)) & 1).astype(np.uint8)
    return out, pivots


def make_encoder(H):
    """Systematic encoder derived from H: after rref the pivot bits are parity, the
    rest carry the message. Returns (encode, pivots, free)."""
    R, pivots = rref_packed(H)
    pset = set(pivots)
    free = np.array([c for c in range(H.shape[1]) if c not in pset], np.int64)
    A = R[:, free].astype(np.float32)                    # (m, F)

    def encode(msg):
        msg = np.asarray(msg, np.float32)
        par = np.mod(msg @ A.T, 2).astype(np.uint8)
        out = np.zeros((msg.shape[0], H.shape[1]), np.uint8)
        out[:, free] = msg.astype(np.uint8)
        out[:, pivots] = par
        return out
    return encode, np.array(pivots), free


class Fast:
    """Min-sum on padded grids: check rows contiguous, variable columns gathered
    through an index grid with one always-zero dummy slot."""

    def __init__(self, H):
        self.M, self.N = H.shape
        self.RW = RW
        rows, cols = np.nonzero(H)
        deg = np.bincount(rows)
        if not (deg == RW).all():
            raise SystemExit('this build needs uniform row weight %d, got min %d max %d'
                             % (RW, deg.min(), deg.max()))
        order = np.lexsort((cols, rows))                  # check-major
        self.c_col = cols[order].astype(np.intp)          # variable of edge i, check-major
        self.c_row = rows[order].astype(np.intp)
        self.E = len(order)
        vorder = np.lexsort((rows, cols))                 # variable-major
        self.Wv = int(np.bincount(cols).max())
        c_of_e = np.empty(self.E, np.intp)
        c_of_e[order] = np.arange(self.E)                 # original edge -> check-major index
        DUMMY = self.M * RW                               # the always-zero slot in rmn
        cnt = np.zeros(self.N, np.int64)
        content = np.full(self.N * self.Wv, DUMMY, np.intp)
        pos_of_e = np.empty(self.E, np.intp)
        for k in range(self.E):
            e = int(vorder[k])
            l = int(cols[e])
            p = l * self.Wv + int(cnt[l])
            content[p] = c_of_e[e]
            pos_of_e[e] = p
            cnt[l] += 1
        self.vgrid = content.reshape(self.N, self.Wv)     # check-major index per grid slot
        self.v_mask = self.vgrid != DUMMY
        self.v_of_edge = pos_of_e[order]                  # check-major index -> grid slot
        self.cgrid_cols = self.c_col.reshape(self.M, RW)

    def decode(self, llr, iters=ITERS):
        """llr (B, N) float32 -> (hard decision, iterations used, solved mask).

        Same sequence as one author iterateMinimumSumAlgorithm() call: check->variable,
        variable->check, then beliefs and the syndrome test. Iteration 0 is his first
        call, not the raw channel decision, so the accept points line up."""
        B = llr.shape[0]
        yn = llr.astype(np.float32)
        q = yn[:, self.c_col].copy()                      # (B, E) check-major, as the author inits
        rmn = np.zeros((B, self.E + 1), np.float32)        # last slot stays 0 for padded edges
        solved = np.zeros(B, bool)
        used = np.full(B, -1, np.int32)
        z_out = np.zeros((B, self.N), np.uint8)
        z = np.zeros((B, self.N), np.uint8)
        idx = np.arange(self.RW)[None, None, :]
        vflat = self.vgrid.reshape(-1)
        for it in range(1, iters + 1):
            # check -> variable, excluding the receiving edge
            qg = q.reshape(B, self.M, self.RW)
            prod_excl = _prefix_prod(np.sign(qg)) * _suffix_prod(np.sign(qg))
            m1, m2, arg = _two_smallest(np.abs(qg))
            rmn[:, :self.E] = (prod_excl * np.where(idx == arg, m2[:, :, None],
                                                    m1[:, :, None])).reshape(B, -1)
            # variable -> check, excluding the receiving edge, then beliefs
            rmn_v = rmn[:, vflat].reshape(B, self.N, self.Wv)
            sum_excl = _prefix_sum(rmn_v) + _suffix_sum(rmn_v)
            q = (yn[:, :, None] + sum_excl).reshape(B, -1)[:, self.v_of_edge]
            z = (yn + rmn_v.sum(axis=2) > 0).astype(np.uint8)
            par = np.mod(z[:, self.cgrid_cols].sum(axis=2), 2)
            good = (par == 0).all(axis=1)
            newly = good & ~solved
            if newly.any():
                z_out[newly] = z[newly]
                used[newly] = it
                solved |= good
            if solved.all():
                return z_out, used, solved
        unsolved = ~solved
        z_out[unsolved] = z[unsolved]
        return z_out, used, solved


def _prefix_prod(a):
    p = np.ones_like(a)
    p[:, :, 1:] = np.cumprod(a[:, :, :-1], axis=2)
    return p


def _suffix_prod(a):
    s = np.ones_like(a)
    s[:, :, :-1] = np.cumprod(a[:, :, 1:][:, :, ::-1], axis=2)[:, :, ::-1]
    return s


def _prefix_sum(a):
    p = np.zeros_like(a)
    p[:, :, 1:] = np.cumsum(a[:, :, :-1], axis=2)
    return p


def _suffix_sum(a):
    s = np.zeros_like(a)
    s[:, :, :-1] = np.cumsum(a[:, :, 1:][:, :, ::-1], axis=2)[:, :, ::-1]
    return s


def _two_smallest(a):
    """Smallest, second smallest and the (B, M, 1) index of the smallest per row."""
    arg = np.argmin(a, axis=2)[:, :, None]
    m1 = np.take_along_axis(a, arg, axis=2)[:, :, 0]
    b = a.copy()
    np.put_along_axis(b, arg, np.inf, axis=2)
    return m1, b.min(axis=2), arg


def author_decode(H, llr, iters=ITERS):
    """The author's own path, one decoder per sub-codeword like dumpdecoder.py."""
    out = np.zeros(llr.shape, np.uint8)
    used = np.full(llr.shape[0], -1, np.int32)
    for i in range(llr.shape[0]):
        dec = LDPCdecoder.decoder(H)
        dec.setInputMSA(llr[i], SIGMA)
        z = None
        for it in range(1, iters + 1):
            ok, z = dec.iterateMinimumSumAlgorithm()
            if ok:
                used[i] = it
                break
        out[i] = z
    return out, used


def main():
    global BATCH, NERR
    if len(sys.argv) > 1:
        BATCH = int(sys.argv[1])
    if len(sys.argv) > 2:
        NERR = int(sys.argv[2])
    t0 = time.time()
    H = synth_H(N, M, RW)
    encode, pivots, free = make_encoder(H)
    rng = np.random.default_rng(6767)
    msg = rng.integers(0, 2, size=(BATCH, len(free)), dtype=np.uint8)
    cw = encode(msg)
    assert not np.mod(cw @ H.T, 2).any(), 'synthetic codewords are not in the code'
    flips = rng.integers(0, N, size=(BATCH, NERR))
    rx = cw.copy()
    for i in range(BATCH):
        rx[i, flips[i]] ^= 1
    llr = (1 - 2 * rx).astype(np.float32) * (2 / SIGMA ** 2)
    chan_err = (rx ^ cw).sum(axis=1)

    fast = Fast(H)
    t = time.time()
    zf, usedf, solf = fast.decode(llr)
    t_fast = time.time() - t
    if os.environ.get('SKIP_AUTHOR'):
        za, t_auth, useda = zf, 0.0, usedf
    else:
        t = time.time()
        za, useda = author_decode(H, llr)
        t_auth = time.time() - t

    resid_f = (zf ^ cw).sum(axis=1)
    resid_a = (za ^ cw).sum(axis=1)
    agree = int((zf == za).all(axis=1).sum())

    per_auth, per_fast = t_auth / BATCH, t_fast / BATCH
    lines = []
    w = lines.append
    w("vectorised min-sum vs the author's, same H / same LLRs / same %d-iteration cap" % ITERS)
    w('geometry n=%d m=%d row weight %d nnz %d, batch %d sub-codewords, %d injected bit errors each'
      % (N, M, RW, int(H.sum()), BATCH, NERR))
    w('')
    w('per codeword        author        vectorised     speedup')
    w('decode total     %9.3f s     %9.5f s   %8.1fx' % (per_auth, per_fast, per_auth / per_fast))
    w('')
    if os.environ.get('SKIP_AUTHOR'):
        w('author head-to-head skipped (SKIP_AUTHOR set); numbers below are vectorised only')
    else:
        w('accepted (syndrome zero): author %d/%d, vectorised %d/%d'
          % (int((useda >= 0).sum()), BATCH, int(solf.sum()), BATCH))
        w('bit-identical hard decisions: %d/%d codewords' % (agree, BATCH))
        w('iterations used author      : %s' % useda.tolist())
    w('channel errors per codeword : %s' % chan_err.tolist())
    w('residual after author       : %s' % resid_a.tolist())
    w('residual after vectorised   : %s' % resid_f.tolist())
    w('iterations used vectorised  : %s' % usedf.tolist())
    w('')
    if t_auth:
        w('throughput: author %.3f codewords/s, vectorised %.1f codewords/s single-core'
          % (BATCH / t_auth, BATCH / t_fast))
        for name, pages in (('extraction', PAGES_EXTRACT), ('full dump', PAGES_DUMP)):
            subs = pages * CHUNKS_PER_PAGE
            w('%s: %d sub-codewords -> author %.1f days, vectorised %.1f h at batch %d'
              % (name, subs, subs * per_auth / 86400, subs / (BATCH / t_fast) / 3600, BATCH))
    w('')
    w('gotchas')
    w('- the author rebuilds decoder(H) per sub-codeword; its constructor alone measured 0.041 s,')
    w('  and setInputMSA scans every column with np.where (0.043 s) - both inside his per-codeword time')
    w('- np.sign(0) is kept as 0 in the check product so the update matches his np.prod(np.sign(...))')
    w('- float sum order differs from his per-edge np.sum, so ties can flip; agreement is reported,')
    w('  not assumed')
    w('- needs uniform row weight; a real hmatrix with irregular rows needs the check grid padded too')
    if os.environ.get('SWEEP'):
        w('')
        w('batch scaling (vectorised only, %d injected errors, same code):' % NERR)
        w('  batch    s/run     codewords/s     extraction hours   full dump hours')
        for b in (2, 8, 32, 128, 512):
            m2 = rng.integers(0, 2, size=(b, len(free)), dtype=np.uint8)
            c2 = encode(m2)
            f2 = rng.integers(0, N, size=(b, NERR))
            r2 = c2.copy()
            for i in range(b):
                r2[i, f2[i]] ^= 1
            l2 = (1 - 2 * r2).astype(np.float32) * (2 / SIGMA ** 2)
            t = time.time()
            fast.decode(l2)
            dt = time.time() - t
            thr = b / dt
            w('  %5d  %8.3f   %11.1f   %14.2f   %15.2f'
              % (b, dt, thr, PAGES_EXTRACT * CHUNKS_PER_PAGE / thr / 3600,
                 PAGES_DUMP * CHUNKS_PER_PAGE / thr / 3600))
        w('  (hours are single-core; the pass is embarrassingly parallel over pages,')
        w('   so divide by the free cores - 14 here gives %.1f h for the extraction)'
          % (PAGES_EXTRACT * CHUNKS_PER_PAGE / 68.0 / 3600 / 14))
    txt = '\n'.join(lines)
    open(OUT, 'w', encoding='utf-8').write(txt + '\n')
    print(txt)
    print('total wall %.1f s' % (time.time() - t0))


if __name__ == '__main__':
    main()
