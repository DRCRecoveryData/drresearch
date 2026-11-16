#!/usr/bin/env python3
"""Can the LDPC parity-check matrix be recovered locally from this dump?

The page is 16 chunks of 1144 B = 1024 DATA + 6 SA + 114 ECC (confirmed by the
anchor slots sitting at +0 and +512 inside each chunk, and by xorsearch's own
discovered case).  A chunk is therefore one codeword of a binary [9152, k] code
with 912 parity bits.  The dual code (the set of vectors orthogonal to every
codeword) is exactly the row space of the parity-check matrix, so it can be found
by linear algebra on codewords we already have: no known plaintext and no upload.

Two numbers decide feasibility:
  rank       of the sampled codewords        -> how much of the code we excite
  nullity    = 9152 - rank                   -> how many checks we can pin down
Expected for a full-rank code: rank 8240, nullity 912.  If nullity comes out much
larger than 912 the pattern content does not span the code and the matrix is not
determined by this dump.

Each candidate check is then scored against held-out chunks: a true row of H gives
syndrome 0 on (almost) every clean codeword, a spurious one looks like a coin flip.
"""
import os, sys
import numpy as np

BLK = int(os.environ.get('BLKSZ', 21233664))
PG = int(os.environ.get('PGSZ', 18432))
CH = int(os.environ.get('CHUNK', 1144))
NROW = int(os.environ.get('ROWS', 12000))       # codewords used for the elimination
NVAL = int(os.environ.get('VAL', 4000))         # held-out codewords for scoring
NBLOCKS = int(os.environ.get('NBLOCKS', 30))
PERBLOCK = int(os.environ.get('PERBLOCK', 200))
BSTRIDE = int(os.environ.get('BSTRIDE', 10))
START = int(os.environ.get('START', 500))
STRIDE = int(os.environ.get('STRIDE', 3))       # every Nth chunk, to avoid run-to-run copies
KEYFILE = os.environ.get('XORKEY', 'C:/Users/drclab/Documents/Qoder/2026-11-15/7246e382/run/01_01_plain.dump.xor')
DUMP = sys.argv[1] if len(sys.argv) > 1 else 'C:/Users/drclab/Documents/LDPC/01_01_plain.dump'
NCOL = CH * 8
NWORD = NCOL // 64


def codewords(fh, key, blocks, perblock):
    """Sample chunks spread evenly over the whole dump.

    Taking the first NROW+NVAL chunks in block order is wrong: the dump is written in
    pattern windows, and the XOR-00/77/FF windows are constant data, so a head-loaded
    sample reports a rank of a few hundred.  Take at most `perblock` chunks from each
    visited block and keep walking the block list.
    """
    out = []
    got = 0
    need = NROW + NVAL
    for b in blocks:
        fh.seek(b * BLK)
        data = np.frombuffer(fh.read(BLK), dtype=np.uint8) ^ key
        pages = data[:(len(data) // PG) * PG].reshape(-1, PG)
        chunks = pages[:, :(PG // CH) * CH].reshape(-1, CH)          # 16 whole chunks per page
        take = chunks[::STRIDE][:perblock]
        out.append(take)
        got += len(take)
        if got >= need:
            break
    return np.concatenate(out)[:need]


def pack(mat):
    bits = np.unpackbits(mat, bitorder='little')
    return np.packbits(bits, bitorder='little').view(np.uint64).reshape(-1, NWORD)


def parity_per_row(x):
    """1 if the codeword has odd overlap with the check vector, per row (not per word)."""
    if hasattr(np, 'bitwise_count'):
        return np.bitwise_count(x).sum(1) & 1
    return np.unpackbits(np.ascontiguousarray(x).view(np.uint8), bitorder='little').sum(1) & 1


def rref(rows):
    """Row-reduce packed rows; return (reduced rows, rank, pivot columns)."""
    m = rows.copy()
    piv = []
    r = 0
    n = len(m)
    for col in range(NCOL):
        if r >= n:
            break
        w, bt = col // 64, col % 64
        bit = np.uint64(1) << np.uint64(bt)
        sel = (m[:, w] & bit) != 0
        cand = np.flatnonzero(sel)
        cand = cand[cand >= r]
        if len(cand) == 0:
            continue
        i = cand[0]
        if i != r:
            m[[r, i]] = m[[i, r]]
        # Recompute the mask after the swap: row r may not have had this bit at all,
        # so the pre-swap mask would XOR the pivot into a row that must stay clean.
        sel = (m[:, w] & bit) != 0
        sel[r] = False
        others = np.flatnonzero(sel)
        if len(others):
            m[others] ^= m[r]
        piv.append(col)
        r += 1
        if r == n:
            break
    return m[:r], r, piv


def kernel_basis(reduced, piv):
    """Basis of the null space from an RREF matrix (packed rows)."""
    pivset = set(piv)
    free = [c for c in range(NCOL) if c not in pivset]
    out = []
    for f in free:
        v = np.zeros(NWORD, dtype=np.uint64)
        v[f // 64] |= np.uint64(1) << np.uint64(f % 64)
        for row_i, col in enumerate(piv):
            w, bt = f // 64, f % 64
            if (reduced[row_i, w] >> np.uint64(bt)) & np.uint64(1):
                pw, pbt = col // 64, col % 64
                v[pw] |= np.uint64(1) << np.uint64(pbt)
        out.append(v)
    return np.array(out, dtype=np.uint64)


def selftest_codewords(rng):
    """A code with a KNOWN parity-check matrix, to prove this pipeline measures what it claims.

    n = 9152 bits, message = 8240 bits (8192 data + 48 SA), parity = 912 bits, each check
    the XOR of 12 random message bits, so the true H is [M | I] and its row space is
    exactly the dual code.  Expected: rank 8240, nullity 912, every recovered check scoring
    syndrome-zero 1.0000 on held-out codewords, and nothing scoring near 0.5.
    """
    nmsg, nchk = NCOL - 912, 912
    tot = NROW + NVAL
    nword = (nmsg + 63) // 64
    Mcols = rng.integers(0, nmsg, (nchk, 12))
    M = np.zeros((nchk, nword * 64), np.uint8)
    M[np.arange(nchk)[:, None], Mcols] = 1
    msg = np.zeros((tot, nword * 64), np.uint8)
    msg[:, :nmsg] = rng.integers(0, 2, (tot, nmsg)).astype(np.uint8)
    mp = np.packbits(msg, bitorder='little', axis=1).view(np.uint64)       # (tot, nword)
    Mp = np.packbits(M, bitorder='little', axis=1).view(np.uint64)          # (912, nword)
    par = np.zeros((tot, nchk), np.uint8)
    for i in range(nchk):
        par[:, i] = np.bitwise_count(mp & Mp[i]).sum(1) & 1
    cw = np.concatenate([np.packbits(msg, bitorder='little', axis=1)[:, :nmsg // 8],
                         np.packbits(par, bitorder='little', axis=1)], axis=1)
    assert cw.shape[1] == CH, (cw.shape, CH)
    return cw


def main():
    if os.environ.get('SELFTEST'):
        rng = np.random.default_rng(11)
        raw = selftest_codewords(rng)
        print('SELFTEST on a synthetic [9152,8240] code with 912 known checks', flush=True)
    else:
        key = np.fromfile(KEYFILE, dtype=np.uint8)
        assert len(key) == BLK
        nblk = os.path.getsize(DUMP) // BLK
        blocks = [b for b in range(START, nblk, BSTRIDE)]
        with open(DUMP, 'rb') as fh:
            raw = codewords(fh, key, blocks, PERBLOCK)
    print('chunks sampled: %d of %d B each (%d bits)' % (len(raw), CH, NCOL), flush=True)
    train, val = pack(raw[:NROW]), pack(raw[NROW:])
    print('train %s  val %s' % (train.shape, val.shape), flush=True)

    reduced, rank, piv = rref(train)
    nullity = NCOL - rank
    print('rank of %d codewords : %d  (max possible %d)' % (len(train), rank, NCOL), flush=True)
    print('nullity (checks found): %d   expected for a data+SA message: %d'
          % (nullity, 912 if not os.environ.get('SELFTEST') else 912), flush=True)

    ker = kernel_basis(reduced, piv)
    print('kernel basis rows   : %d' % len(ker), flush=True)
    if len(ker) == 0 or len(val) == 0:
        return

    def rate(v):
        return float((parity_per_row(val & v) == 0).mean())

    rates = np.array([rate(v) for v in ker])
    print('syndrome-zero rate on held-out chunks: min %.4f median %.4f max %.4f'
          % (rates.min(), float(np.median(rates)), rates.max()), flush=True)
    for th in (0.99, 0.999, 1.0):
        print('  checks with rate >= %.3f : %d' % (th, int((rates >= th).sum())))
    order = np.argsort(-rates)
    print('top 8 rates         : %s' % ' '.join('%.4f' % rates[i] for i in order[:8]), flush=True)
    np.save(os.environ.get('SAVED', 'h_candidate.npy'), ker[rates >= 0.999] if (rates >= 0.999).any() else ker)
    print('candidate H written : %s (%d x %d bits)' % (os.environ.get('SAVED', 'h_candidate.npy'),
                                                       int((rates >= 0.999).sum()) if (rates >= 0.999).any() else len(ker), NCOL))


if __name__ == '__main__':
    main()
