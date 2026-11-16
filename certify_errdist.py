"""Error-count distribution per sub-codeword, from the certified-plaintext model.

certify_data.py prints a mean and a 14-bin head of the histogram; the decision this
project needs is further down the tail: a rate-0.9 LDPC with 912 parity bits per
1,144 B chunk has to live with whatever the raw channel leaves there. So collect the
per-chunk bit-error counts for whole blocks and report them as quantiles plus the
share under candidate correction budgets.

Two things this settles at the same time:
  - if the counts were a model gap they would cluster at a fixed value per (pattern,
    pos) group and a few chunks would be off by whole sectors; binomial-looking counts
    with zero >100-bit chunks mean the residual is raw NAND noise;
  - whether 21/71,776 clean chunks is a ceiling on any "known-clean codeword" route.

Usage: python certify_errdist.py [nblocks]
Env: START (500), POOL (8)
"""
import os
import sys

import numpy as np

sys.argv = sys.argv[:1]                      # certify_data reads DUMP from argv[1]
import certify_data as C                      # noqa: E402  reuse the validated model verbatim

NBLK = int(sys.argv[1]) if len(sys.argv) > 1 else int(os.environ.get('NBLK', '2'))
KEYFILE = os.environ.get('XORKEY', 'C:/Users/drclab/Documents/Qoder/2026-11-15/7246e382/run/'
                                   '01_01_plain.dump.xor')
BITS = 8192                                   # DATA only, what the model predicts


def main():
    start = int(os.environ.get('START', '500'))
    pool = int(os.environ.get('POOL', '8'))
    blocks = list(range(start, start + NBLK))
    print(f'blocks {blocks}  key {os.path.basename(KEYFILE)}  DATA field {BITS} bits')
    jobs = [(b, C.DUMP, KEYFILE) for b in blocks]
    counts, nopage = [], 0
    with pool_ctx(pool) as P:
        for b, nerr, npg, sazero, satot in P.imap_unordered(C.work, jobs):
            if nerr.size:
                counts.append(nerr)
            nopage += npg
            print(f'  block {b}: {nerr.size:,} chunks, {npg} anchorless pages', flush=True)
    x = np.concatenate(counts)
    n = x.size
    ber = x.sum() / (n * BITS)
    print(f'\nchunks sampled            : {n:,}')
    print(f'mean errors / chunk       : {x.mean():.3f} bits')
    print(f'raw bit error rate        : {ber:.3e}   (sd/mean expected sqrt(1/(BITS*ber))='
          f'{1/np.sqrt(BITS*ber):.3f} for pure binomial noise)')
    print(f'median                    : {int(np.median(x))} bits')
    for q in (50, 75, 90, 95, 99, 99.9):
        print(f'  p{q:<5}: {int(np.percentile(x, q)):>4} bits')
    print(f'max                       : {int(x.max())} bits')
    print('\nshare of chunks under a candidate correction budget')
    for t in (0, 5, 10, 15, 20, 30, 50, 100):
        frac = (x <= t).mean()
        print(f'  <= {t:>3} errors : {int((x <= t).sum()):>7,}  ({100.0 * frac:7.3f} %)')
    print('\nwhat this decides:')
    print(f'  - known-clean codewords for a local dual-code route: {int((x == 0).sum())} of {n:,} '
          f'({100.0 * (x == 0).mean():.3f} %) - too few to span 8,240 dims, which is the second,')
    print('    independent reason local H recovery cannot work on this dump.')
    print(f'  - the code has to absorb a median of {int(np.median(x))} and a p99 of '
          f'{int(np.percentile(x, 99))} errors per 1,144 B chunk, on {n:,} chunks of raw noise.')
    np.save(os.environ.get('OUTNPY', 'errdist.npy'), x)
    return 0


class pool_ctx:
    """Tiny shim so the Pool is created with the right worker count and always closed."""

    def __init__(self, n):
        self.n = n
        self.pool = None

    def __enter__(self):
        from multiprocessing import Pool
        self.pool = Pool(self.n)
        return self.pool

    def __exit__(self, *exc):
        self.pool.terminate()
        self.pool.join()
        return False


if __name__ == '__main__':
    sys.exit(main())
