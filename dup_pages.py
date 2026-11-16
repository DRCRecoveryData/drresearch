#!/usr/bin/env python3
"""Do logical pages appear at more than one physical place in this dump?

The donor is 42 GB of raw NAND (82,114,560 sectors) but the pattern image only
addressed 60,479,358 LBAs, so the FTL had ~1.36x more physical room than logical
work.  If a logical page was written twice (GC copy, or the pattern re-passed over
an area that had already been remapped), the two reads carry *independent* bit
errors at 1.5e-3 per bit, and a majority vote over three copies restores the
error-free codeword with probability 1-2e-7 per bit.  That is the only version of
"clean training data" the parity-check-matrix elimination can use, because the raw
read has ~12 wrong bits in every 8,192-bit DATA field.

This pass derives one base LBA per page (an intact record fixes the page's 32
contiguous LBAs) and counts how many physical pages carry the same base LBA.
"""
import os, sys
import numpy as np
from multiprocessing import Pool

BLK = int(os.environ.get('BLKSZ', 21233664))
PG = int(os.environ.get('PGSZ', 18432))
CH = int(os.environ.get('CHUNK', 1144))
NC = PG // CH
BORDER = int(os.environ.get('BORDER', 3145728))
POOL = int(os.environ.get('POOL', 10))
STEP = int(os.environ.get('STEP', 1))
MAXBLK = int(os.environ.get('MAXBLK', 0))
DUMP = sys.argv[1] if len(sys.argv) > 1 else 'C:/Users/drclab/Documents/LDPC/01_01_plain.dump'
KEYFILE = os.environ.get('XORKEY', 'C:/Users/drclab/Documents/Qoder/2026-11-15/7246e382/run/01_01_plain.dump.xor')
PC = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(1).astype(np.int16)
NREC = 82


def work(args):
    b, path, keyfn = args
    import repair_anchors as R
    key = np.fromfile(keyfn, dtype=np.uint8)
    with open(path, 'rb') as f:
        f.seek(b * BLK)
        data = np.asarray(np.frombuffer(f.read(BLK), np.uint8)) ^ key
    out = []
    for p in range(len(data) // PG):
        start = p * PG
        found = None
        for c in range(NC):
            for s in (0, 512):
                q = start + c * CH + s
                if q + NREC >= len(data):
                    continue
                obs = np.frombuffer(data[q:q + NREC], np.uint8)
                if int(PC[obs[R.SKEL] ^ R.SKELREF].sum()):
                    continue
                r = R.classify(data, q)
                if not r:
                    continue
                cand = r[0] - (2 * c + s // 512)
                if cand >= BORDER and cand % 9 == 0:
                    found = cand
                    break
            if found is not None:
                break
        if found is not None:
            out.append((found, b, start))
    return np.array(out, np.int64).reshape(-1, 3)


def main():
    nblocks = os.path.getsize(DUMP) // BLK
    blocks = list(range(0, nblocks, STEP))[:MAXBLK or nblocks]
    print('scanning %d blocks, step %d' % (len(blocks), STEP), flush=True)
    rows = []
    with Pool(POOL) as pool:
        for r in pool.imap_unordered(work, ((x, DUMP, KEYFILE) for x in blocks), chunksize=2):
            if len(r):
                rows.append(r)
    a = np.concatenate(rows)
    base = a[:, 0]
    uq, cnt = np.unique(base, return_counts=True)
    hist = {int(k): int((cnt == k).sum()) for k in sorted(set(cnt.tolist()))}
    print('physical pages anchored : %d' % len(a))
    print('distinct logical bases  : %d' % len(uq))
    print('copies-per-base         : %s' % dict(sorted(hist.items())[:10]))
    print('bases with >=2 copies   : %d' % int((cnt >= 2).sum()))
    print('bases with >=3 copies   : %d' % int((cnt >= 3).sum()))
    m = cnt.max() if len(cnt) else 0
    print('max copies of one base  : %d' % m)
    if (cnt >= 2).sum():
        ex = uq[np.argmax(cnt)]
        print('example base %d appears at: %s' % (ex, a[base == ex][:8].tolist()))
        np.savetxt('/tmp/dup_bases.txt', a[base > BORDER][:, :2], fmt='%d')
    return 0


if __name__ == '__main__':
    sys.exit(main())
