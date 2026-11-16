#!/usr/bin/env python3
"""Last software pass for the anchors anchor_coverage.py calls hard-missing.

anchor_coverage.py rejects a grid point when the '|Block#' magic is rotted beyond
tolerance AND the record fails its self-consistency test.  That test needs BOTH
decimal fields readable, so a record whose 20-digit byte field is damaged is
thrown away even when its 12-digit sector number is perfectly intact.

This pass scores the record FORMAT instead (the 11 fixed bytes after the magic,
magic excluded), and accepts a point when the format score is high and the sector
number equals one of the still-missing LBAs exactly.  With 11 format bytes at
score >= 9 and an exact hit on a 1,485-element target list, a coincidence is
around 2**-72 per position: the recovery is evidence, not a guess.

Grade A = format score >= MINFMT, sector parses, byte field parses, and
          sector*512 == byte (record proves itself twice over).
Grade B = format score >= MINFMT and the sector number equals a missing LBA, but
          the byte field is too damaged to cross-check.
Grade C = same as B plus the 8 hex digits parse; kept as a separate count because
          those digits are a physical id, not a second copy of the LBA, so they
          neither help nor hurt the identity test.
"""
import os, sys
import numpy as np
from multiprocessing import Pool

BLK = int(os.environ.get('BLKSZ', 21233664))
ECCSTART = int(os.environ.get('ECCSTART', 3145728))
ECCEND = int(os.environ.get('ECCEND', 21725184))
INC = int(os.environ.get('INC', 9))
POOL = int(os.environ.get('POOL', 8))
MAXBLK = int(os.environ.get('MAXBLK', 0))
START = int(os.environ.get('START', 0))
MINFMT = int(os.environ.get('MINFMT', 9))
MAGIC = np.frombuffer(b'|Block#', dtype=np.uint8)
TAIL = np.frombuffer(b'xxxxx\n\x00', dtype=np.uint8)
PC = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(1).astype(np.uint8)
# '(' '0' 'x' at +20, ')' ' ' 'B' 'y' 't' 'e' ':' ' ' at +31..+38.  No magic bytes:
# the whole point is to recognise a record whose magic is gone.
FMT = ((20, 40), (21, 48), (22, 120), (31, 41), (32, 32), (33, 66),
       (34, 121), (35, 116), (36, 101), (37, 58), (38, 32))
NFMT = len(FMT)

DUMP = sys.argv[1] if len(sys.argv) > 1 else 'C:/Users/drclab/Documents/LDPC/01_01_plain.dump'
KEYFILE = os.environ.get('XORKEY', 'C:/Users/drclab/Documents/Qoder/2026-11-15/7246e382/run/01_01_plain.dump.xor')
TARGETS = sys.argv[2] if len(sys.argv) > 2 else 'real_hard_missing.txt'
OUT = os.environ.get('OUTLIST', 'hard_recovered.txt')

NPOINTS = (ECCEND - ECCSTART) // INC + 1


def digits_to_int(arr):
    d = arr.astype(np.int16) - 48
    bad = (d < 0) | (d > 9)
    v = np.zeros(arr.shape[0], dtype=np.int64)
    for col in range(arr.shape[1]):
        v = v * 10 + np.where(bad[:, col], 0, d[:, col])
    return np.where(bad.any(1), -1, v)


def fixdecimal(arr):
    """The tool's rescue: force low nibble | 0x30 when at most 2 bytes are not digits."""
    d = arr.astype(np.int16) - 48
    bad = (d < 0) | (d > 9)
    nb = bad.sum(1)
    low = arr.astype(np.int16) & 15
    cols = np.where(bad, low, d)
    v = np.zeros(arr.shape[0], dtype=np.int64)
    for col in range(arr.shape[1]):
        v = v * 10 + np.clip(cols[:, col], 0, 9)
    valid = (low <= 9).all(1) & (nb <= 2)
    return np.where(valid, v, -1)


def hex_ok(arr):
    b = arr.astype(np.int16)
    return ((b >= 48) & (b <= 57)) | ((b >= 97) & (b <= 102)) | ((b >= 65) & (b <= 70))


def magic_bits(data):
    """Per-position popcount distance of the 7-byte '|Block#' magic."""
    m = np.zeros(len(data) - 6, dtype=np.int16)
    for i in range(7):
        m += PC[data[i:i + len(m)] ^ MAGIC[i]]
    return m


def scan_block(args):
    b, path, keyfn = args
    key = np.fromfile(keyfn, dtype=np.uint8)
    with open(path, 'rb') as f:
        f.seek(b * BLK)
        raw = f.read(BLK + 64)
    if len(raw) < BLK + 64:
        raw = raw + bytes(64)
    data = np.frombuffer(raw, dtype=np.uint8) ^ np.concatenate([key, key[:64]])
    m = magic_bits(data)
    t = np.zeros(len(data) - 6, dtype=np.int16)
    for i in range(7):
        t += PC[data[i:i + len(t)] ^ TAIL[i]]

    n = len(data) - 60
    score = np.zeros(n, dtype=np.int8)
    for off, ch in FMT:
        score += (data[off:off + n] == ch).astype(np.int8)
    cand = np.flatnonzero(score >= MINFMT)
    if len(cand) == 0:
        return b, [np.zeros(0, np.int32), np.zeros(0, np.int8), np.zeros(0, np.int16),
                   np.zeros(0, np.int8), np.zeros(0, np.int8)]
    w = cand[:, None]
    d12 = digits_to_int(data[w + np.arange(7, 19)])
    f12 = fixdecimal(data[w + np.arange(7, 19)])
    b20 = fixdecimal(data[w + np.arange(39, 59)])
    hx = hex_ok(data[w + np.arange(23, 31)]).all(1)
    val = np.where(d12 >= 0, d12, f12)          # prefer an intact field
    strict = d12 >= 0
    selfok = (f12 >= 0) & (b20 >= 0) & (f12 * 512 == b20)
    inr = (val >= ECCSTART) & (val <= ECCEND) & ((val - ECCSTART) % INC == 0)
    idx = ((val - ECCSTART) // INC).astype(np.int64)
    keep = inr & np.isin(idx, TARGET_IDX)
    if not keep.any():
        return b, [np.zeros(0, np.int32), np.zeros(0, np.int8), np.zeros(0, np.int16),
                   np.zeros(0, np.int8), np.zeros(0, np.int8)]
    grade = np.where(selfok, 0, np.where(hx, 2, 1))[keep]
    return b, [idx[keep].astype(np.int32), score[cand[keep]].astype(np.int8),
               m[cand[keep]], (strict[keep] & 1).astype(np.int8), grade]


TARGET_IDX = np.zeros(0, dtype=np.int64)


def _init(targets):
    # Windows spawns the workers, so a global set in the parent is invisible to
    # them; hand each worker the target grid indexes through the initializer.
    global TARGET_IDX
    TARGET_IDX = targets


def main():
    global TARGET_IDX
    tl = np.loadtxt(TARGETS, dtype=np.int64)
    tl = np.atleast_1d(tl).astype(np.int64)
    off = tl - ECCSTART
    ok = (off >= 0) & (off <= ECCEND - ECCSTART) & (off % INC == 0)
    TARGET_IDX = np.unique(off[ok] // INC)
    print('targets: %d LBAs read, %d on the searched grid, %d off-grid'
          % (len(tl), len(TARGET_IDX), int((~ok).sum())), flush=True)

    key = np.fromfile(KEYFILE, dtype=np.uint8)
    assert len(key) == BLK, 'key must be exactly one block'
    size = os.path.getsize(DUMP)
    nblocks = size // BLK
    if MAXBLK:
        nblocks = min(nblocks, START + MAXBLK)
    blocks = list(range(START, nblocks))
    print('%s: %d blocks; format score >= %d of %d, grid %d..%d step %d, pool %d'
          % (DUMP, len(blocks), MINFMT, NFMT, ECCSTART, ECCEND, INC, POOL), flush=True)

    best_grade = np.full(NPOINTS, 9, dtype=np.int8)
    best_fmt = np.zeros(NPOINTS, dtype=np.int8)
    best_mag = np.full(NPOINTS, 255, dtype=np.int16)
    best_strict = np.zeros(NPOINTS, dtype=np.int8)
    hits_total = 0
    done = 0
    with Pool(POOL, initializer=_init, initargs=(TARGET_IDX,)) as pool:
        for b, (idx, sc, mb, st, gr) in pool.imap_unordered(
                scan_block, ((x, DUMP, KEYFILE) for x in blocks), chunksize=1):
            done += 1
            hits_total += len(idx)
            if len(idx) == 0:
                continue
            g = np.minimum(best_grade[idx], gr)
            best_grade[idx] = g
            best_fmt[idx] = np.maximum(best_fmt[idx], sc)
            best_strict[idx] = np.maximum(best_strict[idx], st)
            np.minimum.at(best_mag, idx, mb)
            if done % 200 == 0:
                rec = int((best_grade < 9).sum())
                print('  %d/%d blocks  target hits %d  grid points recovered %d'
                      % (done, len(blocks), hits_total, rec), flush=True)

    for g, name in ((0, 'A self-consistent record'), (1, 'B sector number only'),
                    (2, 'C sector number, hex field intact')):
        print('grade %s : %d' % (name, int((best_grade == g).sum())))
    rec = best_grade < 9
    trec = rec[TARGET_IDX]
    tgr, tfmt, tmag, tst = (a[TARGET_IDX[trec]] for a in (best_grade, best_fmt, best_mag, best_strict))
    print('\ngrid points recovered by this pass : %d / %d targets'
          % (int(trec.sum()), len(TARGET_IDX)))
    print('still unreadable                   : %d' % int((~trec).sum()))
    print('exact-magic hits among them        : %d' % int((tmag == 0).sum()))
    print('format score distribution          : %s'
          % dict(zip(*[x.tolist() for x in np.unique(tfmt, return_counts=True)])))
    print('sector field intact (no repair)    : %d' % int((tst == 1).sum()))
    lb = ECCSTART + TARGET_IDX[trec] * INC
    with open(OUT, 'w') as f:
        f.write('\n'.join('%d %d %d %d %d' % t for t in
                          zip(lb, TARGET_IDX[trec], tfmt, tmag, tgr)) + '\n')
    print('recovered LBA grididx fmtscore magicbits grade : %s' % OUT)


if __name__ == '__main__':
    main()
