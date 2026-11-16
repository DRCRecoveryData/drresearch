#!/usr/bin/env python3
"""Recover anchors whose digits are too rotted to parse, by structural consensus.

The pattern record carries more redundancy than dumpextractrelevant.pl uses:

  |Block# %012d  (0x%08X)  Byte: %020d  Pos: %10d MB\n***
     0..6  7..18   20..30      33..58        60..74    76..81

* 32 bytes are CONSTANT (labels, spaces, the newline, the ***) = 256 bits that no
  LBA can change, so a record is recognisable even when every digit is gone.
* the sector number is written three times exactly (decimal, hex, byte offset =
  S*512) and once coarsely (Pos: = S>>11).  Any one of them can be repaired from
  the others.

anchor_coverage.py and hard_probe.py both required the digits to *parse*, so a
record with two rotted bytes in the 12-digit field and one in the 20-digit field
was thrown away although three encodings still agreed.  This pass accepts a slot
when the constant skeleton is within TOL_SKEL bits of the true skeleton, at least
TOL_FIELDS of the three exact encodings land on one value after <= FLIP bit flips
each, and the coarse Pos: field is consistent with that value.  The bit-distance to
rec(L) and the runner-up vote are reported, so an acceptance is evidence not guess.

KAT mode builds a synthetic dump with known records and known injected flips and
checks four things: records with 1-4 flips come back, records whose digits are
obliterated are not fabricated at their old value, records whose body is erased are
not recovered, and filler bytes yield no acceptances at all.
"""
import os, sys
import numpy as np
from multiprocessing import Pool

BLK = int(os.environ.get('BLKSZ', 21233664))
PG = int(os.environ.get('PGSZ', 18432))
CH = int(os.environ.get('CHUNK', 1144))
ECCSTART = int(os.environ.get('ECCSTART', 3145728))
ECCEND = int(os.environ.get('ECCEND', 21725184))
INC = int(os.environ.get('INC', 9))
POOL = int(os.environ.get('POOL', 10))
TOL_SKEL = int(os.environ.get('TOL_SKEL', 12))
TOL_FIELDS = int(os.environ.get('TOL_FIELDS', 2))
FLIP = int(os.environ.get('FLIP', 1))
MAXDIST = int(os.environ.get('MAXDIST', 6))      # bits away from the ideal record
MARGIN = int(os.environ.get('MARGIN', 4))        # how much better than the rival LBA
TARG = os.environ.get('TARGETS', 'real_hard_missing.txt')
OUT = os.environ.get('OUT', 'repaired_anchors.txt')
DUMP = sys.argv[1] if len(sys.argv) > 1 else 'C:/Users/drclab/Documents/LDPC/01_01_plain.dump'
KEYFILE = os.environ.get('XORKEY', 'C:/Users/drclab/Documents/Qoder/2026-11-15/7246e382/run/01_01_plain.dump.xor')
MAXBLK = int(os.environ.get('MAXBLK', 0))
START = int(os.environ.get('START', 0))

PC = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(1).astype(np.int16)
NREC = 82
SKEL = [0, 1, 2, 3, 4, 5, 6, 19, 20, 21, 22, 31, 32, 33, 34, 35, 36, 37, 38, 59,
        60, 61, 62, 63, 64, 75, 76, 77, 78, 79, 80, 81]
# (lo, hi, base, divisor, padding-allowed).  Only Pos: is space padded (%10d); the
# other three are zero padded, so a space inside them is damage, not formatting.
EXACT = [(7, 19, 10, 1, False), (23, 31, 16, 1, False), (39, 59, 10, 512, False)]
MBF = (65, 75, 10, 1, True)


def rec(L, overwritten=False):
    s = f"|Block#{L:012d} (0x{L:08X}) Byte: {L*512:020d} Pos: {L>>11:10d} MB\n***"
    if overwritten:
        s += " OVERWRITTEN"
    return s.encode()


SKELREF = np.frombuffer(rec(0)[:NREC], np.uint8)[SKEL]


def _ok(c, base, pad):
    if 48 <= c <= 57:
        return True
    if pad and c == 32:
        return True
    return base == 16 and ((97 <= c <= 102) or (65 <= c <= 70))


def candidates(raw, base, depth, pad):
    """Values this field supports with at most `depth` bit flips."""
    vals = set()
    txt = [chr(c) for c in raw]
    if all(_ok(c, base, pad) for c in raw):
        try:
            vals.add(int(''.join(txt), base))
        except ValueError:
            pass
    if depth < 1:
        return vals
    n = len(raw)
    for i in range(n):
        for bit in range(8):
            c = raw[i] ^ (1 << bit)
            if not _ok(c, base, pad):
                continue
            t = txt[:]
            t[i] = chr(c)
            try:
                vals.add(int(''.join(t), base))
            except ValueError:
                pass
            if depth >= 2 and base != 512:
                for j in range(i + 1, n):
                    for bit2 in range(8):
                        c2 = raw[j] ^ (1 << bit2)
                        if not _ok(c2, base, pad):
                            continue
                        t2 = t[:]
                        t2[j] = chr(c2)
                        try:
                            vals.add(int(''.join(t2), base))
                        except ValueError:
                            pass
    return vals


def classify(data, p):
    """(L, grididx, skel_bits, votes, dist, runner) for the 82-byte record at p, else None."""
    obs = np.frombuffer(data[p:p + NREC], np.uint8)
    if len(obs) < NREC:
        return None
    skel = int(PC[obs[SKEL] ^ SKELREF].sum())
    if skel > TOL_SKEL:
        return None
    votes = {}
    for lo, hi, base, div, pad in EXACT:
        depth = FLIP if (hi - lo) <= 12 or base == 16 else min(FLIP, 1)
        for v in candidates(obs[lo:hi], base, depth, pad):
            if div != 1 and v % div:
                continue
            L = v // div
            votes[L] = votes.get(L, 0) + 1
    lo, hi, base, div, pad = MBF
    mbset = candidates(obs[lo:hi], base, min(FLIP, 1), pad)
    obsb = np.array(obs, np.uint8)
    ranked = sorted(votes.items(), key=lambda kv: (-kv[1], kv[0]))
    cands = []
    for L, v in ranked:
        if v < TOL_FIELDS:
            break
        if L < ECCSTART or L > ECCEND or (L - ECCSTART) % INC:
            continue
        if L >> 11 not in mbset:
            continue
        cands.append((int(PC[obsb ^ np.frombuffer(rec(L)[:NREC], np.uint8)].sum()), L, v))
    if not cands:
        return None
    cands.sort()
    dist, L, v = cands[0]
    if dist > MAXDIST:
        return None
    second = cands[1][0] if len(cands) > 1 else 10 ** 9
    if second - dist < MARGIN:
        return None                                   # two LBAs explain it almost equally well
    return L, (L - ECCSTART) // INC, skel, v, dist, second


def scan_block(args):
    b, path, keyfn = args
    key = np.fromfile(keyfn, dtype=np.uint8)
    with open(path, 'rb') as f:
        f.seek(b * BLK)
        raw = f.read(BLK)
    data = np.frombuffer(raw, dtype=np.uint8) ^ key
    n = len(data)
    pages = n // PG
    slots = (np.arange(pages)[:, None, None, None] * PG +
             np.arange(PG // CH)[None, :, None, None] * CH +
             np.array([0, 512])[None, None, None, :]).ravel()
    slots = slots[slots + NREC <= n]
    sk = np.zeros(len(slots), np.int16)
    for k, j in enumerate(SKEL):
        sk += PC[data[slots + j] ^ SKELREF[k]]
    g, d, s, v = [], [], [], []
    for p in slots[sk <= TOL_SKEL].tolist():
        r = classify(data, p)
        if r:
            L, gi, skel, votes, dist, second = r
            g.append(gi); d.append(dist); s.append(skel); v.append(votes)
    return b, (np.array(g, np.int64), np.array(d, np.int16),
               np.array(s, np.int16), np.array(v, np.int8))


def main():
    if os.environ.get('KAT'):
        return kat()
    tl = np.atleast_1d(np.loadtxt(TARG, dtype=np.int64))
    targets = set(((tl - ECCSTART) // INC).tolist())
    nblocks = os.path.getsize(DUMP) // BLK
    blocks = list(range(START, min(nblocks, START + MAXBLK) if MAXBLK else nblocks))
    print('%s: %d blocks, skeleton <= %d bits, %d/%d exact fields must agree, flips <= %d, %d targets'
          % (DUMP, len(blocks), TOL_SKEL, TOL_FIELDS, len(EXACT), FLIP, len(targets)), flush=True)
    allg, alld, alls, allv = [], [], [], []
    done = 0
    with Pool(POOL) as pool:
        for b, (g, d, s, v) in pool.imap_unordered(scan_block, ((x, DUMP, KEYFILE) for x in blocks), chunksize=2):
            if len(g):
                allg.append(g); alld.append(d); alls.append(s); allv.append(v)
            done += 1
            if done % 200 == 0:
                print('  %d/%d blocks, %d records accepted'
                      % (done, len(blocks), sum(len(x) for x in allg)), flush=True)
    g = np.concatenate(allg) if allg else np.zeros(0, np.int64)
    d = np.concatenate(alld) if allg else np.zeros(0, np.int16)
    s = np.concatenate(alls) if allg else np.zeros(0, np.int16)
    v = np.concatenate(allv) if allg else np.zeros(0, np.int8)
    order = np.lexsort((d, g))
    g, d, s, v = g[order], d[order], s[order], v[order]
    first = np.r_[True, g[1:] != g[:-1]]
    uniq, ud, us, uv = g[first], d[first], s[first], v[first]
    grid = {int(x[0]): (int(x[1]), int(x[2]), int(x[3])) for x in zip(uniq, ud, us, uv)}
    got = targets & set(grid)
    print('\nrecords accepted overall : %d (%d distinct grid points of %d)'
          % (len(g), len(grid), (ECCEND - ECCSTART) // INC + 1))
    print('coverage of the whole grid: %.4f%%' % (100.0 * len(grid) / ((ECCEND - ECCSTART) // INC + 1)))
    print('targets recovered        : %d / %d' % (len(got), len(targets)))
    print('still missing            : %d' % len(targets - set(grid)))
    if got:
        dg = np.array([grid[x][0] for x in sorted(got)])
        sk = np.array([grid[x][1] for x in sorted(got)])
        print('repaired: bit-distance to rec(L) min %d median %d max %d'
              % (dg.min(), int(np.median(dg)), dg.max()))
        print('repaired: skeleton bits          min %d median %d max %d'
              % (sk.min(), int(np.median(sk)), sk.max()))
        hist = {}
        for x in sorted(got):
            hist[grid[x][0]] = hist.get(grid[x][0], 0) + 1
        print('repaired distance histogram: %s' % ' '.join('%d:%d' % kv for kv in sorted(hist.items())))
    with open(OUT, 'w') as f:
        f.write('# LBA grididx skeleton_bits field_votes dist_to_expected\n')
        for x in sorted(got):
            dist, skel, votes = grid[x]
            f.write('%d %d %d %d %d\n' % (ECCSTART + x * INC, x, skel, votes, dist))
    np.savetxt(OUT + '.allgrid.txt', uniq.astype(np.int64), fmt='%d')
    print('full accepted grid       : %s.allgrid.txt' % OUT)
    print('table                    : %s' % OUT)
    return len(got)


def kat():
    blk, nrec = 2048 * CH, 1200
    rng = np.random.default_rng(5)
    buf = bytearray(b'\x5a' * blk)
    lbs = []
    for i in range(nrec):
        off = i * CH
        L = ECCSTART + i * INC
        buf[off:off + NREC] = rec(L)
        buf[off + NREC:off + CH] = b'x' * (CH - NREC)
        lbs.append((L, off))
    plan = {1: 150, 2: 150, 3: 150, 4: 60}
    idx = list(range(len(lbs)))
    rng.shuffle(idx)
    cur, groups = 0, {}
    for nb, cnt in sorted(plan.items()):
        groups[nb] = idx[cur:cur + cnt]
        for j in groups[nb]:
            off = lbs[j][1]
            for k in range(nb):
                pos = off + 7 + int(rng.integers(0, 68))
                buf[pos] ^= np.uint8(1 << int(rng.integers(0, 8)))
        cur += cnt
    doomed = idx[cur:cur + 30]
    for j in doomed:
        # perfect skeleton, digits rewritten to a value that is NOT on the grid:
        # the only honest outcome is "no acceptance", so a hit here is a fabrication.
        off = lbs[j][1]
        bad = ECCSTART + 1 + int(rng.integers(0, 5000)) * 9
        r = rec(bad)
        buf[off + 7:off + 19] = r[7:19]
        buf[off + 23:off + 31] = r[23:31]
        buf[off + 39:off + 59] = r[39:59]
        buf[off + 65:off + 75] = r[65:75]
    erased = idx[cur + 30:cur + 60]
    for j in erased:
        buf[lbs[j][1]:lbs[j][1] + CH] = b'\x11' * CH
    open('/tmp/kat_repair.dump', 'wb').write(bytes(buf))
    open('/tmp/kat_repair.dump.xor', 'wb').write(bytes(blk))

    globals()['BLK'] = blk
    globals()['PG'] = CH
    globals()['ECCEND'] = ECCSTART + nrec * INC
    b, (gg, dd, ss, vv) = scan_block((0, '/tmp/kat_repair.dump', '/tmp/kat_repair.dump.xor'))
    res = list(zip(gg.tolist(), dd.tolist(), ss.tolist(), vv.tolist()))
    got = set(gg.tolist())
    truth = {int((L - ECCSTART) // INC) for L, off in lbs}

    def gs(j):
        return int((lbs[j][0] - ECCSTART) // INC)

    print('KAT records written : %d   accepted : %d   (mean bit-distance %.2f, max %d)'
          % (len(lbs), len(res), float(np.mean([r[1] for r in res])), max(r[1] for r in res)))
    clean = set(truth)
    for nb in sorted(plan):
        tg = {gs(j) for j in groups[nb]}
        clean -= tg
        print('  %d injected bit flip(s): recovered %4d / %4d   (%.1f%%)'
              % (nb, len(tg & got), len(tg), 100.0 * len(tg & got) / len(tg)))
    for nb in sorted(plan):
        miss = sorted({gs(j) for j in groups[nb]} - got)
        if not miss:
            continue
        d = np.frombuffer(bytes(buf), np.uint8)
        print('  -- %d-flip misses (%d):' % (nb, len(miss)))
        for g in miss[:4]:
            off = int(g) * CH
            obs = d[off:off + NREC]
            print('     g=%d  d12=%r hex=%r byte=%r pos=%r skelbits=%d'
                  % (g, bytes(obs[7:19]), bytes(obs[23:31]), bytes(obs[39:59]), bytes(obs[65:75]),
                     int(PC[obs[SKEL] ^ SKELREF].sum())))
    dg = {gs(j) for j in doomed}
    eg = {gs(j) for j in erased}
    clean -= dg | eg
    print('  non-grid digits    : accepted at all %d / %d (must be 0)' % (len(dg & got), len(dg)))
    print('  body erased        : recovered %d / %d (must be 0)' % (len(eg & got), len(eg)))
    print('  untouched controls : recovered %d / %d (must be all)' % (len(clean & got), len(clean)))
    fp = sorted(got - truth)
    print('  fabricated points  : %d %s (must be 0 - proves the 256-bit skeleton cannot be hit by chance)'
          % (len(fp), fp[:5]))
    ok = (len(eg & got) == 0 and len(dg & got) == 0 and len(clean & got) == len(clean) and not fp)
    for nb in (1, 2):
        tg = {gs(j) for j in groups[nb]}
        if len(tg & got) < 0.95 * len(tg):
            ok = False
    print('KAT %s' % ('PASS' if ok else 'FAIL'))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main() or 0)
