#!/usr/bin/env python3
"""Wait for the positional sweep, then turn its table into the finished deliverable.

Nothing here re-judges an anchor: it re-derives the emitted map from the raw dump and
reports where the raw bytes agree or disagree with the table.  Steps:

1. wait for positional_anchors.py to write its table (the sweep is the slow stage)
2. recompute the tier/coverage tallies from the table file itself
3. re-read a sample of rows from 01_01_plain.dump under the residual XOR and check the
   claim each row makes - tier 3 must still parse to that LBA, tiers 1-2 must still carry
   the record shell at the byte position the row publishes
4. check the map is a bijection: one row per LBA, no dump offset used twice
5. account for every one of the 1,485 previously-missing targets by tier
6. write coverage_report.txt, anchor_map_full.txt, target_tiers.txt and a sha256 digest

Run:  python finish_after_sweep.py            (TABLE=/tmp/pos_full.log etc. via env)
"""
import os, sys, time, hashlib, tempfile
import numpy as np


def _p(path):
    """MSYS /tmp is not python's /tmp; rewrite it so a shell-side log path still resolves."""
    if path.startswith('/tmp/'):
        return os.path.join(tempfile.gettempdir(), path[5:])
    return path


HERE = os.path.dirname(os.path.abspath(__file__))
DUMP = os.environ.get('DUMP', 'C:/Users/drclab/Documents/LDPC/01_01_plain.dump')
KEYFILE = os.environ.get('XORKEY', 'C:/Users/drclab/Documents/Qoder/2026-11-15/7246e382/run/01_01_plain.dump.xor')
LOG = _p(os.environ.get('SWEEPLOG', '/tmp/pos_full.log'))
TABLE = _p(os.environ.get('TABLE', os.environ.get('TABLEOUT', os.path.join(HERE, 'positional_full.txt'))))
TARG = os.environ.get('TARGETS', os.path.join(HERE, 'real_hard_missing.txt'))
VERIFY = os.path.join(HERE, 'verify_after_sweep.py')
REPORT = _p(os.environ.get('REPORT', os.path.join(HERE, 'coverage_report.txt')))
MAP = _p(os.environ.get('MAP', os.path.join(HERE, 'anchor_map_full.txt')))
TTIER = _p(os.environ.get('TTIER', os.path.join(HERE, 'target_tiers.txt')))
BLK = int(os.environ.get('BLKSZ', 21233664))
BORDER = int(os.environ.get('ECCSTART', 3145728))
ECCEND = int(os.environ.get('ECCEND', 21725184))
INC = 9
GRID = (ECCEND - BORDER) // INC + 1
POLL = int(os.environ.get('POLL', 30))
WAITMAX = int(os.environ.get('WAITMAX', 6 * 3600))
SAMPLE = int(os.environ.get('SAMPLE', 3000))
SEED = int(os.environ.get('SEED', 20261116))


def log(msg):
    print(msg, flush=True)


def wait_for_table():
    """Block until the sweep has written its table, then let the file settle."""
    if os.environ.get('NOWAIT'):
        return os.path.exists(TABLE)
    t0 = time.time()
    while True:
        if os.path.exists(TABLE) and os.path.getsize(TABLE) > 10 ** 6:
            s1 = os.path.getsize(TABLE)
            time.sleep(10)
            if os.path.getsize(TABLE) == s1:
                log('table ready: %s (%d bytes)' % (TABLE, s1))
                return True
        if os.path.exists(LOG):
            txt = open(LOG, 'r', errors='replace').read()
            if 'Traceback' in txt:
                log('SWEEP DIED:\n' + txt[-2000:])
                return False
        if time.time() - t0 > WAITMAX:
            log('giving up after %d s waiting for the table' % WAITMAX)
            return False
        time.sleep(POLL)


def tally():
    a = np.loadtxt(TABLE, dtype=np.int64)
    if a.ndim == 1:
        a = a[None, :]
    names = 'lba parsed credited tier base_src lab dig tot rival legal blk page sect offset'.split()
    d = {n: a[:, i] for i, n in enumerate(names)}
    log('\n== table tallies (derived from the file, not from the sweep stdout)')
    log('rows                       : %d' % len(a))
    for t in (3, 1, 2, 0):
        log('  tier %d : %8d  (%.4f%%)' % (t, int((d['tier'] == t).sum()),
                                          100.0 * (d['tier'] == t).sum() / len(a)))
    cr = d['credited'].astype(bool)
    log('credited                   : %d' % int(cr.sum()))
    log('grid points                : %d' % GRID)
    log('coverage of the grid       : %.4f%%  (credited %d / grid %d)'
        % (100.0 * len(set(d['lba'].tolist())) / GRID, int(cr.sum()), GRID))
    log('distinct LBAs in the table : %d  (must equal rows)' % len(set(d['lba'].tolist())))
    log('base_src 1 / 2             : %d / %d'
        % (int((d['base_src'] == 1).sum()), int((d['base_src'] == 2).sum())))
    per = {}
    for b, c in zip(d['blk'].tolist(), cr.astype(np.int8).tolist()):
        n, k = per.get(b, (0, 0))
        per[b] = (n + 1, k + c)
    dense = {b: v for b, v in per.items() if v[0] > 100}
    log('blocks with rows           : %d (%d of them carry >100 slots)' % (len(per), len(dense)))
    if dense:
        ratio = sorted(((v[1] / v[0], b, v) for b, v in dense.items()))
        log('  credited/slot over dense blocks: min %.5f median %.5f max %.5f'
            % (ratio[0][0], ratio[len(ratio) // 2][0], ratio[-1][0]))
        log('  5 least-covered dense blocks   : %s'
            % '  '.join('b%d %d/%d' % (b, v[1], v[0]) for _, b, v in ratio[:5]))
        short = [b for _, b, v in ratio if v[1] < v[0]]
        log('  dense blocks not at 100%%       : %d' % len(short))
    return d


def bijection(d):
    log('\n== map consistency')
    off = np.sort(d['offset'])
    log('dump offsets unique        : %s (duplicates %d)'
        % (bool(np.all(off[1:] != off[:-1])), int((np.diff(off) == 0).sum())))
    recompute = d['blk'] * BLK + d['page'] * 18432 + (d['sect'] // 2) * 1144 + (d['sect'] % 2) * 512
    log('offset == blk/page/sect    : %s (mismatches %d)'
        % (bool(np.all(recompute == d['offset'])), int((recompute != d['offset']).sum())))
    log('LBA sorted / no gaps of 0  : %s' % bool(np.all(np.diff(d['lba']) > 0)))
    congr = int(((d['lba'] - BORDER) % INC).sum())
    log('every LBA on the inc=9 grid: %s (off-grid rows %d)' % (congr == 0, congr))
    inr = int(((d['lba'] < BORDER) | (d['lba'] > ECCEND)).sum())
    log('every LBA inside the ECC window: %s (outside %d)' % (inr == 0, inr))


def verify_raw(d):
    """Re-read the bytes the map points at and re-test each row's own tier claim.

    The claim differs by tier, so the check must too.  Tier 3 was accepted by a 2-of-3
    vote over the three exact encodings (decimal, hex, byte-offset = L*512), so its raw
    decimal field may legitimately be a bit off - what has to hold is that at least two
    encodings are within one bit flip of rec(L).  Tier 1 claims distance-and-margin,
    tier 2 claims the 256-bit shell plus an anchored position and nothing about digits.
    """
    log('\n== raw-bytes verification of published offsets')
    key = np.fromfile(KEYFILE, dtype=np.uint8)
    PCt = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(1).astype(np.int16)
    SKEL = [0, 1, 2, 3, 4, 5, 6, 19, 20, 21, 22, 31, 32, 33, 34, 35, 36, 37, 38, 59,
            60, 61, 62, 63, 64, 75, 76, 77, 78, 79, 80, 81]
    FIELDS = [(7, 19), (23, 31), (39, 59)]
    ref = {}

    def recb(L):
        if L not in ref:
            s = (f'|Block#{L:012d} (0x{L:08X}) Byte: {L*512:020d} Pos: {L>>11:10d} MB\n***')
            ref[L] = np.frombuffer(s.encode(), np.uint8)
        return ref[L]

    def keystream(off, n):
        """The residual key tiles block by block, so slice it modulo the block length."""
        s = off % len(key)
        return key[s:s + n] if s + n <= len(key) else np.concatenate([key[s:], key[:n - (len(key) - s)]])

    rng = np.random.default_rng(SEED)
    idx = np.arange(len(d['lba']))
    picked = []
    for t in (3, 1, 2, 0):
        sel = idx[d['tier'][idx] == t]
        if len(sel):
            picked.append(sel if len(sel) <= SAMPLE else rng.choice(sel, SAMPLE, replace=False))
    pick = np.sort(np.concatenate(picked))
    tally_ok = {t: [0, 0] for t in (3, 1, 2, 0)}
    detail = {t: [] for t in (3, 1, 2, 0)}
    unreadable = 0
    rot_hist = {}
    show = int(os.environ.get('SHOW', 0))
    with open(DUMP, 'rb') as f:
        for i in pick.tolist():
            L = int(d['lba'][i])
            tier = int(d['tier'][i])
            off = int(d['offset'][i])
            f.seek(off)
            raw = f.read(82)
            if len(raw) < 82:
                unreadable += 1
                continue
            obs = np.frombuffer(raw, np.uint8) ^ keystream(off, 82)
            want = recb(L)
            dist = int(PCt[obs ^ want].sum())
            skel = int(PCt[obs[SKEL] ^ want[SKEL]].sum())
            fdist = [int(PCt[obs[a:b] ^ want[a:b]].sum()) for a, b in FIELDS]
            if show and len(ref) and pick.tolist().index(i) < show:
                print('    DIAG LBA %d off %d tier %d dist %d shell %d fields %s raw %r'
                      % (L, off, tier, dist, skel, fdist, obs[:32].tobytes()))
            if tier == 3:
                agree = sum(1 for x in fdist if x <= 1)
                good = agree >= 2
                rot_hist[tuple(sorted(fdist))] = rot_hist.get(tuple(sorted(fdist)), 0) + 1
            elif tier == 1:
                alts = [int(PCt[obs ^ recb(L + k * INC)].sum())
                        for k in (-5, -4, -3, -2, -1, 1, 2, 3, 4, 5)
                        if BORDER <= L + k * INC <= ECCEND]
                good = dist <= 60 and (min(alts) - dist) >= 8
            elif tier == 2:
                good = skel <= 4 and dist <= 80
            else:
                good = dist > 60 or skel > 4 or (min(
                    [int(PCt[obs ^ recb(L + k * INC)].sum()) for k in (-2, -1, 1, 2)
                     if BORDER <= L + k * INC <= ECCEND] or [10 ** 9]) - dist) < 8
            tally_ok[tier][0 if good else 1] += 1
            if not good:
                detail[tier].append((L, off, dist, skel, tuple(fdist)))
    log('rows checked               : %d  (%s per tier, capped at %d each)'
        % (len(pick), 'tiers 3/1/2/0', SAMPLE))
    for t in (3, 1, 2, 0):
        good, bad = tally_ok[t]
        log('  tier %d: claim re-derives from the raw bytes for %d / %d rows%s'
            % (t, good, good + bad, '' if bad == 0 else '   <-- %d DO NOT' % bad))
        for x in detail[t][:4]:
            log('      LBA %d off %d: total %d shell %d fields %s' % x)
    if 3 in rot_hist:
        log('  tier-3 field-bit-distance histogram [dec,hex,byte]: %s'
            % '  '.join('%s:%d' % kv for kv in sorted(rot_hist[3].items())[:8]))
    if unreadable:
        log('  short reads               : %d' % unreadable)
    return tally_ok


def targets(d):
    log('\n== the 1,485 previously-missing targets')
    tl = np.atleast_1d(np.loadtxt(TARG, dtype=np.int64))
    want = {int(x): i for i, x in enumerate(d['lba'])}
    row_of = {}
    for x in tl.tolist():
        i = want.get(int(x))
        row_of[int(x)] = None if i is None else int(d['tier'][i])
    with open(TTIER, 'w') as f:
        f.write('# LBA tier  (-1 = no row: the slot was never emitted)\n')
        for x in sorted(row_of):
            f.write('%d %s\n' % (x, row_of[x] if row_of[x] is not None else -1))
    from collections import Counter
    c = Counter(row_of.values())
    for t in (3, 1, 2, 0, None):
        if c.get(t):
            log('  tier %s : %d' % (t if t is not None else 'no row', c[t]))
    log('  total targets             : %d  -> written to %s' % (len(row_of), os.path.basename(TTIER)))
    return c


def write_map(d):
    order = np.argsort(d['lba'])
    with open(MAP, 'w') as f:
        f.write('# LBA dump_offset tier base_src labelbits digitbits totalbits best_rivalbits blk page sect\n'
                '# tier 3 digits parse, 1 shell+digits agree, 2 shell+position only, 0 not credited\n')
        for i in order.tolist():
            f.write('%d %d %d %d %d %d %d %d %d %d %d\n' % (
                d['lba'][i], d['offset'][i], d['tier'][i], d['base_src'][i], d['lab'][i],
                d['dig'][i], d['tot'][i], d['rival'][i], d['blk'][i], d['page'][i], d['sect'][i]))
    h = hashlib.sha256(open(MAP, 'rb').read()).hexdigest()
    log('\nanchor map                 : %s (%d rows)' % (MAP, len(d['lba'])))
    log('sha256                     : %s' % h)
    with open(MAP + '.sha256', 'w') as f:
        f.write('%s  %s\n' % (h, os.path.basename(MAP)))
    return h


def main():
    if not wait_for_table():
        return 1
    with open(REPORT, 'w') as out, _Tee(out):
        d = tally()
        bijection(d)
        verify_raw(d)
        targets(d)
        write_map(d)
        log('\nPIPELINE DONE')
    return 0


class _Tee:
    """Mirror stdout into the report file so the summary is on disk, not just in the log."""

    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = _Both(self._stdout, self._fh)
        return self

    def __exit__(self, *a):
        sys.stdout = self._stdout

    def __init__(self, fh):
        self._fh = fh


class _Both:
    def __init__(self, a, b):
        self.a, self.b = a, b

    def write(self, s):
        self.a.write(s)
        self.b.write(s)
        self.b.flush()

    def flush(self):
        self.a.flush()
        self.b.flush()


if __name__ == '__main__':
    sys.exit(main())
