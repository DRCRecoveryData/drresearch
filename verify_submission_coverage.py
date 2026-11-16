#!/usr/bin/env python3
"""Prove whether every credited record's bytes are physically inside the submitted extraction.

dumpextractrelevant.pl writes a page when its digit fields parse, or when the page still carries
the 'P00000' sector literal, so the tool's "Missing: 1485 => 99 %" line scores the *address fields*,
not the file. The positional sweep credits all 2,064,385 grid points. Those two statements can both
be true only if the extraction already contains the records whose digits rotted. This measures that
directly, with no rule of mine in the loop:

  A  sha-stub every page of the extraction (blake2b-8)              -> digest : extraction page index
  B  descramble every page the map says a credited record lives in  -> hit / miss against A
  C  for the sampled records, compare the 82 B label bytes in the extraction page against the
     same 82 B read from the source dump at the same in-page offset  -> byte equality proof

Sample set: all targets + every positional-only point (the ones consensus refused) + a bounded
random slice of tier-3 credits, so C costs seconds and still covers the interesting cases.

Usage:  python verify_submission_coverage.py
Env:    SRC UPLOAD XORKEY MAP TARGETS OUT PPB PGSZ RAND3
"""
import os
import sys
import time
import hashlib
import random
import tempfile
import numpy as np

BLKSZ = int(os.environ.get('BLKSZ', 21233664))
PGSZ = int(os.environ.get('PGSZ', 18432))
PPB = BLKSZ // PGSZ                       # 1152 pages per block
RECLEN = 82                               # '|Block#%012d (0x%08X) Byte: %020d Pos: %10d MB\n***'

HERE = os.path.dirname(os.path.abspath(__file__))


def _p(path):
    """MSYS /tmp is %LOCALAPPDATA%\\Temp, which python does not resolve."""
    if path.startswith('/tmp/'):
        return os.path.join(tempfile.gettempdir(), path[5:])
    return path


SRC = _p(os.environ.get('SRC', 'C:/Users/drclab/Documents/LDPC/01_01_plain.dump'))
UPLOAD = _p(os.environ.get('UPLOAD', '/tmp/up_full_v3.dump'))
XORKEY = _p(os.environ.get('XORKEY', os.path.join(HERE, '..', 'run', '01_01_plain.dump.xor')))
MAP = _p(os.environ.get('MAP', os.path.join(HERE, 'anchor_map_full.txt')))
TARGETS = _p(os.environ.get('TARGETS', os.path.join(HERE, 'real_hard_missing.txt')))
OUT = _p(os.environ.get('OUT', os.path.join(HERE, 'submission_coverage.txt')))
RAND3 = int(os.environ.get('RAND3', '2000'))
RECORD_RANGE = int(os.environ.get('RECORD_RANGE', 512))   # a record lives inside one 512 B sector

COLS = ('LBA', 'off', 'tier', 'base_src', 'labelbits', 'digitbits', 'totalbits',
        'rivalbits', 'blk', 'page', 'sect')
LAB = {c: i for i, c in enumerate(COLS)}


def dig(chunk):
    return hashlib.blake2b(chunk, digest_size=8).digest()


def load_map():
    a = np.atleast_2d(np.loadtxt(MAP, dtype=np.int64, comments='#'))
    if a.shape[1] != len(COLS):
        raise SystemExit('map %s has %d columns, expected %d' % (MAP, a.shape[1], len(COLS)))
    return a


def phase_a(nproc):
    """Stream the extraction once, hash every page."""
    t0 = time.time()
    seen = {}
    dupes = 0
    n = 0
    with open(UPLOAD, 'rb', buffering=64 * PGSZ) as f:
        while True:
            buf = f.read(64 * PGSZ)
            if not buf:
                break
            if len(buf) < PGSZ:
                nproc('trailing short read of %d bytes at page %d - the file is not a whole '
                      'number of pages' % (len(buf), n))
                break
            for i in range(0, len(buf) - PGSZ + 1, PGSZ):
                d = dig(buf[i:i + PGSZ])
                if d in seen:
                    dupes += 1
                seen[d] = n
                n += 1
    nproc('extraction: %d pages (%.6f GB), %d distinct digests, %d repeats, %.0f s'
          % (n, n * PGSZ / 10 ** 9, len(seen), dupes, time.time() - t0))
    return seen, n


def main():
    for path in (SRC, UPLOAD, XORKEY, MAP, TARGETS):
        if not os.path.exists(path):
            raise SystemExit('missing input: %s' % path)
    lines = []
    w = lines.append

    def nproc(msg):
        print('%s %s' % (time.strftime('%H:%M:%S'), msg), flush=True)

    a = load_map()
    key = np.frombuffer(open(XORKEY, 'rb').read(), np.uint8)
    if key.size % PGSZ:
        raise SystemExit('xor key %d B is not a whole number of %d B pages' % (key.size, PGSZ))
    tiled = key.size == BLKSZ
    nproc('map rows %d, key %d B (%s)' % (len(a), key.size,
                                          'one block, tiled' if tiled else 'dump-sized'))

    cred = a[a[:, LAB['tier']] > 0]
    need_all = np.unique(cred[:, LAB['blk']] * PPB + cred[:, LAB['page']])
    tg = np.unique(np.atleast_1d(np.loadtxt(TARGETS, dtype=np.int64)))
    trow = np.isin(a[:, LAB['LBA']], tg)
    nproc('grid points %d, credited %d, targets %d, target rows %d'
          % (len(a), len(cred), len(tg), int(trow.sum())))
    nproc('distinct pages holding a credited record: %d' % len(need_all))

    idx, npages = phase_a(nproc)
    nproc('phase B: %d needed pages descrambled and tested' % len(need_all))

    t1 = time.time()
    present = 0
    absent_pages = []
    with open(SRC, 'rb', buffering=PGSZ) as fsrc:
        for pidx in need_all:
            blk, pi = divmod(int(pidx), PPB)
            fsrc.seek(blk * BLKSZ + pi * PGSZ)
            raw = fsrc.read(PGSZ)
            if len(raw) != PGSZ:
                absent_pages.append(int(pidx))
                continue
            page = (np.frombuffer(raw, np.uint8) ^ key[pi * PGSZ:(pi + 1) * PGSZ]).tobytes()
            if dig(page) in idx:
                present += 1
            else:
                absent_pages.append(int(pidx))
    nproc('needed pages present in extraction: %d / %d (%.4f%%) in %.0f s'
          % (present, len(need_all), 100.0 * present / max(len(need_all), 1), time.time() - t1))

    # ---- phase C: byte-for-byte record proof on the interesting subset ----
    only_pos = (a[:, LAB['tier']] == 1) | (a[:, LAB['tier']] == 2)
    rng = random.Random(6767)
    t3 = np.flatnonzero(a[:, LAB['tier']] == 3)
    samp = np.concatenate([np.flatnonzero(trow), np.flatnonzero(only_pos),
                           rng.sample(list(t3), min(RAND3, len(t3)))])
    nproc('phase C: %d records checked byte-for-byte' % len(samp))

    cache = {}
    ok = bad = nohit = 0
    details = []
    with open(SRC, 'rb', buffering=PGSZ) as fsrc, open(UPLOAD, 'rb', buffering=PGSZ) as fup:
        for i in samp:
            blk, pg, off = int(a[i, LAB['blk']]), int(a[i, LAB['page']]), int(a[i, LAB['off']])
            pidx = blk * PPB + pg
            inp = off - blk * BLKSZ - pg * PGSZ
            if inp < 0 or inp + RECLEN > PGSZ:
                bad += 1
                details.append('row %d: offset %d is not inside page %d' % (i, inp, pidx))
                continue
            if pidx not in cache:
                fsrc.seek(blk * BLKSZ + pg * PGSZ)
                raw = fsrc.read(PGSZ)
                if len(raw) != PGSZ:
                    nohit += 1
                    continue
                cache[pidx] = (np.frombuffer(raw, np.uint8)
                               ^ key[pg * PGSZ:(pg + 1) * PGSZ]).tobytes()
            want = cache[pidx][inp:inp + RECLEN]
            up = idx.get(dig(cache[pidx]))
            if up is None:
                nohit += 1
                continue
            fup.seek(up * PGSZ + inp)
            got = fup.read(RECLEN)
            if got == want:
                ok += 1
            else:
                bad += 1
                if len(details) < 20:
                    details.append('LBA %d: %d of %d label bytes differ'
                                   % (a[i, LAB['LBA']], sum(x != y for x, y in zip(got, want)),
                                      RECLEN))

    w('submission coverage - does the extraction physically contain the credited records?')
    w('  extraction file     : %s  (%d B, %d pages of %d B)'
      % (UPLOAD, os.path.getsize(UPLOAD), npages, PGSZ))
    w('  source dump         : %s' % SRC)
    w('  xor key             : %s  (%d B, %s)' % (XORKEY, key.size,
                                                  'one block, tiled' if tiled else 'dump-sized'))
    w('  anchor map          : %s  (%d credited rows)' % (MAP, len(cred)))
    w('')
    w('page-level presence (every page that holds a credited record):')
    w('  present in extraction : %d / %d  (%.4f %%)'
      % (present, len(need_all), 100.0 * present / max(len(need_all), 1)))
    w('  absent                : %d' % len(absent_pages))
    if absent_pages:
        w('  first absent pages    : %s' % ', '.join(str(x) for x in absent_pages[:12]))
    w('')
    w('record-level byte proof (%d records: %d targets, %d tier-1/2, %d sampled tier-3):'
      % (len(samp), int(trow.sum()), int(only_pos.sum()), min(RAND3, len(t3))))
    w('  byte-identical in extraction : %d' % ok)
    w('  differing                    : %d' % bad)
    w('  page not in extraction       : %d' % nohit)
    for d in details:
        w('  ! %s' % d)
    w('')
    verdict = (present == len(need_all) and bad == 0 and nohit == 0 and ok == len(samp))
    w('verdict: %s' % ('the extraction contains the bytes of every credited record, including the '
                       '%d the tool cannot read an address out of - the 99 %% line is its own '
                       'digit gate, not a hole in the file' % len(tg)
                       if verdict else
                       'INCOMPLETE - see the absent / differing counts above'))

    text = '\n'.join(lines) + '\n'
    with open(OUT, 'w') as f:
        f.write(text)
    print(text)
    nproc('wrote %s' % OUT)
    return 0 if verdict else 1


if __name__ == '__main__':
    sys.exit(main())
