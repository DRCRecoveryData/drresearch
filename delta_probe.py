#!/usr/bin/env python3
"""Explain the delta between dumpextractrelevant.pl's Found count and the audit's.

For every sector whose 7-byte magic is within 3 bits of '|Block#' and whose parsed
LBA is one the tool found but the audit did not, record the true bit distance and
the raw bytes.  If those anchors are clean here too, the audit is losing them; if
they are rotted, the tool is matching them despite the rot, and the difference is
a detection rule we have to reproduce.
"""
import os, sys, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import anchor_coverage as ac

DELTA = sorted(int(l) for l in open(os.environ.get('DELTA', '../delta_lb.txt')) if l.strip())
DELTA_SET = set(DELTA)


def probe_block(args):
    b, path, keyfn = args
    key = np.fromfile(keyfn, dtype=np.uint8)
    with open(path, 'rb') as f:
        f.seek(b * ac.BLK)
        raw = f.read(ac.BLK + 64)
    if len(raw) < ac.BLK + 64:
        raw = raw + bytes(64)
    d = np.frombuffer(raw, dtype=np.uint8) ^ np.concatenate([key, key[:64]])
    m = np.zeros(len(d) - 6, dtype=np.int16)
    for i in range(7):
        m += ac.PC[d[i:i + len(m)] ^ ac.MAGIC[i]]
    cand = np.flatnonzero(m <= 3)
    keep = cand[(cand + 59 < len(d))]
    if len(keep) == 0:
        return b, 0, [], collections.Counter({0: 0})
    lbad = ac.digits_to_int(d[keep[:, None] + np.arange(7, 19)])
    lbad = np.where(lbad < 0, ac.fixdecimal(d[keep[:, None] + np.arange(7, 19)]), lbad)
    rows = []
    for pos, lba, bits in zip(keep, lbad, m[keep]):
        if int(lba) in DELTA_SET:
            rows.append((int(lba), int(bits), int(pos), d[pos:pos + 59].tobytes()))
    # how many candidate anchors at all, and their alignment
    aligned = int((keep % 512 == 0).sum())
    return b, aligned, rows, collections.Counter(int(x) for x in m[keep])


def main():
    nblocks = os.path.getsize(ac.DUMP) // ac.BLK
    if ac.MAXBLK:
        nblocks = min(nblocks, ac.MAXBLK)
    args = [(b, ac.DUMP, ac.KEYFILE) for b in range(ac.START, nblocks)]
    if ac.POOL > 1:
        import multiprocessing as mp
        it = mp.get_context('spawn').Pool(ac.POOL).imap(probe_block, args, chunksize=4)
    else:
        it = map(probe_block, args)
    dist = collections.Counter()
    examples = []
    seen = set()
    tot_exact = 0
    for done, (b, aligned, rows, c) in enumerate(it, 1):
        for lba, bits, pos, head in rows:
            dist[bits] += 1
            seen.add(lba)
            if len(examples) < 6:
                examples.append((b, pos, lba, bits, head))
        tot_exact += c[0]
        if done % 200 == 0:
            print('  %d/%d blocks, delta anchors matched so far %d' % (done, len(args), sum(dist.values())), flush=True)
    print('\nexact magic hits overall      : %d' % tot_exact)
    print('delta-LBA anchors by bit error:')
    for k in sorted(dist):
        print('   %d bad bit(s): %d' % (k, dist[k]))
    print('\n%d records covering %d distinct delta LBAs (of %d probed)'
          % (sum(dist.values()), len(seen), len(DELTA)))
    for b, pos, lba, bits, head in examples:
        print('\nblock %d offset %d (block-relative) LBA %d  magic %d bad bit(s)' % (b, pos, lba, bits))
        print('   %r' % (head,))
        print('   sector alignment: offset %% 512 = %d' % (pos % 512))


if __name__ == '__main__':
    main()
