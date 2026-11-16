#!/usr/bin/env python3
"""Control for hard_probe.py, the last pass for anchors the audit calls hard-missing.

Two classes of damage, both invisible to anchor_coverage.py:
  rot_fields   magic obliterated (56 bits, beyond any tolerance) AND the 20-digit
               byte field overwritten, so the record cannot prove sector*512==byte.
               The 12-digit sector number is intact, so the record IS recoverable.
  erase_sector the whole 512-byte sector is filled with 'A': no record, no recovery.
hard_probe.py must return every rot_fields anchor and none of the erase_sector ones.
Without the negative class, a detector that matches anything would also look correct.
"""
import os, subprocess, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BLK = 147456
MAGIC = b'|Block#'
ROT = 100          # records: magic gone + byte field gone, sector number readable
ERASE = 20         # records: nothing left at all

key = bytes([int(open(os.path.join(HERE, 'synth_static.keys.txt')).read().split('\n')[0].split()[1], 16)]) * BLK
ct = bytearray(open(os.path.join(HERE, 'synth_static.dump'), 'rb').read())
pt = bytes(a ^ b for a, b in zip(ct, key * (len(ct) // BLK)))

anchors, start = [], 0
while True:
    i = pt.find(MAGIC, start)
    if i < 0:
        break
    if i % 512 == 0:
        anchors.append(i)
    start = i + 1
assert len(anchors) == 512, 'fixture should hold 512 anchors, got %d' % len(anchors)

rng = np.random.default_rng(11)
order = list(anchors)
rng.shuffle(order)
rot, erase = order[:ROT], order[ROT:ROT + ERASE]

buf = np.frombuffer(pt, dtype=np.uint8).copy()
for off in rot:
    buf[off:off + 7] ^= 0xFF                    # magic distance 56
    buf[off + 39:off + 59] = np.uint8(0x5A)     # 'ZZZZ...': 20-digit byte field destroyed
for off in erase:
    buf[off:off + 512] = np.uint8(0x41)         # no format byte of the record survives

damaged = (buf ^ np.tile(np.frombuffer(key, dtype=np.uint8), len(buf) // BLK)).tobytes()
dumpfn = os.path.join(HERE, 'synth_hard.dump')
open(dumpfn, 'wb').write(damaged)
open(dumpfn + '.xor', 'wb').write(key)


def lba(off):
    return int(pt[off + 7:off + 19])


grid = sorted(lba(o) for o in anchors)
targets = os.path.join(HERE, 'hard_targets.txt')
open(targets, 'w').write('\n'.join(str(v) for v in grid) + '\n')
want_a = len(grid) - ROT - ERASE
print('fixture: %d targets, expect grade A %d, recovered-without-selfcheck %d, unrecoverable %d'
      % (len(grid), want_a, ROT, ERASE))

base = dict(os.environ, BLKSZ=str(BLK), ECCSTART='0', ECCEND='4599', INC='9',
            XORKEY=dumpfn + '.xor', POOL='1')

env = dict(base, TOL='4', MISSOUT=os.path.join(HERE, 'hard_missing.txt'),
           HARDOUT=os.path.join(HERE, 'hard_hard.txt'))
out = subprocess.run([sys.executable, os.path.join(os.path.dirname(HERE), 'anchor_coverage.py'), dumpfn],
                     env=env, capture_output=True, text=True).stdout
audit_hard = int([l for l in out.splitlines() if 'hard-missing (no record' in l][0].split(':')[1].split()[0])
print('anchor_coverage hard-missing : %d expected %d' % (audit_hard, ROT + ERASE))

env = dict(base, MINFMT='9', OUTLIST=os.path.join(HERE, 'hard_recovered.txt'))
out2 = subprocess.run([sys.executable, os.path.join(os.path.dirname(HERE), 'hard_probe.py'),
                       dumpfn, targets], env=env, capture_output=True, text=True).stdout
print(out2)

res = [l for l in out2.splitlines() if l.startswith('grid points recovered')]
got_rec = int(res[0].split(':')[1].split('/')[0])
grade = {}
for line in out2.splitlines():
    if line.startswith('grade '):
        grade[line.split()[1]] = int(line.split(':')[1].strip().split()[0])
ok = True
for label, got, want in (('recovered', got_rec, ROT + want_a),
                         ('grade A', grade.get('A', -1), want_a),
                         ('grade B', grade.get('B', -1) + grade.get('C', -1), ROT),
                         ('unrecoverable', int([l for l in out2.splitlines()
                                               if l.startswith('still unreadable')][0].split(':')[1].split()[0]),
                          ERASE)):
    verdict = 'OK' if got == want else 'FAIL'
    ok = ok and got == want
    print('%-14s %d expected %d  %s' % (label, got, want, verdict))

# The 20 erased sectors must not appear in the recovered list at all.
rec_lba = {int(l.split()[0]) for l in open(os.path.join(HERE, 'hard_recovered.txt'))}
leaked = sorted(rec_lba & {lba(o) for o in erase})
print('erased anchors wrongly recovered: %d %s' % (len(leaked), 'OK' if not leaked else 'FAIL ' + str(leaked)))
sys.exit(0 if ok and not leaked else 1)
