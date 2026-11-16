#!/usr/bin/env python3
"""Positive control for anchor_coverage.py's fuzzy path.

Takes the static-key fixture and flips a known number of bits inside the 7-byte
'|Block#' magic of a known set of anchors.  Bit flips are applied to the
ciphertext, which is equivalent because the XOR key is bitwise.  The audit must
then report the anchors missing at tolerance 0 and back at their own tolerance,
with nothing else moving.  Without this, a '0 recovered' verdict on the real dump
is indistinguishable from a dead detector.
"""
import os, subprocess, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import anchor_coverage as ac   # noqa: E402  (same grid/magic definitions)

BLK = 147456
MAGIC = b'|Block#'
PLAN = {1: 100, 2: 100, 3: 100}     # bits flipped -> how many anchors
DESTROY = 100                        # anchors whose magic is inverted beyond any tolerance

key = bytes([int(open(os.path.join(HERE, 'synth_static.keys.txt')).read().split('\n')[0].split()[1], 16)]) * BLK
ct = bytearray(open(os.path.join(HERE, 'synth_static.dump'), 'rb').read())
pt = bytes(a ^ b for a, b in zip(ct, key * (len(ct) // BLK)))

rng = np.random.default_rng(7)
anchors = []
start = 0
while True:
    i = pt.find(MAGIC, start)
    if i < 0:
        break
    if i % 512 == 0:                       # real record starts, not filler text
        anchors.append(i)
    start = i + 1
print('anchors located: %d' % len(anchors))
assert len(anchors) == 512, 'fixture should hold 512 anchors'

rng.shuffle(anchors)
flips = np.frombuffer(pt, dtype=np.uint8).copy()
cursor = 0
expected = {}
for nbits, count in sorted(PLAN.items()):
    for off in anchors[cursor:cursor + count]:
        bytes_used = set()
        while len(bytes_used) < nbits:
            bytes_used.add(int(rng.integers(0, 7)))
        for j in bytes_used:
            flips[off + j] ^= np.uint8(1 << int(rng.integers(0, 8)))
    expected[nbits] = cursor + count
    cursor += count
destroyed = []
for off in anchors[cursor:cursor + DESTROY]:
    flips[off:off + 7] ^= 0xFF                       # magic obliterated, digits untouched
    d = sum(bin(a ^ 0xFF).count('1') for a in MAGIC)
    destroyed.append((off, d))
print('destroyed %d magics (bit distance %d from |Block#)' % (len(destroyed), destroyed[0][1]))

damaged = bytes(a ^ b for a, b in zip(flips.tobytes(), key * (len(ct) // BLK)))
dumpfn = os.path.join(HERE, 'synth_corrupt.dump')
open(dumpfn, 'wb').write(damaged)
open(dumpfn + '.xor', 'wb').write(key)
print('wrote %s (%d bytes)' % (dumpfn, len(damaged)))

env = dict(os.environ, BLKSZ=str(BLK), ECCSTART='0', ECCEND='4599', INC='9',
           XORKEY=dumpfn + '.xor', TOL='4', POOL='1',
           MISSOUT=os.path.join(HERE, 'corrupt_missing.txt'))
out = subprocess.run([sys.executable, os.path.join(os.path.dirname(HERE), 'anchor_coverage.py'), dumpfn],
                     env=env, capture_output=True, text=True).stdout
print(out)

cum = 0
ok = True
for nbits in sorted(PLAN):
    cum += PLAN[nbits]
    line = [l for l in out.splitlines() if 'allowed %d bad' % nbits in l]
    got = int(line[0].split('+')[1].split('recovered')[0]) if line else -1
    verdict = 'OK' if got == cum else 'FAIL'
    ok = ok and got == cum
    print('tolerance %d: recovered %-4d expected %-4d (cumulative flips)  %s' % (nbits, got, cum, verdict))


def field(label, key):
    line = [l for l in out.splitlines() if key in l]
    return int(line[0].split(label)[-1].split()[0]) if line else -1


exact = int([l for l in out.splitlines() if 'found (exact magic)' in l][0].split(':')[1].strip().split()[0])
want_exact = 512 - cum - DESTROY
print('exact-magic found: %d expected %d  %s' % (exact, want_exact, 'OK' if exact == want_exact else 'FAIL'))
ok = ok and exact == want_exact
sline = [l for l in out.splitlines() if 'structural scan adds' in l]
got_s = int(sline[0].split(':')[1].strip().split()[0]) if sline else -1
print('magic-free recovery: %d expected %d  %s' % (got_s, DESTROY, 'OK' if got_s == DESTROY else 'FAIL'))
ok = ok and got_s == DESTROY
hard = [l for l in out.splitlines() if 'hard-missing (no record' in l]
got_h = int(hard[0].split(':')[1].strip().split()[0]) if hard else -1
print('hard-missing:      %d expected 0  %s' % (got_h, 'OK' if got_h == 0 else 'FAIL'))
ok = ok and got_h == 0
sys.exit(0 if ok else 1)
