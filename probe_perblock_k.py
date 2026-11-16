#!/usr/bin/env python3
"""One-pass probe: does a NAND block carry its own single XOR key byte?

For a block and a candidate per-block constant k, an anchor is at position p iff
c[p+i] ^ magic[i] is the same value for all i in 0..6.  That test is vectorised, so
each block is scanned once and yields both the anchor positions and the k that each
anchor implies.  No descrambling pass per candidate key.
"""
import sys, os
import numpy as np

MAGIC = np.frombuffer(b'|Block#', dtype=np.uint8)


def probe(data):
    c = np.frombuffer(data[:len(data) - 6], dtype=np.uint8)
    k = c ^ MAGIC[0]
    ok = np.ones(len(c), dtype=bool)
    for i in range(1, 7):
        ok &= (np.frombuffer(data[i:i + len(c)], dtype=np.uint8) ^ MAGIC[i]) == k
    idx = np.flatnonzero(ok)
    return k[idx], idx


def main():
    dump, pagesize, blocksize = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    nblocks = int(sys.argv[4]) if len(sys.argv) > 4 else 40
    size = os.path.getsize(dump)
    total = size // blocksize
    step = max(1, total // nblocks)
    print('%s: %d bytes, %d blocks of %d, sampling every %d' % (dump, size, total, blocksize, step))
    agree = disagree = 0
    with open(dump, 'rb') as f:
        for b in range(0, total, step):
            f.seek(b * blocksize)
            data = f.read(blocksize)
            if len(data) < blocksize:
                break
            ks, idx = probe(data)
            if len(ks) == 0:
                print('  block %-5d no anchor found under any single-byte key' % b)
                disagree += 1
                continue
            vals, cnts = np.unique(ks, return_counts=True)
            top = vals[np.argmax(cnts)]
            zero = int((ks == 0).sum())
            print('  block %-5d anchors %-6d dominant implied key 0x%02X (%d), key 0x00 (%d)'
                  % (b, len(ks), top, cnts.max(), zero))
            agree += 1
            if top != 0:
                disagree += 1
    print('Blocks with anchors: %d, blocks whose dominant key is NOT 0x00: %d' % (agree, disagree))


if __name__ == "__main__":
    main()
