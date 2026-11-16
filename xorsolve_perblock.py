#!/usr/bin/env python3
"""Derive one XOR key byte per NAND block straight from the pattern anchors.

The initpattern record starts with the constant 7 bytes '|Block#'.  If a block was
scrambled with a single key byte k, then at an anchor position p every difference
c[p+i] ^ magic[i] equals k, while at a random position the seven differences agree
only with probability 2^-48.  So one pass over a block votes for k without knowing
it in advance - no 256 descramble passes, no external key needed.

Emits a key as long as the dump, which dumpextractrelevant.pl indexes absolutely.
With --base the existing one-block key is folded in, so the result describes
"static block keystream plus per-block byte" in a single file.
"""
import os, sys
import numpy as np

MAGIC = np.frombuffer(b'|Block#', dtype=np.uint8)


def parse_case(path):
    pagesize = blocksize = None
    with open(path, 'rb') as f:
        txt = f.read().replace(b'\x00', b'').decode('latin-1')
    for line in txt.splitlines():
        s = line.strip()
        if s.startswith('Page '):
            pagesize = int(s.split()[1])
        elif s.startswith('Block '):
            blocksize = int(s.split()[1], 16)
        elif '<Page_size>' in s:
            pagesize = int(s.split('>')[1].split('<')[0])
        elif '<Actual_block_size>' in s:
            blocksize = int(s.split('>')[1].split('<')[0])
    if not pagesize or not blocksize:
        sys.exit('Error: could not read Page/Block size from ' + path)
    return pagesize, blocksize


def block_key(d):
    """Return (key_byte or None, votes) for one already-base-descrambled block."""
    n = len(d) - 6
    if n <= 0:
        return None, 0
    c = d[:n]
    k = c ^ MAGIC[0]
    ok = np.ones(n, dtype=bool)
    for i in range(1, 7):
        ok &= (d[i:i + n] ^ MAGIC[i]) == k
    ks = k[ok]
    if len(ks) == 0:
        return None, 0
    v, cnt = np.unique(ks, return_counts=True)
    return int(v[np.argmax(cnt)]), int(cnt.max())


def main():
    if len(sys.argv) < 4:
        print('Usage: xorsolve_perblock.py <dump> <geometry.case> <output.xor> [--base <one-block.xor>]')
        sys.exit(0)
    dump, casefn, outfn = sys.argv[1], sys.argv[2], sys.argv[3]
    base = None
    if '--base' in sys.argv:
        base = np.fromfile(sys.argv[sys.argv.index('--base') + 1], dtype=np.uint8)
    pagesize, blocksize = parse_case(casefn)
    size = os.path.getsize(dump)
    nblocks = size // blocksize
    print('Geometry: page %d, block %d (%d pages). Dump %d bytes = %d blocks%s'
          % (pagesize, blocksize, blocksize // pagesize, size, nblocks,
             ', base key %d bytes' % len(base) if base is not None else ''), flush=True)

    out = open(outfn, 'wb')
    solved = 0
    try:
        with open(dump, 'rb') as f:
            for b in range(nblocks):
                raw = np.frombuffer(f.read(blocksize), dtype=np.uint8)
                if len(raw) < blocksize:
                    break
                if base is not None:
                    tile = np.resize(base, blocksize)
                    d = raw ^ tile
                else:
                    tile = np.zeros(blocksize, dtype=np.uint8)
                    d = raw
                k, votes = block_key(d)
                if k is None:
                    key = tile                       # nothing to correct
                else:
                    key = tile ^ np.uint8(k)
                    solved += 1
                out.write(key.tobytes())
                if (b + 1) % 100 == 0:
                    print('  block %d/%d, %d keys solved' % (b + 1, nblocks, solved), flush=True)
    finally:
        out.close()

    pad = size - nblocks * blocksize
    if pad:
        with open(dump, 'rb') as f:
            f.seek(nblocks * blocksize)
            out2 = open(outfn, 'ab')
            out2.write(f.read(pad))
            out2.close()
    print('Solved per-block keys for %d/%d blocks (%.2f %%) -> %s (%d bytes)'
          % (solved, nblocks, 100.0 * solved / max(1, nblocks), outfn,
             os.path.getsize(outfn)))
    if solved < nblocks:
        print('Unsolved blocks keep the base key: they hold no readable anchor in this pass.')


main()
