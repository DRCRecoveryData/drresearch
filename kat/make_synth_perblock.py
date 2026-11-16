#!/usr/bin/env python3
"""Synthetic known-answer test for the per-block XOR limitation.

Builds a small pattern-shaped image exactly like stage 2 of initpattern.pl
(|Block#... records on the (DA/512+1) grid, P...... text elsewhere), scrambles
each NAND block with its own random key byte the way the author's own
'LFSR Based XOR key Generator' C program does, and writes the geometry and
pattern files dumpextractrelevant.pl needs.  Ground truth is kept next to it.
"""
import os, random, struct, sys

PAGESIZE = 18432
BLOCKSIZE = 147456          # 8 pages
NBlocks = 16
DA = 4096
INC = DA // 512 + 1         # 9
SECTORS_PER_BLOCK = BLOCKSIZE // 512   # 288
SECTORS_PER_PAGE = PAGESIZE // 512     # 36
HERE = os.path.dirname(os.path.abspath(__file__))


def record(lbab):
    b = lbab * 512
    block = 0x20000000 + lbab // SECTORS_PER_BLOCK
    pos = (b // 1000000) * 100
    text = ('|Block#%012d (0x%08X) Byte: %020d Pos: %10d MB\n***OVERWRITTEN'
            % (lbab, block, b, pos))
    return (text.encode() + b'x' * 512)[:512]


def precord():
    return (b'P0000000000000000' * 32)[:512]


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else 'perblock'   # static = classic one-key dump
    suf = '_static' if mode == 'static' else ''
    random.seed(20251115)
    keys = ([random.randrange(1, 256)] * NBlocks if mode == 'static'
            else [random.randrange(1, 256) for _ in range(NBlocks)])
    anchors = 0
    with open(os.path.join(HERE, 'synth%s.dump' % suf), 'wb') as out:
        for bi in range(NBlocks):
            block = bytearray()
            for p in range(BLOCKSIZE // PAGESIZE):
                page = bytearray()
                for j in range(SECTORS_PER_PAGE):
                    lbab = bi * SECTORS_PER_BLOCK + p * SECTORS_PER_PAGE + j
                    if lbab % INC == 0:
                        page += record(lbab)
                        anchors += 1
                    else:
                        page += precord()
                block += page
            assert len(block) == BLOCKSIZE
            k = keys[bi]
            out.write(bytes(b ^ k for b in block))
    with open(os.path.join(HERE, 'synth%s.keys.txt' % suf), 'w') as o:
        o.write('\n'.join('%d 0x%02X' % (i, k) for i, k in enumerate(keys)) + '\n')

    chip = ('Chip.txt\r\nFlash\r\nPage         %d\r\nBlock        0x%X\r\nLun          0\r\n'
            % (PAGESIZE, BLOCKSIZE)).encode('utf-16-le')
    with open(os.path.join(HERE, 'synth%s_Chip.txt' % suf), 'wb') as o:
        o.write(chip)
    # the grid must stay inside the image: the last written sector is the final
    # block's last one, so the last scan point is one step short of the end.
    eccend = NBlocks * SECTORS_PER_BLOCK - INC
    with open(os.path.join(HERE, 'synth%s.pattern.xml' % suf), 'w') as o:
        o.write("<patternlist>\n  <pattern type='ECC' begin='0' end='%d' size='%d' coverage='%d' majority='7'/>\n</patternlist>\n"
                % (eccend, NBlocks * BLOCKSIZE, DA))
    print('synth%s.dump: %d bytes, %d blocks, %d anchors on the %d-sector grid'
          % (suf, NBlocks * BLOCKSIZE, NBlocks, anchors, eccend))
    print('true keys:', ' '.join('%02X' % k for k in keys))


main()
