#!/usr/bin/env python3
"""Predict every DATA byte of a chunk from the pattern model, and count errors.

The model, measured against this dump (not assumed):

  page        = 18,432 B = 16 chunks x 1,144 B;  chunk = 1,024 DATA + 6 SA + 114 ECC
  LBAs        = contiguous inside a page: sector (c,s) of a page carries L0 + 2c + s,
                and L0 is a multiple of 9 (verified on the four records of page 0 of
                block 500: 6156288+3, +12, +21, +30 reproduce 6156291/300/309/318)
  sector with (L-3145728) % 9 == 0   -> rec(L) + b'x'*(512-82)
  otherwise                          -> 16-byte unit  "P%011X%04X" % (pattern, pos)
                pattern = (L-3145728)//405, pos = (L-3145728)%9, repeated 32 times,
                then the generator's own bit injection when ((pattern>>3)>>9) == pos-1
                and ((offset%81)//9) is odd.
  majority is 5 (pattern_size = 9*9*5 = 405): block 500 sector 0 reads pattern 0x1D09
  = 7433 and pos 6, and 405*7433 + 195 = 3,010,560 = its offset exactly.  The 7 that
  run/pattern_bics4_4k.dump.xml records does not reproduce that.

Why: a chunk whose 8,192 data bits match the prediction is *known* plaintext, so the
recorded SA+ECC is the encoder's own output for a noiseless data vector.  Those
codewords are what the parity-check-matrix elimination needs; the earlier attempt
used observed data and the bit errors filled the row space (rank 9,152, nullity 0).

Outputs a histogram of data-error counts and, with SAVE=1, the certified codewords
as packed bits for recover_h.py's elimination.
"""
import os, sys
import numpy as np
from multiprocessing import Pool

BLK = int(os.environ.get('BLKSZ', 21233664))
PG = int(os.environ.get('PGSZ', 18432))
CH = int(os.environ.get('CHUNK', 1144))
NC = PG // CH                       # 16 chunks per page
BORDER = int(os.environ.get('BORDER', 3145728))
ECCEND = int(os.environ.get('ECCEND', 21725184))
NCHUNK = int(os.environ.get('NCHUNK', 12000))
POOL = int(os.environ.get('POOL', 10))
SAVE = os.environ.get('SAVE', '0') == '1'
OUT = os.environ.get('OUTBITS', 'certified_bits.npy')
OUTPAR = os.environ.get('OUTPAR', 'certified_full.npy')
DUMP = sys.argv[1] if len(sys.argv) > 1 else 'C:/Users/drclab/Documents/LDPC/01_01_plain.dump'
KEYFILE = os.environ.get('XORKEY', 'C:/Users/drclab/Documents/Qoder/2026-11-15/7246e382/run/01_01_plain.dump.xor')
MAXBLK = int(os.environ.get('MAXBLK', 1))
START = int(os.environ.get('START', 500))
PC = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(1).astype(np.int16)
NREC = 82


def rec(L):
    return (f"|Block#{L:012d} (0x{L:08X}) Byte: {L*512:020d} Pos: {L>>11:10d} MB\n***").encode()


def pred_sector(L):
    """The 512 bytes the pattern generator put at logical sector L."""
    off = L - BORDER
    if off % 9 == 0:
        return rec(L) + b'x' * (512 - NREC)
    pattern, pos = off // 405, off % 9
    unit = b'P%011X%04X' % (pattern, pos)
    buf = bytearray(unit * 32)
    if ((pattern >> 3) >> 9) == pos - 1 and ((off % 81) // 9) & 1:
        buf[(pattern >> 3) & 0x1FF] ^= 1 << (pattern & 7)
    return bytes(buf)


def page_base(data, pos):
    """LBA of the first sector of the page containing byte position pos, or None.

    A page's 32 sectors carry contiguous LBAs, so one intact record anywhere in it
    fixes the base.  Requiring an exact skeleton (0 wrong bits) keeps the base honest.
    """
    import repair_anchors as R
    start = pos - (pos % PG)
    for c in range(NC):
        for s in (0, 512):
            q = start + c * CH + s
            if q + NREC >= len(data):
                continue
            obs = np.frombuffer(data[q:q + NREC], np.uint8)
            if int(PC[obs[R.SKEL] ^ R.SKELREF].sum()):
                continue
            r = R.classify(data, q)
            if not r:
                continue
            base = r[0] - (2 * c + s // 512)
            if base >= BORDER:
                return base
    return None


def build_pred(pcache, L):
    if L not in pcache:
        pcache[L] = np.frombuffer(pred_sector(L), np.uint8)
    return pcache[L]


def work(args):
    b, path, keyfn = args
    key = np.fromfile(keyfn, dtype=np.uint8)
    with open(path, 'rb') as f:
        f.seek(b * BLK)
        raw = f.read(BLK)
    data = np.asarray(np.frombuffer(raw, dtype=np.uint8)) ^ key
    import repair_anchors as R
    nerr = []
    nopage = 0
    sazero = 0
    satot = 0
    pcache = {}
    pages = len(data) // PG
    for p in range(pages):
        start = p * PG
        base = page_base(data, start)
        if base is None:
            nopage += 1
            continue
        for c in range(NC):
            q = start + c * CH
            if base + 2 * c + 1 > ECCEND:
                break
            want = np.concatenate([build_pred(pcache, base + 2 * c),
                                   build_pred(pcache, base + 2 * c + 1)])
            got = data[q:q + 1024]
            if len(got) < 1024:
                continue
            nerr.append(int(PC[want ^ got].sum()))
            satot += 1
            sazero += int((data[q + 1024:q + 1030] == 0).all())
    return b, np.array(nerr, np.int32), nopage, sazero, satot


def main():
    nblocks = os.path.getsize(DUMP) // BLK
    blocks = list(range(START, min(nblocks, START + MAXBLK)))
    print('predicting DATA for %d blocks from %d, model: 9 sectors/record, 405 sectors/pattern'
          % (len(blocks), blocks[0]), flush=True)
    hist = {}
    tot = 0
    clean = 0
    nopage = 0
    sa = sat = 0
    cert = []
    with Pool(POOL) as pool:
        for b, ne, npage, saz, n in pool.imap_unordered(work, ((x, DUMP, KEYFILE) for x in blocks)):
            nopage += npage
            sa += saz
            sat += n
            for x in ne.tolist():
                hist[x] = hist.get(x, 0) + 1
                tot += 1
                if x == 0:
                    clean += 1
                    if SAVE and len(cert) < NCHUNK:
                        cert.append((b, x))
            print('  block %d done, %d chunks so far' % (b, tot), flush=True)
    print('\nchunks compared        : %d' % tot)
    print('pages with no anchor    : %d' % nopage)
    print('SA all-zero             : %d / %d (%.3f%%)' % (sa, sat, 100.0 * sa / max(1, sat)))
    print('DATA fully error-free   : %d (%.3f%%)' % (clean, 100.0 * clean / max(1, tot)))
    top = sorted(hist.items(), key=lambda kv: kv[0])[:14]
    print('data bit-error histogram: %s' % ' '.join('%d:%d' % t for t in top))
    tail = sum(v for k, v in hist.items() if k > 100)
    print('chunks with >100 wrong bits: %d  (model or anchor wrong there)' % tail)
    mean = sum(k * v for k, v in hist.items()) / max(1, tot)
    print('mean wrong bits per 8,192-bit DATA field: %.3f  -> per-bit error rate %.2e'
          % (mean, mean / 8192.0))
    return 0


if __name__ == '__main__':
    sys.exit(main())
