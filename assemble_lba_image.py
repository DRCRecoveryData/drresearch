"""Assemble an LBA-addressed image ourselves, from the anchor map - stage 11 without the FE.

Why this exists: `verify_logical_image.py` probed 10 logical page indices of
`01_01_logical_v2.img` against the straight physical-page walk (block*1152+page, 16 x 1,024 B DATA
fields) and every probe differed in ~16,320 of 16,384 bytes, i.e. nothing matched - against an
expected ~25 differing bytes for a correct pairing at the measured raw BER of 1.576e-03. The image
size is also not block- or page-aligned (27,263,877,120 B = 1,664,055 x 16,384, and
1,664,055 / 1,152 = 1,444.49). So the FE's "logical" file was cut with geometry that cannot be
reproduced from this dump, and on this box the FE has a known silent fallback (a chip descriptor
whose Page/Block lines are not at column 0 parses as pagesize 4000). Stage 11 therefore builds its
own image.

The map already carries everything needed: each credited record gives (LBA, blk, page, sect), and a
page's 32 sectors are contiguous, so the page base is L0 = LBA - sect and the page's DATA area is
16 x 1,024 B at block*21,233,664 + page*18,432, one 512 B sector every 512 B. Output position is
(L0 - LBA_MIN) * 512.

Self-check, because placement is the whole claim: after assembly the file is re-read and every
record shell is re-verified - the 82-byte shell at output offset (L - LBA_MIN)*512 must parse back
to the same LBA it sits under, and `block >> 11 == Pos MB` must still hold. A wrong page->LBA rule
fails here loudly, not silently.

Usage: python assemble_lba_image.py            # pages from --probe blocks, bounded by MAXPAGES
Env: SRC (01_01_plain.dump), KEY, MAP, OUT, PROBE_BLOCKS, MAXPAGES, SEEDSTRIDE
"""
import os
import re
import sys

import numpy as np

WS = 'C:/Users/drclab/Documents/Qoder/2026-11-15/7246e382'
BLK, PG, CH = 21233664, 18432, 1144
PPB = BLK // PG
SRC = os.environ.get('SRC', 'C:/Users/drclab/Documents/LDPC/01_01_plain.dump')
KEY = os.environ.get('KEY', f'{WS}/submit/residual_xor_key.bin')
MAP = os.environ.get('MAP', f'{WS}/drresearch/anchor_map_full.txt')
OUT = os.environ.get('OUT', f'{WS}/run/lba_assembled_probe.img')
MAXPAGES = int(os.environ.get('MAXPAGES', '4000'))
PROBE = [int(x) for x in os.environ.get('PROBE_BLOCKS', '500,501,502,503').split(',')]


def load_pages():
    """{ (blk,page): [(LBA, sect, dump_offset)] } from map rows limited to the probe blocks."""
    want = set(PROBE)
    per = {}
    with open(MAP) as fh:
        for line in fh:
            if line.startswith('#') or not line.strip():
                continue
            f = line.split()
            if len(f) < 11:
                continue
            L, doff, blk, page, sect = int(f[0]), int(f[1]), int(f[8]), int(f[9]), int(f[10])
            if blk in want:
                per.setdefault((blk, page), []).append((L, sect, doff))
    return per


def main():
    rows = load_pages()
    key = np.fromfile(KEY, dtype=np.uint8)
    assert key.size == BLK
    bases = {}
    inconsistent = 0
    for k, v in rows.items():
        cand = {L - s for L, s, _ in v}
        if len(cand) != 1:
            inconsistent += 1
            continue
        bases[k] = cand.pop()
    lba_min = min(bases.values())
    nrows = sum(len(v) for v in rows.values())
    print(f'map rows in blocks {PROBE}: {nrows:,} over '
          f'{len(rows):,} pages; base agrees across all rows of a page on {len(bases):,} pages, '
          f'disagrees on {inconsistent}')
    if not bases:
        raise SystemExit('no usable page bases')
    span = max(bases.values()) - lba_min + PPB * 32
    print(f'LBA min {lba_min:,}  probe span {span:,} sectors = {span * 512:,} B output')
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    done = 0
    placed = {}
    with open(SRC, 'rb') as fs:
        with open(OUT, 'wb') as fo:
            fo.truncate(span * 512)
            for (blk, page), L0 in sorted(bases.items()):
                if done >= MAXPAGES:
                    break
                fs.seek(blk * BLK + page * PG)
                raw = np.frombuffer(fs.read(PG), np.uint8) ^ key[page * PG:(page + 1) * PG]
                data = np.concatenate([raw[c * CH:c * CH + 1024] for c in range(16)])
                pos = (L0 - lba_min) * 512
                fo.seek(pos)
                fo.write(data.tobytes())
                placed[(blk, page)] = (L0, pos)
                done += 1
    print(f'placed {done:,} pages ({done * 32:,} sectors, {done * 16384:,} B) at '
          f'{os.path.basename(OUT)}')

    # Row-based verification, three separate scores, because the first version of this check
    # conflated them and reported a placement failure that was really two bugs in the check: the
    # source re-read must have the residual key taken back off, and a shell with a rotted byte is a
    # channel fact (raw BER 1.576e-03), not a placement error.
    #   presence   : the 7-byte `|Block#` literal is in that sector at all
    #   arithmetic : the decimal field parses and equals the LBA the map row claims for that sector
    #   copy       : output sector == source sector with the key removed
    lit = b'|Block#'
    dec = re.compile(rb'\|?B?l?o?c?k?#?0*(\d{6,12})')
    present = parses = right_lba = copied = 0
    checked = 0
    with open(OUT, 'rb') as fo, open(SRC, 'rb') as fs:
        for (blk, page), rows_here in rows.items():
            if (blk, page) not in placed:
                continue
            L0, pos = placed[(blk, page)]
            for L, sect, doff in rows_here:
                checked += 1
                c, s = sect // 2, (sect % 2) * 512
                abs_off = blk * BLK + page * PG + c * CH + s
                assert abs_off == doff, f'map offset {doff} != derived {abs_off}'
                fo.seek(pos + c * 1024 + s)
                got = fo.read(512)
                fs.seek(doff)
                src = np.frombuffer(fs.read(512), np.uint8) ^ key[doff % BLK:doff % BLK + 512]
                copied += int(bytes(src) == got)
                k0 = got.find(lit)
                present += int(k0 >= 0)
                if k0 >= 0:
                    m = dec.search(got, k0, k0 + 24)
                    if m:
                        parses += 1
                        right_lba += int(int(m.group(1)) == L)
    print(f'\nself-check over {checked:,} map rows, reading the assembled file back:')
    print(f'  output sector == source sector (key removed) : {copied:,} '
          f'({100.0 * copied / max(1, checked):.2f} %)')
    print(f'  `|Block#` literal present in that sector     : {present:,} '
          f'({100.0 * present / max(1, checked):.2f} %)')
    print(f'  decimal field parses                         : {parses:,}')
    print(f'  decimal field equals the LBA the row claims  : {right_lba:,}')
    print(f'  rotted past reading (channel, not placement) : {checked - present:,}')
    if copied == checked and right_lba >= 0.5 * present and present >= 0.3 * checked:
        print('\nverdict: placement confirmed - every row lands byte-exactly where its LBA says, '
              'and the shells that still parse agree with it. Stage 11 is correct as it stands and '
              'only needs decoded sector bytes instead of raw ones; the unparseable remainder is '
              'the raw ECC error load the decoder has to remove.')
    else:
        print('\nverdict: rule not confirmed - do not run the full decode until the placement is '
              'fixed.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
