"""Is the 1.58e-3 raw BER channel noise, or a hole in the pattern model?

certify_data.py proves the model is right *in structure* (no chunk is off by more than 51 of 8,192
bits, and a wrong model blows up whole sectors), but a structural gap can still hide in the small
counts. The only bytes the generator does anything unusual to are:

  - the 82-byte `|Block#...` record header of a record-carrying sector ((L-BORDER) % 9 == 0);
  - exactly ONE injected byte per pattern sector, at index (pattern >> 3) & 0x1FF inside the 512 B
    sector, and only when ((pattern>>3)>>9) == pos-1 and ((off%81)//9) is odd.

If the error bits concentrate in those two classes - which between them cover a few hundred of the
8,192 bits - the model is wrong somewhere and the BER above is inflated. If they spread over the
plain `P%011X%04X` text in proportion to how much space it occupies, the residual is the channel.

Usage: python err_cause.py [nblocks]
Env: START (500), POOL (8)
"""
import os
import sys

import numpy as np

sys.argv = sys.argv[:1]
import certify_data as C                       # noqa: E402  reuse the validated model verbatim

NBLK = int(sys.argv[1]) if len(sys.argv) > 1 else int(os.environ.get('NBLK', '2'))
KEYFILE = os.environ.get('XORKEY', 'C:/Users/drclab/Documents/Qoder/2026-11-15/7246e382/run/'
                                   '01_01_plain.dump.xor')
BORDER, PC8 = C.BORDER, C.PC


def class_map(base_chunk):
    """Per 1,024 B chunk: 0 = plain pattern text, 1 = record header, 2 = the injected byte."""
    cls = np.zeros(1024, np.uint8)
    for s in (0, 1):
        L = base_chunk + s
        off = L - BORDER
        q = s * 512
        if off % 9 == 0:
            cls[q:q + C.NREC] = 1
            continue
        pattern, pos = off // 405, off % 9
        if ((pattern >> 3) >> 9) == pos - 1 and ((off % 81) // 9) & 1:
            cls[q + ((pattern >> 3) & 0x1FF)] = 2
    return cls


def work(args):
    b, path, keyfn = args
    key = np.fromfile(keyfn, dtype=np.uint8)
    with open(path, 'rb') as f:
        f.seek(b * C.BLK)
        data = np.frombuffer(f.read(C.BLK), np.uint8) ^ key
    err = np.zeros(3, np.int64)
    space = np.zeros(3, np.int64)
    pcache = {}
    cmap = {}
    pages = len(data) // C.PG
    for p in range(pages):
        start = p * C.PG
        base = C.page_base(data, start)
        if base is None:
            continue
        for c in range(C.NC):
            q = start + c * C.CH
            if base + 2 * c + 1 > C.ECCEND:
                break
            want = np.concatenate([C.build_pred(pcache, base + 2 * c),
                                   C.build_pred(pcache, base + 2 * c + 1)])
            got = data[q:q + 1024]
            if len(got) < 1024:
                continue
            if base + 2 * c not in cmap:
                key_id = base + 2 * c
                if len(cmap) > 200000:
                    cmap.clear()
                cmap[key_id] = class_map(key_id)
            m = cmap[base + 2 * c]
            bad = PC8[want ^ got]                      # wrong bits per byte position
            for k in (0, 1, 2):
                sel = m == k
                err[k] += int(bad[sel].sum())
                space[k] += int(sel.sum()) * 8
    return b, err, space


def main():
    start = int(os.environ.get('START', '500'))
    pool = int(os.environ.get('POOL', '8'))
    blocks = list(range(start, start + NBLK))
    tot_e = np.zeros(3, np.int64)
    tot_s = np.zeros(3, np.int64)
    with_pool = __import__('multiprocessing').Pool(pool)
    try:
        for b, e, s in with_pool.imap_unordered(work, [(x, C.DUMP, KEYFILE) for x in blocks]):
            tot_e += e
            tot_s += s
            print(f'  block {b}: errors by class {e.tolist()}  space {s.tolist()}', flush=True)
    finally:
        with_pool.terminate()
        with_pool.join()
    names = ['plain pattern text', 'record header (82 B)', 'the injected byte']
    print(f'\nblocks {blocks}   wrong bits attributed to each structural class')
    print(f'{"class":<22} {"errors":>9} {"share of errors":>16} {"bits of space":>15} '
          f'{"share of space":>15} {"BER in class":>14}')
    te, ts = tot_e.sum(), tot_s.sum()
    for k in (0, 1, 2):
        print(f'{names[k]:<22} {int(tot_e[k]):>9,} {100.0 * tot_e[k] / te:>15.2f} % '
              f'{int(tot_s[k]):>15,} {100.0 * tot_s[k] / ts:>14.2f} % '
              f'{tot_e[k] / max(1, tot_s[k]):>14.2e}')
    print(f'{"TOTAL":<22} {int(te):>9,} {"":>16} {int(ts):>15,}')
    lift = (tot_e[2] / max(1, tot_s[2])) / max(tot_e[0] / max(1, tot_s[0]), 1e-18)
    print(f'\nerror-rate lift on the injected byte vs plain pattern text: {lift:.2f}x')
    print('verdict:')
    if lift > 3.0 and tot_e[2] / te > 0.05:
        print('  the injection rule is not the whole story - model gap confirmed on that byte.')
    elif tot_e[0] / te > 0.85:
        print('  errors sit where the space sits (plain pattern text carries 8,128 of 8,192 bits), '
              'so the residual is channel noise and the 1.6e-3 BER is real.')
    else:
        print(f'  mixed: {100.0 * tot_e[0] / te:.1f} % of errors in plain text, '
              f'{100.0 * tot_e[1] / te:.1f} % in record headers, '
              f'{100.0 * tot_e[2] / te:.1f} % on injected bytes.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
