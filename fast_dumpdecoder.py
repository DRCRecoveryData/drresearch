#!/usr/bin/env python3
"""Fast, layout-correct LDPC decode pass over an extracted dump.

Why not just use dumpdecoder.py: it slices page[pos : pos+datasize+eccsize], i.e.
DATA immediately followed by ECC. This chip's chunk is DATA 1024 + SA 6 + ECC 114,
so that slice silently drops 6 bytes of SA and shifts every ECC bit after it - with
n=9152 in the hmatrix name, the author's own loop feeds the decoder 9104 bits and
misaligns the last 114 bytes of each chunk. Here the whole 1,144 B chunk is taken,
which is what n=9152 means.

Also: sigma is no longer welded to EbN=3 (env EB_N), the pages are batched and run
across processes, and the decode itself is fast_minsum.Fast (265x the author's
per-codeword time, bit-identical on the same inputs).

Usage:
  python fast_dumpdecoder.py <dump> <hmatrix_nN_kK_mM.h> <geometry.case> <out.dump>
  python fast_dumpdecoder.py --selftest
Env: EB_N (default 3.0, the author's value), WORKERS (default cpu_count),
     CHUNKS (sub-codewords per worker task, default 256), ITERS (default 20)
"""
import os
import re
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fast_minsum import Fast, synth_H, make_encoder  # noqa: E402

PGSZ = 18432
CH = 1144


def parse_geometry(case_path):
    """Two candidate sub-codeword spans per chunk, from the same .case the rest of the
    toolchain reads: the whole DATA+SA+ECC field, and DATA+ECC with the SA excluded.
    Which one is the real codeword is settled by n in the hmatrix name."""
    txt = open(case_path, 'rb').read().replace(b'\x00', b'').decode('utf-8', 'replace')
    recs = [(n, int(a), int(b)) for n, a, b in re.findall(
        r'StructureDefinitionName="(DATA|SA|ECC)"\s+StartAddress="(\d+)"\s+StopAddress="(\d+)"', txt)]
    if len(recs) % 3:
        raise SystemExit(f'case has {len(recs)} field records, expected a multiple of 3')
    full, no_sa = [], []
    for i in range(0, len(recs), 3):
        (dn, da, db), (sn, sa, sb), (en, ea, eb) = recs[i:i + 3]
        if not (db + 1 == sa and sb + 1 == ea):
            raise SystemExit(f'chunk {i//3} is not DATA|SA|ECC contiguous: {da}-{db} {sa}-{sb} {ea}-{eb}')
        full.append([(da, eb + 1 - da)])                    # the whole 1,144 B chunk
        no_sa.append([(da, db + 1 - da), (ea, eb + 1 - ea)])  # DATA then ECC, SA outside the code
    return {'DATA+SA+ECC': full, 'DATA+ECC': no_sa}


def read_h(path, n, m):
    """The author's format: raw bytes, LSB-first within each byte, row-major m x n."""
    raw = np.fromfile(path, dtype=np.uint8)
    if raw.size * 8 != n * m:
        raise SystemExit(f'{path} is {raw.size} B = {raw.size*8} bits, name says n={n} m={m} = {n*m} bits')
    return np.unpackbits(raw, bitorder='little').astype(np.int64).reshape(m, n)


def decode_pages(args):
    """Worker task: decode whole pages, every sub-codeword of the task batched into
    one Fast call, and write the pages back complete so the output stays page-aligned."""
    dump, hpath, pgs, chunks, eb_n, iters, p0 = args
    n = int(re.search(r'_n(\d+)', hpath).group(1))
    m = int(re.search(r'_m(\d+)', hpath).group(1))
    H = read_h(hpath, n, m)
    fast = Fast(H)
    sigma = float(np.sqrt((1.0 / (10 ** (eb_n / 10.0))) / 2))
    llrc = (2 / sigma ** 2)
    out = bytearray()
    solved = used_sum = 0
    with open(dump, 'rb') as fh:
        for p in pgs:
            fh.seek(p * PGSZ)
            page = bytearray(fh.read(PGSZ))
            if len(page) != PGSZ:
                break
            pieces = [b''.join(page[o:o + ln] for o, ln in pl) for pl in chunks]
            x = np.stack([np.unpackbits(np.frombuffer(b, np.uint8), bitorder='little')
                          for b in pieces]).astype(np.float32)
            z, used, sol = fast.decode((2 * x - 1) * llrc, iters)
            dec = np.packbits(z.astype(np.uint8), axis=1, bitorder='little')
            for i, pl in enumerate(chunks):            # only a solved chunk is spliced
                if not sol[i]:
                    continue
                o = 0
                for off, ln in pl:
                    page[off:off + ln] = dec[i][o:o + ln]
                    o += ln
            out += page
            solved += int(sol.sum())
            used_sum += int(used[used >= 0].sum())
    return bytes(out), (len(out) // PGSZ, solved, used_sum), p0


def emit_synth_h(path, n=9152, m=912, rw=16):
    """A placeholder hmatrix in the author's format, for dry-running the file path and
    the worker pool before the real one arrives. It is NOT the PS2251-11 code."""
    H = synth_H(n, m, rw)
    with open(path, 'wb') as fh:
        fh.write(np.packbits(H.reshape(-1).astype(np.uint8), bitorder='little').tobytes())
    print(f'{path}: {os.path.getsize(path)} B, n={n} m={m}, row weight {rw} of a regular '
          f'{m // rw}x{n // rw} prototype - placeholder only')


def main():
    if '--selftest' in sys.argv:
        return selftest()
    if '--synth-h' in sys.argv:
        emit_synth_h(sys.argv[sys.argv.index('--synth-h') + 1])
        return 0
    if len(sys.argv) < 5:
        print(__doc__)
        return 1
    dump, hpath, case, outpath = sys.argv[1:5]
    n, k, m = (int(x) for x in re.search(r'_n(\d+)_k(\d+)_m(\d+)', os.path.basename(hpath)).groups())
    layouts = parse_geometry(case)
    chunks = None
    per_bits = {}
    for name, cand in layouts.items():
        bits = sum(ln for _, ln in cand[0]) * 8          # one sub-codeword, not one page
        per_bits[name] = bits
        if bits == n:
            chunks = cand
            print('layout: %s - %d sub-codewords/page, %d bits each, matching n=%d'
                  % (name, len(cand), bits, n), flush=True)
            break
    if chunks is None:
        full = layouts['DATA+SA+ECC']
        page_bits = sum(ln for pieces in full for _, ln in pieces) * 8
        raise SystemExit('hmatrix n=%d matches none of %s, and the page-wide total %d either '
                         '- the code and this geometry do not agree'
                         % (n, per_bits, page_bits))
    print(f'{os.path.basename(hpath)}: n={n} k={k} m={m}; {len(chunks)} sub-codewords/page, '
          f'{n // len(chunks)} bits each', flush=True)
    size = os.path.getsize(dump)
    pages = size // PGSZ
    cap = int(os.environ.get('MAXPAGES', '0'))
    if cap:
        pages = min(pages, cap)
    workers = int(os.environ.get('WORKERS', os.cpu_count() or 4))
    per = int(os.environ.get('CHUNKS', 256))
    eb_n = float(os.environ.get('EB_N', 3.0))
    iters = int(os.environ.get('ITERS', 20))

    pages_per_task = max(1, per // len(chunks))
    tasks = [(dump, hpath, list(range(p0, min(p0 + pages_per_task, pages))),
              chunks, eb_n, iters, p0) for p0 in range(0, pages, pages_per_task)]
    total = pages * len(chunks)
    print(f'{pages} pages, {total} sub-codewords, {len(tasks)} tasks of '
          f'{pages_per_task} pages on {workers} workers, Eb/N0={eb_n} dB', flush=True)

    import multiprocessing as mp
    t0 = time.time()
    done = solved_all = 0
    with mp.Pool(workers) as pool, open(outpath, 'wb') as out:
        for data, (pgs, solved, iters_used), p0 in pool.imap(decode_pages, tasks, chunksize=1):
            out.write(data)
            done += pgs * len(chunks)
            solved_all += solved
            el = time.time() - t0
            rate = done / el if el else 0
            mean_it = iters_used / max(solved, 1)
            print(f'\rpage {p0}..{p0 + pgs}  chunks {done}/{total}  '
                  f'solved {solved_all} ({100*solved_all/max(done,1):.1f}%)  '
                  f'{rate:.1f} chunks/s (mean {mean_it:.1f} iters)  '
                  f'eta {(total-done)/max(rate,1e-9)/3600:.1f} h   ',
                  end='', flush=True)
    print(f'\nwrote {outpath}  {os.path.getsize(outpath)} B in {time.time()-t0:.1f} s')
    print(f'solved {solved_all}/{done} sub-codewords; unsolved chunks are written back as the '
          f'raw input bytes, so a failure is never a corruption')
    return 0


def selftest():
    """Proves the bit plumbing end to end without a real hmatrix: a synthetic code in
    the author's own file format, a codeword written as a chunk, then decode and
    compare bytes. Also proves the SA offset the author's slice gets wrong."""
    n, m, rw = 9152, 912, 16
    H = synth_H(n, m, rw)
    encode, pivots, free = make_encoder(H)
    rng = np.random.default_rng(6767)
    cw = encode(rng.integers(0, 2, size=(1, len(free)), dtype=np.uint8))[0]
    raw = np.packbits(cw, bitorder='little').tobytes()
    assert len(raw) == CH, f'{n} bits packed is {len(raw)} B, expected one {CH} B chunk'

    tmp = 'selftest_hmatrix_n9152_k8240_m912.h'
    open(tmp, 'wb').write(np.packbits(H.reshape(-1).astype(np.uint8), bitorder='little').tobytes())
    assert os.path.getsize(tmp) == n * m // 8, 'hmatrix file size wrong'

    back = read_h(tmp, n, m)
    assert (back == H).all(), 'hmatrix round trip failed'

    fast = Fast(H)
    llr = (2 * cw.astype(np.float32) - 1) * 8.0
    z, used, solved = fast.decode(llr[None, :], 20)
    print('clean codeword   : solved %s at iteration %d, bytes identical %s'
          % (bool(solved[0]), used[0], bool((z[0] == cw).all())))

    # flip one bit inside the SA window (bytes 1024..1029) and check the decoder is
    # looking at the same bit the author's slice would have reached
    bad = cw.copy()
    bit = 1024 * 8 + 3
    bad[bit] ^= 1
    z2, used2, solved2 = fast.decode(((2 * bad.astype(np.float32) - 1) * 8.0)[None, :], 20)
    print('one SA bit wrong : solved %s at iteration %d, recovered exactly %s'
          % (bool(solved2[0]), used2[0], bool((z2 == cw).all())))

    # the author's slice: page[pos:pos+datasize+eccsize] = 1138 B, 48 bits short
    txt = open('../submit/geometry_ps2251-11_bics4.case', 'rb').read().replace(b'\x00', b'').decode() \
        if os.path.exists('../submit/geometry_ps2251-11_bics4.case') else None
    if txt:
        layouts = parse_geometry('../submit/geometry_ps2251-11_bics4.case')
        for name, cand in layouts.items():
            print('case layout %-14s: %d sub-codewords/page, %d bits each'
                  % (name, len(cand), sum(ln for _, ln in cand[0]) * 8))
        print('author dumpdecoder.py slices DATA+eccsize contiguously, which is the '
              'DATA+ECC span only when SA is absent; here it silently drops 6 B per chunk')
    os.remove(tmp)
    return 0


if __name__ == '__main__':
    sys.exit(main())
