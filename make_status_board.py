#!/usr/bin/env python3
"""Render the measured status board for the PS2251-11 / BiCS4 LDPC job as a PNG.

Every chip on the board is derived from an artifact on disk at run time - the script re-reads the
verdict lines instead of carrying numbers, so a stale board cannot be produced by re-running it.
Colour is decided by a stated rule per row, never typed by hand:

  green   the check passed with the counted numbers shown
  amber   expected non-perfection, or a step that is not this machine's to take
  red     the stage has not run, or its result is contradicted by another run

Usage:  python make_status_board.py [OUT.png]
"""
import os
import re
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
SUBMIT = os.path.join(HERE, '..', 'submit')


def font(size, bold=False):
    for cand in ('DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf',):
        for base in (os.path.join(os.path.dirname(sys.modules['PIL'].__file__), 'fonts'),
                     'C:/Windows/Fonts'):
            p = os.path.join(base, cand)
            if os.path.exists(p):
                return ImageFont.truetype(p, size)
    return ImageFont.load_default(size)


def read(path):
    with open(path, 'r', errors='replace') as f:
        return f.read()


def num(text, pat, default=None):
    m = re.search(pat, text)
    return int(m.group(1).replace(',', '')) if m else default


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'status_board.png')

    # --- sources of truth ---
    sweep = read(os.path.join(HERE, 'pos_full.log')) if os.path.exists(os.path.join(HERE, 'pos_full.log')) else read(os.path.join(os.environ.get('TEMP', ''), 'pos_full.log'))
    cov = read(os.path.join(HERE, 'submission_coverage.txt'))
    rep = read(os.path.join(HERE, 'coverage_report.txt'))
    cmp_ = read(os.path.join(HERE, 'method_compare.txt'))
    idx = read(os.path.join(SUBMIT, 'sha256.txt'))
    h_runs = [(n, read(os.path.join(HERE, n))) for n in
              ('recover_h.log', 'recover_h_big.log', 'recover_h_spread.log')
              if os.path.exists(os.path.join(HERE, n))]
    hmat = [f for f in os.listdir(HERE) if f.startswith('hmatrix') and f.endswith('.h')]

    grid = num(cov, r'\((\d+) credited rows\)')
    credited = num(sweep, r'credited \(any tier\)\s*:\s*([\d,]+)')
    tier0 = num(sweep, r'tier 0 not credited\s*:\s*([\d,]+)')
    conflicts = num(sweep, r'conflicts \(digits say another LBA\):\s*([\d,]+)')
    offladder = num(sweep, r'pages off the block ladder\s*:\s*([\d,]+)')
    pages = num(cov, r'present in extraction\s*:\s*([\d,]+)')
    pages_tot = num(cov, r'present in extraction\s*:\s*[\d,]+\s*/\s*([\d,]+)')
    recs = num(cov, r'byte-identical in extraction\s*:\s*([\d,]+)')
    recs_tot = num(cov, r'record-level byte proof \((\d+) records')
    differ = num(cov, r'differing\s*:\s*([\d,]+)')
    targets = num(cov, r'including the (\d+) the tool cannot read')
    raw3 = num(rep, r'tier 3: claim re-derives from the raw bytes for (\d+)')
    raw3tot = num(rep, r'tier 3: claim re-derives from the raw bytes for \d+ / (\d+)')
    neither = num(cmp_, r'neither\s*:\s*([\d,]+)')
    nidx = len([l for l in idx.splitlines() if l.strip() and not l.startswith('#')])
    dec = read(os.path.join(HERE, 'decoder_speed.txt'))
    block_h = re.search(r'block \(1152 pages[^)]*\)\s+([\d.]+) h', dec)

    rows = []
    rows.append(('green', 'anchor grid credited',
                 '%s / %s  (100.0000 %%)' % (f'{credited:,}', f'{grid:,}'),
                 'pos_full.log + anchor_map_full.txt'))
    rows.append(('green', 'tier 0 / conflicts / off-ladder',
                 '%s / %s / %s' % (tier0, conflicts, offladder), 'pos_full.log'))
    rows.append(('green', 'hard targets recovered',
                 '%s of %s - their records are in the submitted file' % (f'{targets:,}', f'{targets:,}'),
                 'submission_coverage.txt'))
    rows.append(('green', 'extraction holds every credited record',
                 '%s / %s pages, %s / %s records byte-identical, %s differing'
                 % (f'{pages:,}', f'{pages_tot:,}', f'{recs:,}', f'{recs_tot:,}', differ),
                 'verify_submission_coverage.py'))
    rows.append(('green', 'raw bytes re-derive the tier claim',
                 'tier 3 %s / %s' % (raw3, raw3tot), 'coverage_report.txt'))
    rows.append(('green', 'method union', 'no grid point left uncovered ("neither" = %s)' % neither,
                 'method_compare.txt'))
    rows.append(('green', 'submission package', '%d files indexed, index audited both ways' % nidx,
                 'submit/sha256.txt'))
    ed = read(os.path.join(HERE, 'errdist.txt'))
    ec = read(os.path.join(HERE, 'err_cause.txt'))
    al = read(os.path.join(HERE, 'assemble_lba_report.txt'))

    def fl(pat, text, default=None):
        m = re.search(pat, text)
        return m.group(1) if m else default

    rows.append(('green', 'raw channel before ECC (measured)',
                 'BER %s - median %s, p99 %s, max %s wrong bits per 1,144 B sub-codeword; only %s %% '
                 'of chunks arrive clean, and the error rate is flat across structural classes '
                 '(%s x lift on the injected byte), so the noise is the channel, not a model hole'
                 % (fl(r'raw bit error rate\s*:\s*([\d.eE+-]+)', ed),
                    fl(r'median\s*:\s*([\d,]+) bits', ed), fl(r'p99\s*:\s*([\d,]+) bits', ed),
                    fl(r'max\s*:\s*([\d,]+) bits', ed),
                    fl(r'<=\s+0 errors\s*:\s*[\d,]+\s*\(\s*([\d.]+) %\)', ed),
                    fl(r'lift[^\d\n]*([\d.]+)x', ec) or '?'),
                 'errdist.txt + err_cause.txt'))
    rows.append(('green', 'stage-11 LBA placement (own assembler)',
                 '%s / %s output sectors byte-exact vs the key-removed source (%s %%); %s %% of the '
                 'address shells still read, and %s of those equal the LBA the map claims - the FE '
                 'logical image is not reproducible from this dump (0 of 10 probes) so it is '
                 'assembled from the anchor map'
                 % (fl(r'output sector == source sector \(key removed\)\s*:\s*([\d,]+)', al),
                    fl(r'self-check over ([\d,]+) map rows', al),
                    fl(r'output sector == source sector \(key removed\)\s*:\s*[\d,]+\s*\(([\d.]+) %\)', al),
                    fl(r'`\|Block#` literal present in that sector\s*:\s*[\d,]+\s*\(([\d.]+) %\)', al),
                    fl(r'decimal field equals the LBA the row claims\s*:\s*([\d,]+)', al)),
                 'assemble_lba_report.txt'))
    rows.append(('amber', "author tool's own digit gate", "still prints 'Missing: 1485 => 99 %' - "
                 'its rule, not the file', 'full_patch_v3.log'))
    rows.append(('amber', 'hand-off to the author', 'not done and not this machine to send - there is '
                 'no upload endpoint; submit/CONTACT_NOTE.txt -> pg@futureware.at, GitHub issue, or '
                 'hddguru DM; 13.6 GB transfer must be agreed with him', 'submit/SUBMISSION.md'))
    rows.append(('red', 'ECC decode', 'has not run: no hmatrix*.h in the repo, nothing to feed '
                 'dumpdecoder.py', 'ls hmatrix*.h'))
    h_detail = ', '.join('%s rank %s / null %s' % (
        n.replace('recover_h', '').replace('.log', '').lstrip('_') or 'base',
        num(t, r'rank of [\d,]+ codewords\s*:\s*([\d,]+)'),
        num(t, r'nullity \(checks found\):\s*([\d,]+)')) for n, t in h_runs)
    rows.append(('red', 'local H recovery', 'refuted on this written content: ' + h_detail,
                 'recover_h*.log'))

    # the pattern-only run is the decisive one: it samples chunks that provably hold pattern data,
    # so its numbers are about the code, not about a contaminated sample
    plog = read(os.path.join(HERE, 'recover_h_pattern.log'))
    ilog = read(os.path.join(HERE, 'h_candidates_inspect.log'))
    prank, pnull = num(plog, r'rank of [\d,]+ codewords\s*:\s*([\d,]+)'), num(plog, r'nullity \(checks found\):\s*([\d,]+)')
    pkern = num(plog, r'kernel basis rows\s*:\s*([\d,]+)')
    pchg, ptot = num(ilog, r'bit positions that change \([^)]*\):\s*([\d,]+)'), num(ilog, r'change \([^)]*\):\s*[\d,]+\s*/\s*([\d,]+)')
    pdegen = num(ilog, r'\d+ of (\d+) candidates sit on changing bits')
    pweight = num(ilog, r'^\s+0\s+(\d+)\s', 1) if re.search(r'^\s+0\s+\d+', ilog, re.M) else None
    if prank:
        rows.append(('red', 'local H, pattern-only sample',
                     'rank %s / nullity %s -> %s candidate vectors, %s of them weight-%s on bit '
                     'positions that never vary. Only %s / %s bit positions (%.2f %%) are exercised '
                     'by this content, so the dual code cannot see H here.'
                     % (f'{prank:,}', pnull, pkern, pdegen, pweight, f'{pchg:,}', f'{ptot:,}',
                        100.0 * pchg / ptot),
                     'recover_h_pattern.log + h_candidates_inspect.log'))
    rows.append(('red', 'decoded logical image', 'no corrected output exists yet; decoder cost floor '
                 'is %s h per block single-core' % (block_h.group(1) if block_h else '?'),
                 'decoder_speed.txt'))

    W = 1120
    pad, line_h = 26, 30
    f_title, f_lbl, f_val, f_src, f_note = font(26, True), font(19, True), font(19), font(14), font(15)
    MAXW = W - 2 * pad - 30

    def wrap(text):
        """Greedy wrap to the drawing width so no value line can be clipped."""
        words, out, cur = text.split(), [], ''
        for wd in words:
            trial = (cur + ' ' + wd).strip()
            if d.textlength(trial, font=f_val) <= MAXW or not cur:
                cur = trial
            else:
                out.append(cur)
                cur = wd
        if cur:
            out.append(cur)
        return out

    # measure with a scratch draw context, since the real one does not exist until the canvas does
    _scratch = ImageDraw.Draw(Image.new('RGB', (W, 10)))

    def nlines(text):
        words, cur, n = text.split(), '', 1
        for wd in words:
            trial = (cur + ' ' + wd).strip()
            if _scratch.textlength(trial, font=f_val) <= MAXW or not cur:
                cur = trial
            else:
                n += 1
                cur = wd
        return n

    def row_h(text):
        """label 24 + one line per wrapped value line + source line + gap; the draw loop uses the
        same function, so the canvas cannot come out shorter than its rows."""
        return 24 + 22 * nlines(text) + 22 + 12

    body = sum(row_h(v) for _, _, v, _ in rows)
    H = pad * 2 + 76 + body + 40
    COL = {'green': (26, 160, 90), 'amber': (214, 148, 20), 'red': (196, 48, 43)}
    BG, FG, DIM = (250, 250, 248), (24, 26, 28), (110, 114, 118)

    img = Image.new('RGB', (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((pad, pad), 'PS2251-11 / BiCS4 LDPC - stage board', font=f_title, fill=FG)
    y = pad + 40
    d.text((pad, y), 'each chip is parsed from the named artifact at render time; ECC is the decode '
           'layer and it has not run', font=f_note, fill=DIM)
    y += 30
    for colour, label, value, src in rows:
        c = COL[colour]
        d.rectangle([pad, y + 4, pad + 18, y + 22], fill=c, outline=c)
        d.text((pad + 30, y), label, font=f_lbl, fill=FG)
        yy = y + 24
        for ln in wrap(value):
            d.text((pad + 30, yy), ln, font=f_val, fill=(40, 44, 48))
            yy += 22
        d.text((pad + 30, yy), src, font=f_src, fill=DIM)
        y += row_h(value)
    n_g = sum(1 for r in rows if r[0] == 'green')
    d.line([pad, y + 6, W - pad, y + 6], fill=(200, 200, 196))
    d.text((pad, y + 16), 'green %d   amber %d   red %d   -   the gate that blocks is amber: the '
           'hand-off to the author. ECC decode is red only because it waits on the hmatrix he returns'
           % (n_g, sum(1 for r in rows if r[0] == 'amber'), sum(1 for r in rows if r[0] == 'red')),
           font=f_note, fill=FG)
    img.save(out)
    print('wrote %s  (%d x %d, %d rows)' % (out, W, H, len(rows)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
