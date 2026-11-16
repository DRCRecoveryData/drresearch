#!/usr/bin/env python3
"""Join the positional credits with the digit-consensus grid set, per grid point.

Two independent software routes cover the same 2,064,385 grid points:

  positional_anchors.py  credits an LBA from record shell + block ladder + digit evidence
  repair_anchors.py      repairs the digits by 3-field majority vote over the whole dump

Consensus publishes its accepted set as grid indices, positional publishes one row per LBA it
evaluated with a tier.  Three states per grid point have to be told apart, and conflating them
is how a method conflict looks like coverage:

  credited by both        agreement
  credited by one only    the other route's gate rejected it - reported with the rejecting
                          route's own measurements so the reason is checkable
  evaluated, not credited positional saw the slot and refused it, which is a conflict with a
                          consensus acceptance, not a hole
  never evaluated         the slot's page had no base, so positional emitted no row at all

Usage:  python compare_methods.py [TABLE [ALLGRID]]
Writes: method_compare.txt, target_methods.txt
"""
import os
import sys
import numpy as np

BORDER = int(os.environ.get('BORDER', 3145728))
INC = int(os.environ.get('INC', 9))
ECCEND = int(os.environ.get('ECCEND', 21725184))
GRID = (ECCEND - BORDER) // INC + 1

TABLE = os.environ.get('TABLE') or (sys.argv[1] if len(sys.argv) > 1 else 'positional_full.txt')
ALLGRID = os.environ.get('ALLGRID') or (sys.argv[2] if len(sys.argv) > 2 else 'repair_full.txt.allgrid.txt')
REPTAB = os.environ.get('REPTAB', 'repair_full.txt')
TARG = os.environ.get('TARGETS', 'real_hard_missing.txt')
OUT = os.environ.get('OUT', 'method_compare.txt')
TOUT = os.environ.get('TOUT', 'target_methods.txt')

COLS = ('LBA', 'parsed', 'credited', 'tier', 'base_src', 'labelbits', 'digitbits',
        'totalbits', 'rivalbits', 'legal_digits', 'blk', 'page', 'sect', 'off')


def grid_of(lba):
    """Grid index of an LBA, or -1 when it is off the inc=9 grid or outside the window."""
    lba = np.asarray(lba, dtype=np.int64)
    d = lba - BORDER
    ok = (d >= 0) & (d % INC == 0) & (lba <= ECCEND)
    return np.where(ok, d // INC, -1)


def load_positional(path):
    a = np.atleast_2d(np.loadtxt(path, dtype=np.int64, comments='#'))
    if a.size == 0:
        raise SystemExit('positional table %s is empty' % path)
    if a.shape[1] != len(COLS):
        raise SystemExit('positional table %s has %d columns, expected %d'
                         % (path, a.shape[1], len(COLS)))
    d = {k: a[:, i] for i, k in enumerate(COLS)}
    return d


def load_allgrid(path):
    idx = np.atleast_1d(np.loadtxt(path, dtype=np.int64))
    return np.unique(idx)


def load_repairs(path):
    """target LBA -> (skeleton_bits, field_votes, dist) from the consensus target table."""
    if not os.path.exists(path):
        return {}
    a = np.atleast_2d(np.loadtxt(path, dtype=np.int64, comments='#'))
    if a.size == 0:
        return {}
    return {int(r[0]): (int(r[2]), int(r[3]), int(r[4])) for r in a}


def main():
    pos = load_positional(TABLE)
    cons_idx = load_allgrid(ALLGRID)
    rep = load_repairs(REPTAB)
    targets = np.unique(np.atleast_1d(np.loadtxt(TARG, dtype=np.int64)))

    pidx = grid_of(pos['LBA'])
    bad = int((pidx < 0).sum())
    if bad:
        raise SystemExit('%d positional rows are off the grid - the table is not joinable' % bad)

    credited = pos['credited'].astype(bool)
    ev_idx = pidx[~credited]           # seen and refused
    cr_idx = pidx[credited]            # seen and credited
    # the positional table is deduped per LBA, so distinct grid indices must equal row counts
    assert len(np.unique(cr_idx)) == len(cr_idx) and len(np.unique(ev_idx)) == len(ev_idx), \
        'duplicate grid points in %s - the join would silently drop rows' % TABLE

    cset = np.zeros(GRID, np.bool_)
    cset[cons_idx] = True
    pset = np.zeros(GRID, np.bool_)
    pset[cr_idx] = True
    eset = np.zeros(GRID, np.bool_)
    eset[ev_idx] = True                # evaluated-but-refused only, disjoint from pset

    both = int((cset & pset).sum())
    p_only = int((pset & ~cset).sum())
    c_only_refused = int((cset & eset).sum())      # real conflict
    c_only_unseen = int((cset & ~(pset | eset)).sum())  # positional emitted no row
    union = int((cset | pset).sum())

    lines = []
    w = lines.append
    w('method comparison: positional vs digit consensus')
    w('  positional table : %s  (%d rows, %d credited, %d refused)'
      % (TABLE, len(pos['LBA']), int(credited.sum()), int(eset.sum())))
    w('  consensus set    : %s  (%d accepted grid indices of %d)'
      % (ALLGRID, len(cons_idx), GRID))
    w('  consensus target table: %s (%d rows)' % (REPTAB, len(rep)))
    w('')
    w('grid points (of %d):' % GRID)
    w('  credited by positional        : %10d  (%.4f%%)' % (int(pset.sum()), 100.0 * pset.sum() / GRID))
    w('  accepted by consensus         : %10d  (%.4f%%)' % (int(cset.sum()), 100.0 * cset.sum() / GRID))
    w('  credited by BOTH              : %10d' % both)
    w('  positional only, no consensus : %10d' % p_only)
    w('  consensus, positional REFUSED : %10d   <-- method conflict, explained below' % c_only_refused)
    w('  consensus, positional no row  : %10d   (page had no base, so never evaluated)' % c_only_unseen)
    w('  union of the two routes       : %10d  (%.4f%%)' % (union, 100.0 * union / GRID))
    w('  neither                       : %10d' % (GRID - union))

    conf = cset[ev_idx] if len(ev_idx) else np.zeros(0, np.bool_)
    if c_only_refused:
        why = pos['totalbits'][~credited][conf]
        lab = pos['labelbits'][~credited][conf]
        dig = pos['digitbits'][~credited][conf]
        tier = pos['tier'][~credited][conf]
        w('')
        w('conflict rows - what positional measured where consensus accepted:')
        w('  total distance  : min %d median %d max %d   (gate TOT_MAX)'
          % (why.min(), int(np.median(why)), why.max()))
        w('  label distance  : min %d median %d max %d   (gate LAB_MAX)'
          % (lab.min(), int(np.median(lab)), lab.max()))
        w('  digit distance  : min %d median %d max %d' % (dig.min(), int(np.median(dig)), dig.max()))
        w('  tier recorded   : ' + ', '.join('%s:%d' % (t, int((tier == t).sum()))
                                            for t in (3, 1, 2, 0) if (tier == t).any()))

    pon = ~cset[cr_idx] if len(cr_idx) else np.zeros(0, np.bool_)
    if p_only:
        w('')
        w('positional-only rows - consensus rejected these; its gate was skeleton<=12 bits:')
        pl = pos['labelbits'][credited][pon]
        pt = pos['tier'][credited][pon]
        if len(pl):
            w('  label distance : min %d median %d max %d' % (pl.min(), int(np.median(pl)), pl.max()))
        if len(pt):
            w('  tier: ' + ', '.join('%s:%d' % (t, int((pt == t).sum()))
                                    for t in (3, 1, 2) if (pt == t).any()))

    tidx = grid_of(targets)
    if (tidx < 0).any():
        w('')
        w('  WARNING %d targets are off the grid' % int((tidx < 0).sum()))
    tok = tidx >= 0
    ti = tidx[tok]
    tin_c = cset[ti]
    tin_p = pset[ti]
    w('')
    w('the %d hard targets:' % len(targets))
    w('  both routes           : %d' % int((tin_c & tin_p).sum()))
    w('  consensus only        : %d' % int((tin_c & ~tin_p).sum()))
    w('  positional only       : %d' % int((tin_p & ~tin_c).sum()))
    w('  neither               : %d' % int((~tin_c & ~tin_p).sum()))

    with open(OUT, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))

    with open(TOUT, 'w') as f:
        f.write('# LBA consensus positional_tier positional_credited labelbits digitbits '
                'totalbits cons_skel cons_votes cons_dist method\n')
        for L, g in zip(targets[tok], ti):
            row = np.where(pos['LBA'] == L)[0]
            if len(row):
                i = int(row[0])
                pt, pc, lb, dg, tt = (int(pos['tier'][i]), int(pos['credited'][i]),
                                      int(pos['labelbits'][i]), int(pos['digitbits'][i]),
                                      int(pos['totalbits'][i]))
            else:
                pt, pc, lb, dg, tt = -1, 0, -1, -1, -1
            cs = rep.get(int(L), (-1, -1, -1))
            mc = bool(cset[g])
            mp = bool(pset[g])
            method = ('both' if mc and mp else 'consensus' if mc else
                      'positional' if mp else 'neither')
            f.write('%d %d %d %d %d %d %d %d %d %d %s\n'
                    % (L, int(mc), pt, pc, lb, dg, tt, cs[0], cs[1], cs[2], method))
    print('\nper-target join: %s' % TOUT)
    print('summary        : %s' % OUT)
    return 0


if __name__ == '__main__':
    sys.exit(main())
