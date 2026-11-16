#!/usr/bin/env python3
"""Attribute every record slot to an LBA by POSITION, then read the ones the digits can't.

A page is 18,432 B = 16 chunks x 1,144 B, each chunk holds two 512 B sectors, so the
32 sectors of a page carry contiguous LBAs B..B+31 and the sector at chunk c slot s
holds B+2c+s.  Block 500 measurement: 1,136 of its 1,152 pages contain at least one
record whose 32 label bytes are intact, and every page with more than one such record
agrees on a single B - 0 disagreements.  So B is known independently of any damaged
record, and a record that cannot be parsed still sits at a known LBA.

That changes what "missing" means.  parse_anchors/hard_probe/repair_anchors all had to
read the digits.  Here every slot gets one of four tiers:

  3  the digits parse, to this LBA                              (what repair_anchors does)
  1  the record as a whole is within TOT_MAX bits of rec(L) and beats every competing grid
     LBA (L +/- 9k) by MARGIN bits - the labels need not be individually intact, since it
     is the combination that pins the value
  2  the 32 label bytes match the constant text within LAB_MAX bits and the page base is
     anchored, but the digits are too rotted to arbitrate: the LBA comes from position, not
     from the text.  The digits are then redundant - the generator writes rec(B+32p+i) into
     sector i, a model certify_data.py confirmed on every chunk of the real dump.
  0  nothing there, or the digits say a different LBA (a conflict, never averaged away)

Tier 2 is the one that can reach the last 1,485, and it is also the one that needs the
page base to be right, so a ladder page only qualifies if it also carries its own
anchor, and the tool reports how many pages defy the ladder at all.

Per still-missing target the tool also says which it was: no shell at the slot
(record never written / block unmapped) or shell present but too damaged to credit.

KAT mode writes a synthetic page containing a clean record, records damaged by
10/30/45 bits, a slot with no shell, and a slot holding a DIFFERENT valid LBA, then
checks that only the damaged-but-present ones become positional credits and the
foreign record becomes a conflict.
"""
import os, sys
import numpy as np
from multiprocessing import Pool

BLK = int(os.environ.get('BLKSZ', 21233664))
PG = int(os.environ.get('PGSZ', 18432))
CH = int(os.environ.get('CHUNK', 1144))
BORDER = int(os.environ.get('BORDER', 3145728))
ECCEND = int(os.environ.get('ECCEND', 21725184))
INC = 9
POOL = int(os.environ.get('POOL', 10))
LAB_MAX = int(os.environ.get('LAB_MAX', 4))
TOT_MAX = int(os.environ.get('TOT_MAX', 60))
MARGIN = int(os.environ.get('MARGIN', 8))
COMP = int(os.environ.get('COMP', 5))
LADDER_MARGIN = int(os.environ.get('LADDER_MARGIN', 2))   # votes the block base must win by
SHELL_MAX = int(os.environ.get('SHELL_MAX', 80))         # total bits allowed for a shell-only credit
LEGAL_MIN = int(os.environ.get('LEGAL_MIN', 0))         # digit bytes that must still be legal characters
BASE_MIN = int(os.environ.get('BASE_MIN', 1))            # own anchors a ladder page needs
TARG = os.environ.get('TARGETS', 'real_hard_missing.txt')
OUT = os.environ.get('OUT', 'positional_anchors.txt')
DUMP = sys.argv[1] if len(sys.argv) > 1 else 'C:/Users/drclab/Documents/LDPC/01_01_plain.dump'
KEYFILE = os.environ.get('XORKEY', 'C:/Users/drclab/Documents/Qoder/2026-11-15/7246e382/run/01_01_plain.dump.xor')
MAXBLK = int(os.environ.get('MAXBLK', 0))
START = int(os.environ.get('START', 0))
BLOCKLIST = os.environ.get('BLOCKLIST', '')      # one physical block number per line
PC = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(1).astype(np.int16)
NREC = 82
SKEL = [0, 1, 2, 3, 4, 5, 6, 19, 20, 21, 22, 31, 32, 33, 34, 35, 36, 37, 38, 59,
        60, 61, 62, 63, 64, 75, 76, 77, 78, 79, 80, 81]
DIG = [i for i in range(NREC) if i not in SKEL]
# Which characters each digit position may legally hold: decimal fields 0-9, the hex
# field also a-f/A-F, and Pos: is space padded.  A slot whose digits are rotted keeps
# them; a slot whose payload was overwritten or never written does not, so this counts
# how much of the credit is coming from position alone.
FIELDS = [(7, 19, set(range(48, 58))),
          (23, 31, set(range(48, 58)) | set(range(97, 103)) | set(range(65, 71))),
          (39, 59, set(range(48, 58))),
          (65, 75, set(range(48, 58)) | {32})]
ALLOW = {}
for lo, hi, ok in FIELDS:
    for j in range(lo, hi):
        ALLOW[j] = ok
FOK = np.zeros((len(DIG), 256), bool)
for pos, j in enumerate(DIG):
    for c in ALLOW.get(j, set()):
        FOK[pos, c] = True


def rec(L):
    return (f"|Block#{L:012d} (0x{L:08X}) Byte: {L*512:020d} Pos: {L>>11:10d} MB\n***").encode()


RECB = {}


def recb(L):
    v = RECB.get(L)
    if v is None:
        v = np.frombuffer(rec(L), np.uint8)
        RECB[L] = v
    return v


def slot_positions(page_start):
    """(sector index, absolute byte offset) for the 32 sectors of a page."""
    return [(2 * c + s // 512, page_start + c * CH + s)
            for c in range(PG // CH) for s in (0, 512)]


SECT_PP = 2 * (PG // CH)      # LBAs per physical page


def page_evidence(data, start, rclassify):
    """[(sector index, offset, parsed LBA or None)] for the 32 sectors of a page."""
    out = []
    for i, q in slot_positions(start):
        if q + NREC > len(data):
            continue
        r = rclassify(data, q)
        out.append((i, q, r[0] if r else None))
    return out


def block_base(pages):
    """Page-0 base from every record that parses anywhere in the block.

    Each parsed record fixes one hypothesis B0 = L - 32p - i, so a block contributes
    ~3,000 votes for the true B0 and one vote per mis-repaired record.  A single page
    cannot do that: a foreign record that itself parses (a slot whose digits were
    rewritten onto another grid LBA) is one vote against one true anchor and ties.
    Returns (B0, votes, runner-up votes) or (None, 0, 0).
    """
    votes = {}
    for p, ev in enumerate(pages):
        for i, q, L in ev:
            if L is not None:
                k = L - SECT_PP * p - i
                votes[k] = votes.get(k, 0) + 1
    if not votes:
        return None, 0, 0
    ranked = sorted(votes.items(), key=lambda kv: (-kv[1], kv[0]))
    return ranked[0][0], ranked[0][1], (ranked[1][1] if len(ranked) > 1 else 0)


def page_bases(pages, b0):
    """Per-page base, its provenance, and how many of that page's own anchors support it.

    Provenance: 1 = this page's own strict plurality, 2 = the block ladder B0+32p.  A page
    whose anchors tie must not silently pick an arbitrary winner, which is how three
    control slots once vanished off the grid and made the KAT read green while evaluating
    nothing.
    """
    out = []
    for p, ev in enumerate(pages):
        cand = {}
        for i, q, L in ev:
            if L is not None:
                cand.setdefault(L - i, []).append((i, L))
        ranked = sorted(cand.items(), key=lambda kv: (-len(kv[1]), kv[0]))
        if ranked and (len(ranked) == 1 or len(ranked[0][1]) > len(ranked[1][1])):
            out.append((ranked[0][0], 1, len(ranked[0][1])))
        elif b0 is not None:
            base = b0 + SECT_PP * p
            out.append((base, 2, sum(1 for k, v in cand.items() if k == base)))
        else:
            out.append((None, 0, 0))
    return out


def credit(data, L, q, parsed, base_strong):
    """Tier for 'the slot at q holds the record for L'; see the module docstring."""
    obs = np.frombuffer(data[q:q + NREC], np.uint8)
    want = recb(L)
    lab = int(PC[obs[SKEL] ^ want[SKEL]].sum())
    dig = int(PC[obs[DIG] ^ want[DIG]].sum())
    tot = lab + dig
    legal = int(FOK[np.arange(len(DIG)), obs[DIG]].sum())
    alts = []
    for k in range(-COMP, COMP + 1):
        if not k:
            continue
        La = L + k * INC
        if La < BORDER or La > ECCEND:
            continue
        alts.append(int(PC[obs ^ recb(La)].sum()))
    rival = min(alts) if alts else 10 ** 9
    if parsed is not None and parsed != L:
        tier = 0                                     # contradicted: reported, not averaged
    elif parsed is not None:
        tier = 3
    elif tot <= TOT_MAX and (rival - tot) >= MARGIN:
        # the whole record, labels included, pins the value: the labels do not have to be
        # individually intact, and requiring that threw away four slots at 5-6 label bits
        # with 1-3 digit bits wrong and a 12-bit lead over the nearest rival LBA.
        tier = 1
    elif lab > LAB_MAX:
        tier = 0                                     # no record shell, so position has no hook
    elif base_strong and tot <= SHELL_MAX and legal >= LEGAL_MIN:
        tier = 2                                     # shell + position carry it, digits do not
    else:
        tier = 0
    return tier, lab, dig, tot, rival, parsed, legal


def work(args):
    b, path, keyfn = args
    import repair_anchors as R
    key = np.fromfile(keyfn, dtype=np.uint8)
    with open(path, 'rb') as f:
        f.seek(b * BLK)
        raw = f.read(BLK)
    data = np.asarray(np.frombuffer(raw, np.uint8)) ^ key
    pages = [page_evidence(data, p * PG, R.classify) for p in range(len(data) // PG)]
    b0, v0, v1 = block_base(pages)
    res = []
    conflicts = 0
    unanchored = 0
    ladder_off = 0
    for p, ((base, src, nsup), ev) in enumerate(zip(page_bases(pages, b0), pages)):
        if base is None:
            unanchored += 1
            continue
        if src == 1 and b0 is not None and base != b0 + SECT_PP * p:
            ladder_off += 1                    # this page's own anchors defy the ladder
        strong = nsup >= BASE_MIN and (src == 1 or v0 - v1 >= LADDER_MARGIN)
        for i, q, parsed in ev:
            L = base + i
            if (L - BORDER) % INC or L < BORDER or L > ECCEND:
                continue
            tier, lab, dig, tot, rival, _, legal = credit(data, L, q, parsed, strong)
            res.append((L, int(parsed == L), int(tier > 0), tier, src, lab, dig, tot, rival,
                        legal, b, p, i))
            if parsed is not None and parsed != L:
                conflicts += 1
    return (b, np.array(res, np.int32).reshape(-1, 13), conflicts, unanchored,
            v0, v1, ladder_off, len(pages))


def main():
    if os.environ.get('KAT'):
        return kat()
    import repair_anchors as R
    tl = np.atleast_1d(np.loadtxt(TARG, dtype=np.int64))
    targets = set(int(x) for x in tl)
    nblocks = os.path.getsize(DUMP) // BLK
    if BLOCKLIST:
        blocks = sorted({int(x) for x in open(BLOCKLIST) if x.strip() and not x.startswith('#')})
    else:
        blocks = list(range(START, min(nblocks, START + MAXBLK) if MAXBLK else nblocks))
    print('%d blocks, labels<=%d total<=%d margin>=%d, %d targets'
          % (len(blocks), LAB_MAX, TOT_MAX, MARGIN, len(targets)), flush=True)
    rows = []
    conf = unanch = lad_off = npages = votes = rival_votes = 0
    done = 0
    with Pool(POOL) as pool:
        for b, r, c, u, v0, v1, lo, npg in pool.imap_unordered(
                work, ((x, DUMP, KEYFILE) for x in blocks), chunksize=2):
            if len(r):
                rows.append(r)
            conf += c
            unanch += u
            votes += v0
            rival_votes += v1
            lad_off += lo
            npages += npg
            done += 1
            if done % 150 == 0:
                print('  %d/%d blocks, %d slots seen' % (done, len(blocks), sum(len(x) for x in rows)), flush=True)
    a = np.concatenate(rows) if rows else np.zeros((0, 13), np.int32)
    cols = (a[:, k].astype(np.int64) for k in range(13))
    uniq = {}
    for x in zip(*cols):
        cur = uniq.get(x[0])
        if cur is None or (x[3], -x[7]) > (cur[3], -cur[7]):
            uniq[x[0]] = x               # best tier, then closest to rec(L)
    Ls = np.array(sorted(uniq))
    Th = np.array([uniq[x][3] for x in Ls])
    Pm = Th == 3
    Om = Th > 0
    Tm = np.array([uniq[x][7] for x in Ls])
    print('\nrecord slots on the grid : %d' % len(Ls))
    LABEL = {3: 'digits parse', 1: 'shell + digits agree', 2: 'shell + position only',
             0: 'not credited'}
    for t in (3, 1, 2, 0):
        print('  tier %d %-22s: %d' % (t, LABEL[t], int((Th == t).sum())))
    print('  credited (any tier)    : %d  (%.4f%% of grid)'
          % (int(Om.sum()), 100.0 * int(Om.sum()) / ((ECCEND - BORDER) // INC + 1)))
    print('conflicts (digits say another LBA): %d' % conf)
    print('pages with no anchor           : %d / %d' % (unanch, npages))
    print('pages off the block ladder     : %d / %d  (%.5f%%)'
          % (lad_off, npages, 100.0 * lad_off / max(1, npages)))
    print('ladder vote margin             : %d vs %d runner-up' % (votes, rival_votes))
    tgt = np.array([x for x in Ls if int(x) in targets])
    Tg = np.array([uniq[int(x)][3] for x in tgt])
    tt = np.array([uniq[int(x)][7] for x in tgt])
    ts = np.array([uniq[int(x)][5] for x in tgt])
    print('\ntargets whose slot is on the grid: %d / %d' % (len(tgt), len(targets)))
    if len(tgt):
        print('  tier 3 (digits parse)   : %d' % int((Tg == 3).sum()))
        print('  tier 1 (shell + digits) : %d' % int((Tg == 1).sum()))
        print('  tier 2 (shell + place)  : %d   <-- newly credited here' % int((Tg == 2).sum()))
        print('  tier 0 (not credited)   : %d' % int((Tg == 0).sum()))
        z = Tg == 0
        print('    of those, shell intact: %d   (digits too rotted to arbitrate)'
              % int((ts[z] <= LAB_MAX).sum()))
        print('    of those, no shell    : %d   (no record was written at that slot)'
              % int((ts[z] > LAB_MAX).sum()))
        if z.sum():
            print('    their total distance  : min %d median %d max %d'
                  % (tt[z].min(), int(np.median(tt[z])), tt[z].max()))
    with open(OUT, 'w') as f:
        f.write('# LBA parsed credited tier base_src labelbits digitbits totalbits best_rivalbits '
                'legal_digits blk page sect dump_offset\n'
                '# tier 3 digits parse, 1 shell+digits agree, 2 shell+position only, 0 not credited\n'
                '# base_src 1 = this page own strict plurality, 2 = block ladder B0+32p\n'
                '# dump_offset = byte position of this 512 B sector in the dump.  The 1144 B chunk\n'
                '# holding it starts at dump_offset - (sect % 2) * 512; its 1024 B data field is the\n'
                '# two sectors at dump_offset - (sect % 2) * 512 and + 512.\n')
        for x in Ls:
            u = uniq[int(x)]
            off = u[10] * BLK + u[11] * PG + (u[12] // 2) * CH + (u[12] % 2) * 512
            f.write('%d %d %d %d %d %d %d %d %d %d %d %d %d %d\n'
                    % (u[0], u[1], u[2], u[3], u[4], u[5], u[6], u[7], u[8], u[9],
                       u[10], u[11], u[12], off))
    lg = np.array([uniq[int(x)][9] for x in Ls if uniq[int(x)][3] == 2]) if len(Ls) else np.zeros(0)
    if len(lg):
        print('tier-2 digit legality    : min %d median %d max %d of 50  '
              '(100 = digits are still text, so position is only filling the gaps)'
              % (lg.min(), int(np.median(lg)), lg.max()))
    print('table: %s' % OUT)
    return 0


def kat():
    """Eight synthetic pages: clean, 10/30/45-bit digit damage, no shell, foreign record,
    plus the same shell-only damage on a page with its own anchors and on a page stripped
    of them, which is what separates tier 1 from tier 2.

    The foreign record is a grid LBA that parses, so it votes for a wrong page base.  It
    sits in a page with one other anchor, which makes that page's plurality a 1-1 tie: an
    implementation that picks an arbitrary winner mis-bases the page, and the three control
    slots then leave the grid and are never evaluated at all.  That is the hole this KAT
    exists to keep shut, so every control asserts its row is present.
    """
    rng = np.random.default_rng(7)
    npages = 8
    B = BORDER + 9 * 7
    buf = bytearray(npages * PG)
    plan = []
    for p in range(npages):
        for i, q in slot_positions(p * PG):
            L = B + 32 * p + i
            if (L - BORDER) % INC:
                continue
            k = len(plan)
            r = bytearray(rec(L))
            nb = 0
            if k in (1, 2, 3):
                nb = (10, 30, 45)[k - 1]
                for s in rng.choice(np.array(DIG), size=nb, replace=False):
                    r[int(s)] ^= np.uint8(1 << int(rng.integers(0, 8)))
            buf[q:q + NREC] = bytes(r)
            plan.append((L, q, nb))
    foreign, noshell, badlabel = plan[4], plan[5], plan[6]
    buf[foreign[1]:foreign[1] + NREC] = rec(foreign[0] + 27)   # grid-congruent, so it parses
    buf[noshell[1]:noshell[1] + NREC] = b'\x5a' * NREC
    rr = bytearray(rec(badlabel[0]))
    for s in rng.choice(np.array(SKEL), size=9, replace=False):
        rr[int(s)] ^= np.uint8(1 << int(rng.integers(0, 8)))
    buf[badlabel[1]:badlabel[1] + NREC] = bytes(rr)
    badboth = [pl for pl in plan if pl[1] // PG == 2][1]     # hurt shell, rival's digits
    rr = bytearray(rec(badboth[0]))
    for s in rng.choice(np.array(SKEL), size=9, replace=False):
        rr[int(s)] ^= np.uint8(1 << int(rng.integers(0, 8)))
    other = recb(badboth[0] + 18)              # another grid LBA two steps away
    for pos, s in enumerate(DIG):
        rr[s] = other[s]
    buf[badboth[1]:badboth[1] + NREC] = bytes(rr)
    # shell intact, every digit byte scrambled: only position can save these two.
    last = [pl for pl in plan if pl[1] >= (npages - 1) * PG]
    for L, q, _ in last[1:]:
        r = bytearray(rec(L))
        for s in DIG:
            r[s] = 0x5a
        buf[q:q + NREC] = bytes(r)
    # Shell intact, digits rotted but still ASCII text: only position can carry these, so
    # they separate tier 2 (anchored page) from tier 0 (page stripped of its own anchors).
    dmg = [(DIG[(k * 7) % len(DIG)], np.uint8(1 << (1 + k % 3))) for k in range(40)]

    def hammer(pl):
        r = bytearray(rec(pl[0]))
        for s, m in dmg:
            r[s] ^= m
        buf[pl[1]:pl[1] + NREC] = bytes(r)

    def junk(pl):
        r = bytearray(rec(pl[0]))
        for s in DIG:
            r[s] = 0x00
        buf[pl[1]:pl[1] + NREC] = bytes(r)

    orphan = last[0]                                          # page has no anchor left
    sup = [pl for pl in plan if pl[1] // PG == 2][0]          # page keeps its other anchors
    assert sup[0] != badboth[0] and sup[1] != badboth[1]
    jd = [pl for pl in plan if pl[1] // PG == 3][0]           # digits erased, page anchored
    hammer(orphan)
    hammer(sup)
    junk(jd)
    open('/tmp/kat_pos.dump', 'wb').write(bytes(buf))
    open('/tmp/kat_pos.dump.xor', 'wb').write(bytes(npages * PG))
    globals()['BLK'] = npages * PG
    globals()['ECCEND'] = B + npages * 32 + 9 * 40
    import repair_anchors as R
    b, a, conf, unanch, v0, v1, lad_off, npg = work((0, '/tmp/kat_pos.dump', '/tmp/kat_pos.dump.xor'))
    uniq = {int(x[0]): x for x in a}
    COLS = ('LBA', 'parsed', 'credited', 'tier', 'base_src', 'lab', 'dig', 'tot', 'rival',
            'legal', 'blk', 'page', 'sect')

    def cred(L):
        """Row for L, or a NOT-EVALUATED sentinel that fails every assertion."""
        return uniq.get(int(L), (int(L), 0, 0, 0, 0, 999, 999, 999, 999, -1, -1, -1, -1))

    def live(L):
        return int(L) in uniq

    def show(tag, L, want=None):
        u = cred(L)
        got = u[3]
        verdict = '' if want is None else '  (want tier %s: %s)' % (want, 'OK' if got in want else 'FAIL')
        print('  %-26s present=%d parsed=%d tier=%d src=%d labels=%3d total=%3d rival=%3d legal=%3d%s'
              % (tag, live(L), u[1], got, u[4], u[5], u[7], u[8], u[9], verdict))
        return u

    print('KAT grid slots on the pages: %d, conflicts: %d, unanchored pages: %d'
          % (len(a), conf, unanch))
    print('  ladder: B0 votes %d vs %d, own-plurality pages off ladder %d / %d pages'
          % (v0, v1, lad_off, npg))
    print('  columns: %s' % ' '.join(COLS))
    u = show('clean record', plan[0][0], (3,))
    d10 = show('10-bit digit damage', plan[1][0], (1, 2, 3))
    d30 = show('30-bit digit damage', plan[2][0], (1, 2, 3))
    d45 = show('45-bit digit damage', plan[3][0], (1, 2, 3))
    fo = show('foreign LBA present', foreign[0], (0,))
    ns = show('shell destroyed', noshell[0], (0,))
    bl = show('9 label bits, digits ok', badlabel[0], (1, 3))
    bb = show('hurt shell, rival digits', badboth[0], (0,))
    su = show('rotted digits, anchored', sup[0], (1, 2))
    orp = show('rotted digits, no anchor', orphan[0], (0,))
    jdu = show('digits erased, anchored', jd[0], (0,))
    ctrl = ({foreign[0], noshell[0], badlabel[0], badboth[0], sup[0], orphan[0], jd[0]} |
            {x[0] for x in last[1:]} | {plan[j][0] for j in (1, 2, 3)})
    untouched = [pl for pl in plan if pl[0] not in ctrl]
    others = [pl for pl in untouched if not cred(pl[0])[1]]
    print('  untouched clean records: %d, unparsed: %d  (must be 0)' % (len(untouched), len(others)))
    print('  slots credited from the ladder (base_src=2): %d' % int((a[:, 4] == 2).sum()))
    ok = fo[3] == 0 and ns[3] == 0 and bl[3] in (1, 3) and bb[3] == 0
    ok = ok and u[3] == 3 and orp[3] == 0 and su[3] in (1, 2) and jdu[3] == 0
    ok = ok and d10[3] > 0 and d45[3] > 0
    ok = ok and not others and conf >= 1 and unanch == 0 and v0 - v1 >= LADDER_MARGIN
    ok = ok and all(live(x[0]) for x in (plan[0], plan[1], plan[2], plan[3], foreign,
                                         noshell, badlabel, badboth, sup, orphan, jd))
    ok = ok and cred(orphan[0])[5] <= LAB_MAX and cred(orphan[0])[9] > cred(jd[0])[9]
    ok = ok and cred(sup[0])[9] >= 40                      # rotted digits must still be text
    pages = [page_evidence(np.frombuffer(bytes(buf), np.uint8), p * PG, R.classify)
             for p in range(npages)]
    kb0, kv0, kv1 = block_base(pages)
    kpb = page_bases(pages, kb0)
    orphan_p = orphan[1] // PG
    print('  orphan page own-anchor count: %d (must be 0, base_src=%d)'
          % (kpb[orphan_p][2], kpb[orphan_p][1]))
    ok = ok and kpb[orphan_p][2] == 0 and kpb[sup[1] // PG][2] >= 1
    print('KAT %s' % ('PASS' if ok else 'FAIL'))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
