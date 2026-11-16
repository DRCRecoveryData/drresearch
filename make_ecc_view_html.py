"""Compact HTML view of one extracted page: same DATA/SA/ECC colouring as
dumphex.pl, but one cell per chunk field instead of one div per byte, so a
browser lays it out in milliseconds instead of wedging on 18k inline divs."""
import os
import html

DUMP = os.environ.get('DUMP', '../submit/upload_ps2251-11_bics4(18432p).dump')
CASE = os.environ.get('CASE', '../submit/geometry_ps2251-11_bics4.case')
PAGES = os.environ.get('PAGES', '0-2')
OUT = os.environ.get('OUT', 'ecc_view_compact(18432p).html')

PGSZ = 18432
CH = 1144
DATA, SA, ECC = 1024, 6, 114
COL = {'DATA': '#a0ffa0', 'SA': '#ffa0a0', 'ECC': '#ffffa0', 'REST': '#d8d8d4'}


def fields_from_case(path):
    import re
    txt = open(path, 'rb').read().replace(b'\x00', b'').decode('utf-8', 'replace')
    return [(n, int(a), int(b)) for n, a, b in re.findall(
        r'StructureDefinitionName="(DATA|SA|ECC)"\s+StartAddress="(\d+)"\s+StopAddress="(\d+)"', txt)]


def asc(b):
    return ''.join(chr(x) if 32 <= x < 127 else '.' for x in b)


def hx(b, limit=None):
    vals = list(b)
    if limit and len(vals) > limit:
        return ' '.join(f'{x:02X}' for x in vals[:limit]) + f' … (+{len(vals)-limit} B)'
    return ' '.join(f'{x:02X}' for x in vals)


lo, hi = (int(x) for x in PAGES.split('-')) if '-' in PAGES else (0, int(PAGES))
fields = fields_from_case(CASE)
assert len(fields) == 48, f'expected 16 chunks x 3 fields, parsed {len(fields)}'

parts = ['<html><head><meta charset="utf-8"><title>PS2251-11 / BiCS4 page view</title>',
         '<style>body{font:13px/1.45 Consolas,monospace;background:#f5f5f3;color:#232629;margin:18px}',
         'table{border-collapse:collapse;width:100%}td{vertical-align:top;padding:3px 6px;',
         'border-bottom:1px solid #ddd;white-space:pre-wrap;word-break:break-all}',
         '.k{font-weight:bold}.n{text-align:right;color:#787c80;width:34px}',
         'td.n,th,td.k{white-space:nowrap}col.f{width:52px}col.o{width:104px}col.b{width:56px}',
         'td:nth-child(2),td:nth-child(3),td:nth-child(4){white-space:nowrap}',
         'h1{font-size:17px;margin:0 0 4px}h2{font-size:14px;margin:18px 0 6px;color:#232629}',
         '.meta{color:#787c80;font-size:12px;margin-bottom:10px}</style></head><body>',
         '<h1>PS2251-11 / BiCS4 &mdash; extracted pages, fields per '
         '<code>geometry_ps2251-11_bics4.case</code></h1>',
         f'<div class="meta">page {PGSZ} B = 16 x (1,024 DATA + 6 SA + 114 ECC) + 128 B '
         f'remainder &nbsp;·&nbsp; green DATA, red SA, yellow ECC &nbsp;·&nbsp; source '
         f'{html.escape(os.path.basename(DUMP))}</div>']

with open(DUMP, 'rb') as fh:
    for pg in range(lo, hi + 1):
        fh.seek(pg * PGSZ)
        page = fh.read(PGSZ)
        if len(page) != PGSZ:
            parts.append(f'<h2>page {pg}</h2><div class="meta">short read</div>')
            break
        label = asc(page[:82]).replace('\n', '\\n').replace('\r', '\\r')
        parts.append(f'<h2>page {pg} &mdash; offset {pg*PGSZ:,} B</h2>')
        parts.append(f'<div class="meta">label: {html.escape(label)}</div>')
        parts.append('<table><colgroup><col style="width:34px"><col class="f">'
                     '<col class="o"><col class="b"></colgroup>'
                     '<tr><td class="n">#</td><td class="k">field</td>'
                     '<td class="k">page offset</td><td class="k">bytes</td><td class="k">content</td></tr>')
        for i in range(16):
            base = i * CH
            for name, a, b in fields[3 * i:3 * i + 3]:
                blob = page[a:b + 1]
                body = asc(blob[:120]) if name == 'DATA' else hx(blob, 48 if name == 'ECC' else None)
                parts.append(f'<tr><td class="n">{i}</td><td style="background:{COL[name]}">{name}</td>'
                             f'<td>{a}..{b}</td><td>{b-a+1}</td><td>{html.escape(body)}</td></tr>')
        rest = page[16 * CH:]
        parts.append(f'<tr><td class="n">-</td><td style="background:{COL["REST"]}">REST</td>'
                     f'<td>{16*CH}..{PGSZ-1}</td><td>{len(rest)}</td>'
                     f'<td>all 0x00: {set(rest) == {0}}</td></tr>')
        parts.append('</table>')

parts.append('</body></html>')
open(OUT, 'w', encoding='utf-8').write('\n'.join(parts))
print('wrote', os.path.abspath(OUT), os.path.getsize(OUT), 'B,', hi - lo + 1, 'pages')
