"""Renders one extracted page the way dumphex.pl annotates it, as a PNG.

The author's HTML viewer emits one div per byte, so a full page is 18,432 divs
and the browser spends longer laying it out than the screenshot budget allows.
This draws the same field map with GDI-free PIL instead, at 1 px per byte.
"""
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

DUMP = os.environ.get('DUMP', '../submit/upload_ps2251-11_bics4(18432p).dump')
CASE = os.environ.get('CASE', '../submit/geometry_ps2251-11_bics4.case')
PAGE = int(os.environ.get('PAGE', '0'))
OUT = os.environ.get('OUT', 'ecc_view_page.png')

PGSZ = 18432
CH = 1144
DATA, SA, ECC = 1024, 6, 114
REST = PGSZ - 16 * CH  # 128 B page remainder

# field map straight out of the case file, same three colours dumphex.pl uses
COL = {'DATA': (160, 255, 160), 'SA': (255, 160, 160), 'ECC': (255, 255, 160),
       'REST': (210, 210, 210), 'BG': (245, 245, 243), 'INK': (35, 38, 41),
       'DIM': (120, 124, 128), 'LINE': (200, 200, 198)}


def fields_from_case(path):
    """Parse the Record StructureDefinition offsets so the picture cannot drift
    from the file the viewer itself reads."""
    import re
    txt = open(path, 'rb').read().replace(b'\x00', b'').decode('utf-8', 'replace')
    out = []
    for name, a, b in re.findall(
            r'StructureDefinitionName="(DATA|SA|ECC)"\s+StartAddress="(\d+)"\s+StopAddress="(\d+)"',
            txt):
        out.append((name, int(a), int(b)))
    return out


def font(sz):
    for p in ('C:/Windows/Fonts/consola.ttf', 'C:/Windows/Fonts/arial.ttf'):
        try:
            return ImageFont.truetype(p, sz)
        except OSError:
            continue
    return ImageFont.load_default()


f_h, f_v, f_s = font(21), font(15), font(13)

page = None
with open(DUMP, 'rb') as fh:
    fh.seek(PAGE * PGSZ)
    page = fh.read(PGSZ)
assert len(page) == PGSZ, f'short read at page {PAGE}: {len(page)} B'

# per-byte class array, filled from the case so offsets are not assumed
cls = np.full(PGSZ, 'REST', dtype='<U4')
for name, a, b in fields_from_case(CASE):
    cls[a:b + 1] = name
assert (cls == 'DATA').sum() == 16 * DATA, 'case DATA bytes disagree with geometry'
assert (cls == 'ECC').sum() == 16 * ECC, 'case ECC bytes disagree with geometry'

W = CH  # one pixel per byte across a chunk, 16 rows stacked
pad, head, rows_h, zoom_h = 22, 96, 16 * 26 + 40, 250
H = pad * 2 + head + rows_h + zoom_h
img = Image.new('RGB', (W + pad * 2, H), COL['BG'])
d = ImageDraw.Draw(img)

d.text((pad, 20), 'PS2251-11 / BiCS4 - one extracted page as dumphex.pl annotates it',
       font=f_h, fill=COL['INK'])
d.text((pad, 50), f'page {PAGE} of upload_ps2251-11_bics4(18432p).dump - 18,432 B = '
                  f'16 x (1,024 DATA + 6 SA + 114 ECC) + {REST} B remainder, 1 px per byte',
       font=f_s, fill=COL['DIM'])
y = head

# the 16 chunk rows: field colours under the real byte values, so the ECC tail
# reads as noise and the DATA body reads as text
for i in range(16):
    base = i * CH
    row = np.array([COL[c] for c in cls[base:base + CH]], dtype=np.float64)
    shade = (0.35 + 0.65 * np.frombuffer(page[base:base + CH], np.uint8) / 255.0)[:, None]
    strip = Image.fromarray((row * shade).astype(np.uint8)[None, :, :], 'RGB')
    img.paste(strip.resize((CH, 24), Image.NEAREST), (pad, y + i * 26 + 4))
d.text((pad, y + 16 * 26 + 6), 'chunk 0 at the left edge: green DATA, 6 B red SA, '
                               '114 B yellow ECC - the yellow band is the LDPC parity',
       font=f_s, fill=COL['DIM'])
y += rows_h

# zoom: chunk 0 spelled out the way the HTML tooltips spell it
d.line([(pad, y - 6), (W + pad, y - 6)], fill=COL['LINE'])
data, sa, ecc = page[:DATA], page[DATA:DATA + SA], page[DATA + SA:DATA + SA + ECC]
asc = ''.join(chr(b) if 32 <= b < 127 else '.' for b in data)
d.text((pad, y), f'DATA  [0:1024]   {asc[:118]}', font=f_v, fill=COL['INK'])
d.text((pad, y + 24), f'      {asc[118:236]}', font=f_v, fill=COL['INK'])
d.text((pad, y + 48), 'SA    [1024:1030] ' + ' '.join(f'{b:02X}' for b in sa),
       font=f_v, fill=(180, 60, 60))
d.text((pad, y + 72), 'ECC   [1030:1144] ' + ' '.join(f'{b:02X}' for b in ecc[:40]),
       font=f_v, fill=(150, 130, 20))
d.text((pad, y + 96), '                   ' + ' '.join(f'{b:02X}' for b in ecc[40:80]),
       font=f_v, fill=(150, 130, 20))
d.text((pad, y + 120), '                   ' + ' '.join(f'{b:02X}' for b in ecc[80:]),
       font=f_v, fill=(150, 130, 20))
ent = len(set(ecc)) / len(ecc)
d.text((pad, y + 150), f'ECC byte diversity in this field: {ent:.3f} '
                       f'({len(set(ecc))} distinct of {len(ecc)}) - parity, not plaintext',
       font=f_s, fill=COL['DIM'])
d.text((pad, y + 174), f'page-wide: DATA {(cls == "DATA").sum()} B, SA {(cls == "SA").sum()} B, '
                       f'ECC {(cls == "ECC").sum()} B, remainder {(cls == "REST").sum()} B',
       font=f_s, fill=COL['DIM'])

img.save(OUT)
print('wrote', os.path.abspath(OUT), img.size)
