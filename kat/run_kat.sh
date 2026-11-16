#!/bin/sh
# Known-answer test for the per-block XOR fix.  Run from the kat/ directory.
set -e
cd "$(dirname "$0")"
export PERL5LIB="$PWD/shim"
rm -f *.dump *.xor A.dump B.dump C.dump D.dump

python make_synth_perblock.py static
python make_synth_perblock.py perblock
python - <<'EOF'
ks = [int(l.split()[1], 16) for l in open('synth_static.keys.txt')]
open('synth_static.dump.xor', 'wb').write(bytes([ks[0]]) * 147456)
kp = [int(l.split()[1], 16) for l in open('synth.keys.txt')]
open('synth.dump.xor', 'wb').write(b''.join(bytes([k]) * 147456 for k in kp))
EOF

echo "== A static key, stock extractor =="
perl stock/dumpextractrelevant.pl synth_static.dump A.dump synth_static.pattern.xml synth_static_Chip.txt | grep '^Found: [0-9]'
echo "== B static key, patched extractor (regression: must equal A) =="
perl ../dumpextractrelevant.pl synth_static.dump B.dump synth_static.pattern.xml synth_static_Chip.txt | grep '^Found: [0-9]'
cmp A.dump B.dump && echo "A == B: patch preserves the static model"

echo "== C dynamic per-block key, stock extractor (README limitation) =="
perl stock/dumpextractrelevant.pl synth.dump C.dump synth.pattern.xml synth_Chip.txt | grep -E '^Found: [0-9]|XOR key is wrong' || true
echo "== D dynamic per-block key, patched + dump-sized key =="
python ../xorsolve_perblock.py synth.dump synth_Chip.txt synth.dump.xor
perl ../dumpextractrelevant.pl synth.dump D.dump synth.pattern.xml synth_Chip.txt | grep '^Found: [0-9]'
python - <<'EOF'
kp = [int(l.split()[1], 16) for l in open('synth.keys.txt')]
ct = open('synth.dump', 'rb').read()
B = 147456
plain = bytes(c ^ kp[i // B] for i, c in enumerate(ct))
out = open('D.dump', 'rb').read()
assert out == plain, 'descrambled content does not match ground truth'
print('D == true plaintext: per-block key recovered and applied correctly')
EOF

echo "== E dynamic per-block key, ONE command, no key file supplied (the README failure) =="
rm -f synth.dump.xor synth.dump.perblock.xor E.dump
perl ../dumpextractrelevant.pl synth.dump E.dump synth.pattern.xml synth_Chip.txt 2>&1 |
    grep -E '^Found: [0-9]|Trying a key byte|Solved per-block'
python - <<'EOF'
kp = [int(l.split()[1], 16) for l in open('synth.keys.txt')]
B = 147456
ct = open('synth.dump', 'rb').read()
plain = bytes(c ^ kp[i // B] for i, c in enumerate(ct))
out = open('E.dump', 'rb').read()
assert out == plain, 'automatic per-block recovery does not match ground truth'
print('E == true plaintext: the tool recovered the per-block XOR by itself')
EOF

echo "== F anchor classifier: bit-rot tolerance + magic-free structural recovery =="
# Guards anchor_coverage.py in both directions: nothing may be 'recovered' that is
# not damaged (clean fixture) and nothing damaged may stay lost (this control).
python make_corrupt.py | grep -E 'tolerance|exact-magic|magic-free|hard-missing'
BLKSZ=147456 ECCSTART=0 ECCEND=4599 INC=9 XORKEY=synth_static.dump.xor TOL=3 \
    MISSOUT=/tmp/kat_clean_missing.txt HARDOUT=/tmp/kat_clean_hard.txt \
    python ../anchor_coverage.py synth_static.dump | grep -E 'structural scan adds|hard-missing \(no record'

echo "== G hard-anchor pass: records the audit calls unrecoverable =="
# make_hard.py damages 100 records so anchor_coverage.py cannot self-check them and
# erases 20 sectors entirely; hard_probe.py must return the first class and refuse the
# second. A pass that finds nothing on the real dump is only meaningful against this.
python make_hard.py 2>&1 | grep -E 'anchor_coverage hard-missing|recovered|grade|unrecoverable|wrongly'
