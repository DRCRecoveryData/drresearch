#!/usr/bin/env python3
"""Re-check submit/sha256.txt against the files on disk, in both directions.

A hash index is only worth as much as the audit that it is complete and current, so this walks the
package, hashes every file it finds, and compares that set with the index: size, digest, and the
two difference directions (listed-but-changed, present-but-unlisted).

Usage:  python verify_submission_index.py [PACKAGE_DIR]
"""
import hashlib
import os
import sys

CHUNK = 8 * 1024 * 1024


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(CHUNK)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def rel(root, path):
    return os.path.relpath(path, root).replace(os.sep, '/')


def main():
    root = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else 'submit')
    index = os.path.join(root, 'sha256.txt')
    if not os.path.isdir(root) or not os.path.exists(index):
        raise SystemExit('need a package dir with sha256.txt: %s' % root)

    listed = {}
    with open(index) as f:
        for line in f:
            line = line.rstrip('\n')
            if not line or line.startswith('#'):
                continue
            parts = line.split('  ', 2)
            if len(parts) != 3:
                raise SystemExit('malformed index line: %r' % line)
            listed[parts[2]] = (parts[0], int(parts[1]))

    on_disk = {rel(root, os.path.join(dp, fn)) for dp, _, fns in os.walk(root) for fn in fns}
    on_disk -= {'sha256.txt'}

    missing = sorted(on_disk - set(listed))
    stale = sorted(set(listed) - on_disk)
    bad = []
    for r, (dig, size) in sorted(listed.items()):
        path = os.path.join(root, r.replace('/', os.sep))
        if not os.path.exists(path):
            continue
        if os.path.getsize(path) != size:
            bad.append('%s size %d != listed %d' % (r, os.path.getsize(path), size))
        elif sha256_of(path) != dig:
            bad.append('%s digest mismatch' % r)

    print('package          : %s' % root)
    print('index entries    : %d' % len(listed))
    print('files on disk    : %d (excluding sha256.txt)' % len(on_disk))
    print('entries verified : %d / %d' % (len(listed) - len(stale) - len(bad), len(listed)))
    print('present, unlisted: %d %s' % (len(missing), missing))
    print('listed, absent   : %d %s' % (len(stale), stale))
    print('bad size/digest  : %d %s' % (len(bad), bad))
    ok = not missing and not stale and not bad
    print('verdict          : %s' % ('index is complete and current' if ok else 'INDEX DEFECT'))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
