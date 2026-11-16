#!/usr/bin/env python3
"""Write submit/sha256.txt - one line per file, sorted by path, sizes included.

The extraction dump is 13.6 GB, so it is hashed in 8 MB reads rather than slurped. Paths are
relative to the package root so the index survives being moved; the index itself is excluded.

Usage:  python make_submission_index.py [PACKAGE_DIR]
Writes: <PACKAGE_DIR>/sha256.txt
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


def main():
    root = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else 'submit')
    if not os.path.isdir(root):
        raise SystemExit('not a directory: %s' % root)
    index = os.path.join(root, 'sha256.txt')
    files = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for name in sorted(filenames):
            full = os.path.join(dirpath, name)
            if os.path.abspath(full) == os.path.abspath(index):
                continue
            files.append(os.path.relpath(full, root).replace('\\', '/'))
    if not files:
        raise SystemExit('package %s has no files' % root)

    lines = ['# sha256  bytes  path (relative to this folder)']
    for rel in sorted(files):
        full = os.path.join(root, rel.replace('/', os.sep))
        size = os.path.getsize(full)
        print('%s  %d B' % (rel, size), flush=True)
        lines.append('%s  %d  %s' % (sha256_of(full), size, rel))
    with open(index, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('wrote %s (%d files)' % (index, len(files)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
