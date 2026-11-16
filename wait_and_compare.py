#!/usr/bin/env python3
"""Wait for the full positional sweep to publish its table, then run the method comparison.

Completion is read from the sweep's own exit marker - the supervisor prints 'table: <path>' as
its last line - not from a process query: tasklist wedges on inherited pipe handles from here,
and a probe that times out and defaults to "alive" would park this waiter forever.  The table is
written in one loop, so the marker alone can still land ahead of the last flush; size stability
across 30 s closes that.

The finisher gets a grace window first so its numpy pass and this one do not fight the same
contended cores.

Usage:  python wait_and_compare.py       env: SWEEPLOG TABLE REPORT FINISHTIMEOUT WAITMAX
Log:    compare_full.log
"""
import os
import re
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def _p(path):
    """MSYS /tmp is %LOCALAPPDATA%\\Temp, which python does not resolve."""
    if path.startswith('/tmp/'):
        return os.path.join(tempfile.gettempdir(), path[5:])
    return path


SWEEPLOG = _p(os.environ.get('SWEEPLOG', '/tmp/pos_full.log'))
TABLE = _p(os.environ.get('TABLE', os.path.join(HERE, 'positional_full.txt')))
REPORT = _p(os.environ.get('REPORT', os.path.join(HERE, 'coverage_report.txt')))
LOG = _p(os.environ.get('CMPLOG', os.path.join(HERE, 'compare_full.log')))
FINISHTIMEOUT = int(os.environ.get('FINISHTIMEOUT', '1800'))    # 30 min for the finisher
WAITMAX = int(os.environ.get('WAITMAX', '9000'))                # 2.5 h for the sweep itself
MARKER = re.compile(r'^table: ', re.M)


def log(msg):
    with open(LOG, 'a') as f:
        f.write('%s %s\n' % (time.strftime('%H:%M:%S'), msg))
        f.flush()
    print(msg, flush=True)


def published():
    """True once the supervisor has printed its exit marker into the sweep log."""
    if not os.path.exists(SWEEPLOG):
        return False
    with open(SWEEPLOG, 'r', errors='replace') as f:
        return bool(MARKER.search(f.read()))


def stable(path, gap=30):
    s1 = os.path.getsize(path)
    time.sleep(gap)
    return os.path.getsize(path) == s1 and s1 > 10 ** 6


def main():
    t0 = time.time()
    log('waiting for exit marker in %s and a stable %s' % (SWEEPLOG, TABLE))
    ready = False
    while time.time() - t0 < WAITMAX:
        if published() and os.path.exists(TABLE) and stable(TABLE):
            ready = True
            break
        time.sleep(20)
    if not ready:
        log('TIMEOUT after %d s - nothing ran' % WAITMAX)
        return 1

    with open(TABLE) as f:
        nrow = sum(1 for line in f if line[:1] not in '#\n')
    log('table ready: %d bytes, %d data rows (%.4f of grid)'
        % (os.path.getsize(TABLE), nrow, nrow / 2064385.0))

    t1 = time.time()
    while time.time() - t1 < FINISHTIMEOUT:
        if os.path.exists(REPORT):
            log('finisher published %s' % REPORT)
            break
        time.sleep(20)
    else:
        log('finisher did not publish within %ds - running the comparison anyway'
            % FINISHTIMEOUT)

    env = dict(os.environ, TABLE=TABLE)
    try:
        r = subprocess.run([sys.executable, os.path.join(HERE, 'compare_methods.py')],
                           env=env, capture_output=True, text=True, timeout=3600)
    except Exception as e:
        log('compare_methods failed to run: %r' % (e,))
        return 2
    with open(LOG, 'a') as f:
        f.write(r.stdout)
        if r.stderr:
            f.write('--- stderr ---\n' + r.stderr)
    log('compare_methods exit %d -> method_compare.txt, target_methods.txt' % r.returncode)
    return 0 if r.returncode == 0 else 2


if __name__ == '__main__':
    sys.exit(main())
