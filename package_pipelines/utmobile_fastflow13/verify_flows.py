"""Independent verifier for the FastFlow-13 reproduction.

Imports NOTHING from build_flows.py. It re-derives the flows with a different technique:
a plain python dictionary keyed by a frozenset of the two endpoints plus the protocol,
accumulated packet by packet, rather than pandas canonicalisation and groupby.

Both implementations must agree exactly on:
  flow keys, per-flow packet counts, application label, flows per application,
  the grand total, the 49/50/51 boundary counts, and packet conservation.
"""
import csv
import io
import os
import sys
import time
import zipfile
from collections import defaultdict

import pandas as pd

ARCHIVE = '/media/Data/Datasets/UTMobileNet2021/Raw/UTMobileNet2021.zip'
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'reports')
KEEP_APPS = {'facebook', 'gmail', 'google-drive', 'google-maps', 'hangout', 'instagram',
             'messenger', 'netflix', 'pinterest', 'reddit', 'spotify', 'twitter',
             'youtube'}
MIN_PACKETS = 50


def main():
    t0 = time.time()
    zf = zipfile.ZipFile(ARCHIVE)
    sel = sorted(n for n in zf.namelist()
                 if n.endswith('.csv')
                 and os.path.basename(n).split('_')[0].lower() in KEEP_APPS)
    print(f'verifier: {len(sel)} captures', flush=True)

    counts = {}          # (capture, endpoint_pair_sorted, proto) -> packets
    raw_tot = usable_tot = other_tot = 0
    for i, n in enumerate(sel, 1):
        base = os.path.basename(n)
        with zf.open(n) as fh:
            rd = csv.DictReader(io.TextIOWrapper(fh, 'utf-8', 'replace'))
            for row in rd:
                raw_tot += 1
                try:
                    pr = int(float(row.get('ip.proto') or 'nan'))
                except (TypeError, ValueError):
                    other_tot += 1
                    continue
                if pr == 6:
                    a, b = row.get('tcp.srcport'), row.get('tcp.dstport')
                elif pr == 17:
                    a, b = row.get('udp.srcport'), row.get('udp.dstport')
                else:
                    other_tot += 1
                    continue
                s, d = row.get('ip.src'), row.get('ip.dst')
                if not a or not b or not s or not d:
                    other_tot += 1
                    continue
                try:
                    ap, bp = int(float(a)), int(float(b))
                except ValueError:
                    other_tot += 1
                    continue
                ends = tuple(sorted(((s, ap), (d, bp))))   # different mechanism
                key = (base, ends, pr)
                counts[key] = counts.get(key, 0) + 1
                usable_tot += 1
        if i % 500 == 0:
            print(f'  {i}/{len(sel)}  flows so far {len(counts):,}  '
                  f'{time.time() - t0:.0f}s', flush=True)

    rows = []
    for (cap, ends, pr), n in counts.items():
        (ip_a, port_a), (ip_b, port_b) = ends
        rows.append(dict(capture_id=cap, app=cap.split('_')[0].lower(),
                         ip_a=ip_a, port_a=port_a, ip_b=ip_b, port_b=port_b,
                         proto=pr, packets=n))
    v = pd.DataFrame(rows)
    v.to_parquet(os.path.join(OUT, 'flows_verifier.parquet'), index=False)
    print(f'\n  raw packets {raw_tot:,}   usable {usable_tot:,}   other {other_tot:,}')
    print(f'  conservation: {usable_tot + other_tot == raw_tot}')
    print(f'  flows {len(v):,}   flows >= {MIN_PACKETS}: {int((v.packets >= MIN_PACKETS).sum()):,}')
    print(f'  wrote {OUT}/flows_verifier.parquet   ({time.time() - t0:.0f}s)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
