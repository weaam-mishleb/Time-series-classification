"""FastFlow-13 reproduction, pass 1: build bidirectional flows from the RAW capture CSVs.

Source: the official UT Austin Box archive UTMobileNet2021.zip
        md5 a3ccebda6daa4201cf35eba945bee7a5

Flow definition, per the brief:
  * one flow = one full 5-tuple (src ip, src port, dst ip, dst port, protocol)
    WITHIN one capture. The capture id is part of the key, so two captures can never
    merge.
  * bidirectional: the two endpoints are canonicalised as a sorted pair of
    (ip, port) tuples, so address and port always swap together.
  * TCP uses tcp.srcport/tcp.dstport, UDP uses udp.srcport/udp.dstport.
  * no idle timeout, no IP-pair-only grouping, no merging of distinct flows.

Labels: every flow inherits its capture's application label, including DNS, background
and helper traffic. Nothing is removed; DNS is only counted and reported.

Writes one parquet of per-flow rows. Reads nothing but the archive. Touches no other
directory.
"""
import io
import os
import sys
import time
import zipfile
from collections import Counter

import numpy as np
import pandas as pd

ARCHIVE = '/media/Data/Datasets/UTMobileNet2021/Raw/UTMobileNet2021.zip'
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'reports')

KEEP_APPS = {'facebook', 'gmail', 'google-drive', 'google-maps', 'hangout', 'instagram',
             'messenger', 'netflix', 'pinterest', 'reddit', 'spotify', 'twitter',
             'youtube'}
MIN_PACKETS = 50            # inclusive: a flow with exactly 50 packets is kept
TCP, UDP = 6, 17
COLS = ['ip.src', 'ip.dst', 'ip.proto', 'tcp.srcport', 'tcp.dstport',
        'udp.srcport', 'udp.dstport']


def canonical(a_ip, a_pt, b_ip, b_pt):
    """Order the two endpoints so both directions of one connection share a key."""
    if (a_ip, a_pt) <= (b_ip, b_pt):
        return a_ip, a_pt, b_ip, b_pt
    return b_ip, b_pt, a_ip, a_pt


def flows_of_capture(df, capture_id, app, partition):
    """Return per-flow rows plus a packet-accounting dict for one capture."""
    n_raw = len(df)
    proto = pd.to_numeric(df['ip.proto'], errors='coerce')
    src = df['ip.src'].astype(str)
    dst = df['ip.dst'].astype(str)

    is_tcp = proto == TCP
    is_udp = proto == UDP
    sp = pd.Series(np.nan, index=df.index, dtype='float64')
    dp = pd.Series(np.nan, index=df.index, dtype='float64')
    if 'tcp.srcport' in df.columns:
        sp = sp.mask(is_tcp, pd.to_numeric(df['tcp.srcport'], errors='coerce'))
        dp = dp.mask(is_tcp, pd.to_numeric(df['tcp.dstport'], errors='coerce'))
    if 'udp.srcport' in df.columns:
        sp = sp.mask(is_udp, pd.to_numeric(df['udp.srcport'], errors='coerce'))
        dp = dp.mask(is_udp, pd.to_numeric(df['udp.dstport'], errors='coerce'))

    usable = (is_tcp | is_udp) & sp.notna() & dp.notna() & \
             (src != 'nan') & (dst != 'nan')
    acct = dict(packets_raw=n_raw,
                packets_non_tcp_udp=int((~(is_tcp | is_udp)).sum()),
                packets_missing_port_or_ip=int(((is_tcp | is_udp) & ~usable).sum()),
                packets_usable=int(usable.sum()))

    if not usable.any():
        return [], acct

    u = pd.DataFrame({'src': src[usable].to_numpy(),
                      'sp': sp[usable].astype(np.int64).to_numpy(),
                      'dst': dst[usable].to_numpy(),
                      'dp': dp[usable].astype(np.int64).to_numpy(),
                      'proto': proto[usable].astype(np.int64).to_numpy()})

    keys = [canonical(a, int(b), c, int(d))
            for a, b, c, d in zip(u.src, u.sp, u.dst, u.dp)]
    u['ip_a'] = [k[0] for k in keys]
    u['port_a'] = [k[1] for k in keys]
    u['ip_b'] = [k[2] for k in keys]
    u['port_b'] = [k[3] for k in keys]

    g = u.groupby(['ip_a', 'port_a', 'ip_b', 'port_b', 'proto'], sort=False).size()
    rows = []
    for (ip_a, port_a, ip_b, port_b, pr), n in g.items():
        rows.append(dict(capture_id=capture_id, partition=partition, app=app,
                         ip_a=ip_a, port_a=int(port_a), ip_b=ip_b, port_b=int(port_b),
                         proto=int(pr), packets=int(n),
                         is_dns=bool(port_a == 53 or port_b == 53),
                         is_mdns=bool(port_a == 5353 or port_b == 5353)))
    assert sum(r['packets'] for r in rows) == acct['packets_usable'], \
        f'packet conservation failed in {capture_id}'
    return rows, acct


def main():
    t0 = time.time()
    zf = zipfile.ZipFile(ARCHIVE)
    names = [n for n in zf.namelist() if n.endswith('.csv')]
    sel = [n for n in names if os.path.basename(n).split('_')[0].lower() in KEEP_APPS]
    print(f'archive: {ARCHIVE}')
    print(f'  csv entries: {len(names)}   selected for the 13 apps: {len(sel)}', flush=True)

    all_rows, acct_rows = [], []
    for i, n in enumerate(sorted(sel), 1):
        partition = n.split('/')[0]
        base = os.path.basename(n)
        app = base.split('_')[0].lower()
        with zf.open(n) as fh:
            df = pd.read_csv(io.TextIOWrapper(fh, 'utf-8', 'replace'),
                             usecols=lambda c: c in COLS, low_memory=False)
        rows, acct = flows_of_capture(df, base, app, partition)
        all_rows.extend(rows)
        acct.update(capture_id=base, app=app, partition=partition, n_flows=len(rows))
        acct_rows.append(acct)
        if i % 250 == 0:
            print(f'  {i}/{len(sel)} captures   flows so far {len(all_rows):,}   '
                  f'{time.time() - t0:.0f}s', flush=True)

    f = pd.DataFrame(all_rows)
    a = pd.DataFrame(acct_rows)
    os.makedirs(OUT, exist_ok=True)
    f.to_parquet(os.path.join(OUT, 'flows_all.parquet'), index=False)
    a.to_csv(os.path.join(OUT, 'capture_accounting.csv'), index=False)

    print(f'\n  captures read      : {len(a)}')
    print(f'  packets raw        : {a.packets_raw.sum():,}')
    print(f'  packets usable     : {a.packets_usable.sum():,}')
    print(f'  non TCP/UDP        : {a.packets_non_tcp_udp.sum():,}')
    print(f'  missing port or ip : {a.packets_missing_port_or_ip.sum():,}')
    print(f'  conservation       : '
          f'{a.packets_usable.sum() + a.packets_non_tcp_udp.sum() + a.packets_missing_port_or_ip.sum() == a.packets_raw.sum()}')
    print(f'  flows (all)        : {len(f):,}')
    print(f'  flow packets total : {f.packets.sum():,}  == usable: {f.packets.sum() == a.packets_usable.sum()}')
    keep = f[f.packets >= MIN_PACKETS]
    print(f'  flows with >= {MIN_PACKETS}   : {len(keep):,}')
    print(f'\n  wrote {OUT}/flows_all.parquet and capture_accounting.csv'
          f'   ({time.time() - t0:.0f}s)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
