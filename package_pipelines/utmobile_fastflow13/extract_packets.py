"""Extract per-packet arrays for the 12,459 gate-verified biflows.

Pipeline, in the exact order given:
  1. B2 packet filter, applied BEFORE any grouping, verbatim:
         pk   = pd.to_numeric(df['sll.pkttype'], errors='coerce')
         src  = df['ip.src'].astype(str);  dst = df['ip.dst'].astype(str)
         keep = pk.isin([0, 4]) & ~src.str.startswith('127.') & ~dst.str.startswith('127.')
  2. inside each capture separately, never across captures
  3. canonical bidirectional 5-tuple, IP and port swapped together
  4. DNS and other surviving helper flows kept, label inherited from the capture
  5. keep only the 13 FastFlow applications
  6. dropbox / hulu / pandora / skype excluded
  7. packet_count >= 50 over the COMPLETE filtered biflow
  8. no other filter of any kind

Per surviving flow it stores the ordered packet arrays needed to build any duration:
  t_ns  EXACT int64 nanoseconds since the flow's FIRST packet (no float anywhere)
  size  payload: tcp.len for TCP, udp.length for UDP (the canonical definition)
  fwd   True when the packet's source IP equals the flow's first packet source IP,
        i.e. FORWARD is relative to the first packet, never client/server

Read-only with respect to everything else. Writes one npz plus one index parquet.
"""
import io
import os
import sys
import time
import zipfile

import numpy as np
import pandas as pd

ARCHIVE = '/media/Data/Datasets/UTMobileNet2021/Raw/UTMobileNet2021.zip'
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'reports')

KEEP_APPS = {'facebook', 'gmail', 'google-drive', 'google-maps', 'hangout', 'instagram',
             'messenger', 'netflix', 'pinterest', 'reddit', 'spotify', 'twitter',
             'youtube'}
EXCLUDED = {'dropbox', 'hulu', 'pandora', 'skype'}
MIN_PACKETS = 50
TCP, UDP = 6, 17
COLS = ['frame.time', 'ip.src', 'ip.dst', 'ip.proto', 'sll.pkttype',
        'tcp.srcport', 'tcp.dstport', 'tcp.len',
        'udp.srcport', 'udp.dstport', 'udp.length']


def canonical(a_ip, a_pt, b_ip, b_pt):
    if (a_ip, a_pt) <= (b_ip, b_pt):
        return a_ip, a_pt, b_ip, b_pt
    return b_ip, b_pt, a_ip, a_pt


def main():
    t0 = time.time()
    zf = zipfile.ZipFile(ARCHIVE)
    sel = sorted(n for n in zf.namelist()
                 if n.endswith('.csv')
                 and os.path.basename(n).split('_')[0].lower() in KEEP_APPS)
    print(f'captures: {len(sel)}   excluded apps: {sorted(EXCLUDED)}', flush=True)

    index, T, S, F = [], [], [], []
    pk_raw = pk_kept = 0
    for i, name in enumerate(sorted(sel), 1):
        base = os.path.basename(name)
        app = base.split('_')[0].lower()
        partition = name.split('/')[0]
        with zf.open(name) as fh:
            df = pd.read_csv(io.TextIOWrapper(fh, 'utf-8', 'replace'),
                             usecols=lambda c: c in COLS, low_memory=False)
        pk_raw += len(df)

        # ---- step 1: the B2 filter, verbatim, before grouping --------------------
        pk = pd.to_numeric(df['sll.pkttype'], errors='coerce')
        src = df['ip.src'].astype(str)
        dst = df['ip.dst'].astype(str)
        keep = pk.isin([0, 4]) & ~src.str.startswith('127.') & ~dst.str.startswith('127.')
        df = df[keep].copy()
        pk_kept += len(df)
        if df.empty:
            continue

        proto = pd.to_numeric(df['ip.proto'], errors='coerce')
        src = df['ip.src'].astype(str)
        dst = df['ip.dst'].astype(str)
        is_tcp, is_udp = proto == TCP, proto == UDP
        sp = pd.Series(np.nan, index=df.index, dtype='float64')
        dp = pd.Series(np.nan, index=df.index, dtype='float64')
        pay = pd.Series(np.nan, index=df.index, dtype='float64')
        sp = sp.mask(is_tcp, pd.to_numeric(df['tcp.srcport'], errors='coerce'))
        dp = dp.mask(is_tcp, pd.to_numeric(df['tcp.dstport'], errors='coerce'))
        pay = pay.mask(is_tcp, pd.to_numeric(df['tcp.len'], errors='coerce'))
        sp = sp.mask(is_udp, pd.to_numeric(df['udp.srcport'], errors='coerce'))
        dp = dp.mask(is_udp, pd.to_numeric(df['udp.dstport'], errors='coerce'))
        pay = pay.mask(is_udp, pd.to_numeric(df['udp.length'], errors='coerce'))

        ok = (is_tcp | is_udp) & sp.notna() & dp.notna() & (src != 'nan') & (dst != 'nan')
        if not ok.any():
            continue
        tstr = df.loc[ok, 'frame.time'].astype(str).str.rsplit(' ', n=1).str[0]
        ts = pd.to_datetime(tstr, format='%b %d, %Y %H:%M:%S.%f', errors='coerce')

        u = pd.DataFrame({'src': src[ok].to_numpy(),
                          'sp': sp[ok].astype(np.int64).to_numpy(),
                          'dst': dst[ok].to_numpy(),
                          'dp': dp[ok].astype(np.int64).to_numpy(),
                          'proto': proto[ok].astype(np.int64).to_numpy(),
                          'pay': pay[ok].fillna(0.0).to_numpy(),
                          'ts': ts.to_numpy()})
        keys = [canonical(a, int(b), c, int(d))
                for a, b, c, d in zip(u.src, u.sp, u.dst, u.dp)]
        u['ip_a'] = [k[0] for k in keys]; u['port_a'] = [k[1] for k in keys]
        u['ip_b'] = [k[2] for k in keys]; u['port_b'] = [k[3] for k in keys]

        for (ia, pa, ib, pb, pr), g in u.groupby(['ip_a', 'port_a', 'ip_b', 'port_b',
                                                  'proto'], sort=False):
            if len(g) < MIN_PACKETS:                 # step 7, inclusive >= 50
                continue
            g = g.sort_values('ts', kind='stable')
            ns = g.ts.to_numpy().astype('datetime64[ns]').astype(np.int64)
            t_ns = (ns - ns[0]).astype(np.int64)          # exact integer, never float
            fwd = (g.src.to_numpy() == g.src.to_numpy()[0])   # forward = first packet's src
            index.append(dict(flow_id=len(index), capture_id=base, partition=partition,
                              app=app, ip_a=ia, port_a=int(pa), ip_b=ib, port_b=int(pb),
                              proto=int(pr), packets=int(len(g)),
                              duration_ns=int(t_ns[-1]),
                              duration_s=float(t_ns[-1]) / 1e9,
                              payload_total=float(g.pay.to_numpy().sum()),
                              is_dns=bool(pa == 53 or pb == 53)))
            T.append(t_ns)
            S.append(g.pay.to_numpy().astype(np.float64))
            F.append(fwd.astype(np.bool_))
        if i % 500 == 0:
            print(f'  {i}/{len(sel)}  flows {len(index):,}  {time.time() - t0:.0f}s', flush=True)

    idx = pd.DataFrame(index)
    idx.to_parquet(os.path.join(OUT, 'b2_flows_index.parquet'), index=False)
    off = np.cumsum([0] + [len(x) for x in T])
    t_ns_all = np.concatenate(T).astype(np.int64)
    assert t_ns_all.dtype == np.int64, 'timestamps must be stored as int64 nanoseconds'
    np.savez_compressed(os.path.join(OUT, 'b2_flow_packets_ns.npz'),
                        t_ns=t_ns_all, size=np.concatenate(S),
                        fwd=np.concatenate(F), offsets=off)
    print(f'\n  packets raw {pk_raw:,} -> after B2 filter {pk_kept:,} '
          f'(dropped {pk_raw - pk_kept:,})')
    print(f'  flows >= {MIN_PACKETS}: {len(idx):,}   packets in them: {int(idx.packets.sum()):,}')
    print(f'  per app:\n{idx.app.value_counts().sort_index().to_string()}')
    print(f'  wrote b2_flows_index.parquet and b2_flow_packets_ns.npz (int64 nanoseconds)  ({time.time() - t0:.0f}s)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
