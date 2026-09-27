"""Exact Decimal/string reference for slot assignment, compared against production.

Independence: re-reads the RAW CSV strings and converts them to exact integer
nanoseconds by pure string/integer arithmetic. It never calls pd.to_datetime for its
own reference, never reads the stored timestamps to build it, and imports nothing
from the builder.

Reference chain (exact, no floating point anywhere):
    'Apr 30, 2019 07:51:15.381121000 CDT'
      -> drop the trailing timezone token, split on whitespace (tolerates 'Jun  5,')
      -> day  = days_from_civil(year, month, day)          proleptic Gregorian, integer
      -> ns   = day*86400*10**9 + (hh*3600+mm*60+ss)*10**9 + frac padded to 9 digits
      -> elapsed_ns = ns - ns_of_first_packet              (exact integer)
      -> Decimal(elapsed_ns) / Decimal(window_ns), ROUND_FLOOR   -> slot
      -> slot = min(slot, T - 1)

The DATE is included. An earlier version of this file used the time-of-day only, which
made elapsed times wrong for any capture whose rows carry more than one calendar date
and aborted the run; that defect is fixed here.

Production chain under test (no floating point either):
      stored int64 nanoseconds -> slot = min(ns // window_ns, T - 1)

Every included packet at every duration and every window is compared. Packets adjacent
to a boundary (one microsecond below, exactly on, one microsecond above) are counted
and reported separately. Any flow whose reference packet set differs in length from
production is RECORDED AND REPORTED, never allowed to abort the run.
"""
import io
import os
import sys
import time
import zipfile
from decimal import Decimal, getcontext

import numpy as np
import pandas as pd

getcontext().prec = 60

ARCHIVE = '/media/Data/Datasets/UTMobileNet2021/Raw/UTMobileNet2021.zip'
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REP = os.path.join(ROOT, 'reports')
WINDOWS_MS = [20, 30, 40, 50, 75, 100, 150, 200, 250]
DURATIONS = [3, 5, 8]
KEEP_APPS = {'facebook', 'gmail', 'google-drive', 'google-maps', 'hangout', 'instagram',
             'messenger', 'netflix', 'pinterest', 'reddit', 'spotify', 'twitter',
             'youtube'}
MIN_PACKETS = 50
TCP, UDP = 6, 17
COLS = ['frame.time', 'ip.src', 'ip.dst', 'ip.proto', 'sll.pkttype',
        'tcp.srcport', 'tcp.dstport', 'udp.srcport', 'udp.dstport']
MONTHS = {m: i for i, m in enumerate(
    ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], 1)}


def days_from_civil(y, m, d):
    """Proleptic Gregorian day number, pure integer arithmetic (Hinnant)."""
    y -= m <= 2
    era = (y if y >= 0 else y - 399) // 400
    yoe = y - era * 400
    doy = (153 * (m + (-3 if m > 2 else 9)) + 2) // 5 + d - 1
    doe = yoe * 365 + yoe // 4 - yoe // 100 + doy
    return era * 146097 + doe - 719468


def exact_ns(s):
    """'Apr 30, 2019 07:51:15.381121000 CDT' -> exact integer ns, string arithmetic."""
    parts = s.rsplit(' ', 1)[0].split()          # collapses repeated spaces
    mon, day, year, tm = parts[0], parts[1].rstrip(','), parts[2], parts[-1]
    hh, mm, rest = tm.split(':')
    sec, frac = rest.split('.') if '.' in rest else (rest, '')
    frac = (frac + '000000000')[:9]
    days = days_from_civil(int(year), MONTHS[mon], int(day))
    return ((days * 86400 + int(hh) * 3600 + int(mm) * 60 + int(sec)) * 10 ** 9
            + int(frac))


def canonical(a_ip, a_pt, b_ip, b_pt):
    if (a_ip, a_pt) <= (b_ip, b_pt):
        return a_ip, a_pt, b_ip, b_pt
    return b_ip, b_pt, a_ip, a_pt


def main():
    t0 = time.time()
    z = np.load(os.path.join(REP, 'b2_flow_packets_ns.npz'))
    t_all, off = z['t_ns'], z['offsets']
    if t_all.dtype != np.int64:
        print(f'  FATAL: production timestamps are {t_all.dtype}, expected int64 ns')
        return 1
    print(f'  production timestamp dtype: {t_all.dtype}  (exact integer nanoseconds)')
    idx = pd.read_parquet(os.path.join(REP, 'b2_flows_index.parquet'))
    key2fid = {(r.capture_id, r.ip_a, r.port_a, r.ip_b, r.port_b, r.proto): int(r.flow_id)
               for r in idx.itertuples()}
    meta = {D: pd.read_parquet(os.path.join(REP, f'meta_{D}s.parquet')) for D in DURATIONS}
    elig = {D: set(meta[D].flow_id.tolist()) for D in DURATIONS}

    zf = zipfile.ZipFile(ARCHIVE)
    sel = sorted(n for n in zf.namelist()
                 if n.endswith('.csv')
                 and os.path.basename(n).split('_')[0].lower() in KEEP_APPS)

    stats = {D: dict(pkts=0, slot_mismatch=0, keep_mismatch=0, len_mismatch=0,
                     below=0, on=0, above=0, cells=0) for D in DURATIONS}
    ns_mismatch = 0
    ns_maxdiff = 0
    len_mismatch_flows = []
    checked = 0
    for i, name in enumerate(sorted(sel), 1):
        base = os.path.basename(name)
        with zf.open(name) as fh:
            df = pd.read_csv(io.TextIOWrapper(fh, 'utf-8', 'replace'),
                             usecols=lambda c: c in COLS, low_memory=False)
        pk = pd.to_numeric(df['sll.pkttype'], errors='coerce')
        src = df['ip.src'].astype(str); dst = df['ip.dst'].astype(str)
        df = df[pk.isin([0, 4]) & ~src.str.startswith('127.')
                & ~dst.str.startswith('127.')].copy()
        if df.empty:
            continue
        proto = pd.to_numeric(df['ip.proto'], errors='coerce')
        src = df['ip.src'].astype(str); dst = df['ip.dst'].astype(str)
        is_tcp, is_udp = proto == TCP, proto == UDP
        sp = pd.Series(np.nan, index=df.index, dtype='float64')
        dp = pd.Series(np.nan, index=df.index, dtype='float64')
        sp = sp.mask(is_tcp, pd.to_numeric(df['tcp.srcport'], errors='coerce'))
        dp = dp.mask(is_tcp, pd.to_numeric(df['tcp.dstport'], errors='coerce'))
        sp = sp.mask(is_udp, pd.to_numeric(df['udp.srcport'], errors='coerce'))
        dp = dp.mask(is_udp, pd.to_numeric(df['udp.dstport'], errors='coerce'))
        ok = (is_tcp | is_udp) & sp.notna() & dp.notna() & (src != 'nan') & (dst != 'nan')
        if not ok.any():
            continue
        sub = pd.DataFrame({'src': src[ok].to_numpy(),
                            'sp': sp[ok].astype(np.int64).to_numpy(),
                            'dst': dst[ok].to_numpy(),
                            'dp': dp[ok].astype(np.int64).to_numpy(),
                            'proto': proto[ok].astype(np.int64).to_numpy(),
                            'ns': [exact_ns(x)
                                   for x in df.loc[ok, 'frame.time'].astype(str)]})
        keys = [canonical(a, int(b), c, int(d))
                for a, b, c, d in zip(sub.src, sub.sp, sub.dst, sub.dp)]
        sub['ip_a'] = [k[0] for k in keys]; sub['port_a'] = [k[1] for k in keys]
        sub['ip_b'] = [k[2] for k in keys]; sub['port_b'] = [k[3] for k in keys]

        for (ia, pa, ib, pb, pr), g in sub.groupby(['ip_a', 'port_a', 'ip_b', 'port_b',
                                                    'proto'], sort=False):
            if len(g) < MIN_PACKETS:
                continue
            fid = key2fid.get((base, ia, int(pa), ib, int(pb), int(pr)))
            if fid is None:
                continue
            g = g.sort_values('ns', kind='stable')
            ns = g.ns.to_numpy()
            el_ns = ns - ns[0]                                # exact integers
            a, b = off[fid], off[fid + 1]
            prod_ns = t_all[a:b]                              # exact integers, as stored
            checked += 1
            if len(prod_ns) != len(el_ns):
                len_mismatch_flows.append((base, fid, len(el_ns), len(prod_ns)))
                continue
            d = int(np.abs(prod_ns - el_ns).max()) if len(el_ns) else 0
            if d:
                ns_mismatch += 1
                ns_maxdiff = max(ns_maxdiff, d)
            for D in DURATIONS:
                if fid not in elig[D]:
                    continue
                D_ns = D * 10 ** 9
                keep_ref = el_ns <= D_ns
                keep_prod = prod_ns <= D_ns
                if not np.array_equal(keep_ref, keep_prod):
                    stats[D]['keep_mismatch'] += 1
                ref_in, prod_in = el_ns[keep_ref], prod_ns[keep_prod]
                if len(ref_in) != len(prod_in):
                    stats[D]['len_mismatch'] += 1
                    continue
                for w in WINDOWS_MS:
                    T = (D * 1000) // w + 1
                    w_ns = w * 10 ** 6
                    ref_slot = np.array(
                        [min(int((Decimal(int(x)) / Decimal(w_ns)).to_integral_value(
                            rounding='ROUND_FLOOR')), T - 1) for x in ref_in],
                        dtype=np.int64)
                    prod_slot = np.minimum(prod_in // w_ns, T - 1)
                    stats[D]['slot_mismatch'] += int((ref_slot != prod_slot).sum())
                    stats[D]['pkts'] += len(ref_in)
                    stats[D]['cells'] += 1
                    r = ref_in % w_ns
                    stats[D]['on'] += int((r == 0).sum())
                    stats[D]['below'] += int((r == w_ns - 1000).sum())
                    stats[D]['above'] += int((r == 1000).sum())
        if i % 400 == 0:
            print(f'  {i}/{len(sel)}  flows {checked:,}  {time.time() - t0:.0f}s',
                  flush=True)

    print(f'\n  flows cross-checked against the exact reference : {checked:,}')
    print(f'  flows with a different packet count than production : {len(len_mismatch_flows)}')
    for e in len_mismatch_flows[:10]:
        print(f'      {e[0]}  flow {e[1]}  reference {e[2]} vs production {e[3]}')
    print(f'  flows whose stored ns differ from the exact reference : {ns_mismatch}'
          f'   (max |diff| {ns_maxdiff} ns)')
    allok = (ns_mismatch == 0 and not len_mismatch_flows)
    for D in DURATIONS:
        s = stats[D]
        ok = (s['slot_mismatch'] == 0 and s['keep_mismatch'] == 0
              and s['len_mismatch'] == 0)
        allok &= ok
        print(f'\n  --- {D}s ---')
        print(f'      flow x window cells                  : {s["cells"]:,}')
        print(f'      packet slot assignments compared     : {s["pkts"]:,}')
        print(f'      SLOT MISMATCHES                      : {s["slot_mismatch"]}')
        print(f'      included-set mismatches              : {s["keep_mismatch"]}')
        print(f'      packet-count mismatches inside D     : {s["len_mismatch"]}')
        print(f'      packets exactly ON a boundary        : {s["on"]:,}')
        print(f'      packets one microsecond BELOW        : {s["below"]:,}')
        print(f'      packets one microsecond ABOVE        : {s["above"]:,}')
        print(f'      -> {"PASS" if ok else "FAIL"}')
    print(f'\n  EXACT REFERENCE AGREES EVERYWHERE: {allok}')
    return 0 if allok else 1


if __name__ == '__main__':
    sys.exit(main())
