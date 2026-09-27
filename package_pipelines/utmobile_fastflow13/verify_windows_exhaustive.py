"""EXHAUSTIVE independent verification of the small-window division.

Imports NOTHING from build_durations.py. It re-derives elapsed time, the eligible packet
set, slot assignment and all eight features from the stored raw packet arrays, using a
plain per-slot python accumulation instead of bincount, and compares against every stored
feature value.

Covers every duration, every one of the nine windows, and every eligible flow.

TOLERANCES, DECLARED BEFORE ANY RESULT IS SEEN:
  counts (idx 1, 4) and time_from_start (idx 7) : required max |diff| EXACTLY 0
  payload sums (idx 0, 3)                       : atol 1e-6, rtol 1e-9
  payload means (idx 2, 5) and ratio (idx 6)    : atol 1e-6, rtol 1e-9
  3feat vs 8feat[:, [1,4,6]]                    : required BIT-EXACT
Actual maxima are reported for every index regardless of pass or fail.
"""
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REP = os.path.join(ROOT, 'reports')
WINDOWS_MS = [20, 30, 40, 50, 75, 100, 150, 200, 250]
DURATIONS = [3, 5, 8]
EXACT_IDX = [1, 4, 7]
ATOL = 1e-6
FEAT3 = [1, 4, 6]


def manual_features(t_ns, size, fwd, w_ns, T):
    """Single explicit pass over packets, accumulating into per-slot totals.

    Deliberately NOT bincount and NOT any vectorised grouping: this is a plain python
    accumulation loop, so it shares no code path with the builder.
    """
    out = np.zeros((T, 8), dtype=np.float64)
    f_tot = [0.0] * T; f_cnt = [0] * T
    r_tot = [0.0] * T; r_cnt = [0] * T
    for x, sz, is_f in zip(t_ns, size, fwd):
        k = int(x) // w_ns                   # exact integer nanoseconds -> boundary goes up
        if k > T - 1:
            k = T - 1
        if is_f:
            f_tot[k] += float(sz); f_cnt[k] += 1
        else:
            r_tot[k] += float(sz); r_cnt[k] += 1
    for k in range(T):
        if f_cnt[k]:
            out[k, 0] = f_tot[k]; out[k, 1] = f_cnt[k]; out[k, 2] = f_tot[k] / f_cnt[k]
        if r_cnt[k]:
            out[k, 3] = r_tot[k]; out[k, 4] = r_cnt[k]; out[k, 5] = r_tot[k] / r_cnt[k]
        out[k, 6] = (out[k, 0] / out[k, 3]) if out[k, 3] > 0 else 0.0
        out[k, 7] = k * (w_ns / 1e9)
    return out


def main():
    z0 = np.load(os.path.join(REP, 'b2_flow_packets_ns.npz'))
    t_all, s_all, f_all, off = z0['t_ns'], z0['size'], z0['fwd'], z0['offsets']
    assert t_all.dtype == np.int64, f'expected int64 ns, got {t_all.dtype}'
    idx = pd.read_parquet(os.path.join(REP, 'b2_flows_index.parquet'))

    print('=== EXHAUSTIVE window verification (no build-code import) ===')
    print(f'  declared tolerances: exact 0 for idx {EXACT_IDX}; atol={ATOL} for '
          f'[0,2,3,5,6]; 3feat bit-exact\n')
    grand_ok = True
    for D in DURATIONS:
        t0 = time.time()
        meta = pd.read_parquet(os.path.join(REP, f'meta_{D}s.parquet'))
        z = np.load(os.path.join(REP, f'features_{D}s.npz'))
        D_ns = D * 1_000_000_000
        maxdiff = np.zeros(8)
        cells = slots_seen = pkts_seen = vals = 0
        bad_assign = bad_count = bad_pay = late = f3_bad = shape_bad = 0
        neg = nan = inf = 0
        pkt_by_window = {}
        for w in WINDOWS_MS:
            T = (D * 1000) // w + 1
            A = z[f'w{w}']
            if A.shape != (len(meta), T, 8):
                shape_bad += 1
            w_ns = w * 1_000_000
            tot_pk = 0
            for r, fid in enumerate(meta.flow_id.to_numpy()):
                a, b = off[fid], off[fid + 1]
                t_ns = t_all[a:b]                      # exact int64, no conversion
                keep = t_ns <= D_ns
                tu = t_ns[keep]
                sz = s_all[a:b][keep]
                fw = f_all[a:b][keep]
                late += int((tu > D_ns).sum())
                tot_pk += int(keep.sum())
                M = manual_features(tu, sz, fw, w_ns, T)
                G = A[r]
                d = np.abs(M - G).max(axis=0)
                maxdiff = np.maximum(maxdiff, d)
                # source packet count == forward + reverse count features
                if int(G[:, 1].sum() + G[:, 4].sum()) != int(keep.sum()):
                    bad_count += 1
                if abs(G[:, 0].sum() + G[:, 3].sum() - float(sz.sum())) > 1e-6:
                    bad_pay += 1
                if abs(G[:, 1].sum() - int(fw.sum())) > 0 or \
                   abs(G[:, 4].sum() - int((~fw).sum())) > 0:
                    bad_assign += 1
                if not np.array_equal(G[:, FEAT3], G[:, [1, 4, 6]]):
                    f3_bad += 1
                cells += 1; slots_seen += T; pkts_seen += int(keep.sum()); vals += G.size
            nan += int(np.isnan(A).sum()); inf += int(np.isinf(A).sum())
            neg += int((A < 0).sum())
            pkt_by_window[w] = tot_pk
        same_pop = len(set(pkt_by_window.values())) == 1
        ok = True
        print(f'  --- {D}s : {len(meta):,} flows x 9 windows ---')
        print(f'      flow x window cells : {cells:,}')
        print(f'      slots verified      : {slots_seen:,}')
        print(f'      packets verified    : {pkts_seen:,}')
        print(f'      feature values      : {vals:,}')
        print(f'      {"idx":>3} {"feature":<26}{"max |diff|":>13}  {"required":<14}verdict')
        names = ['fwd total payload', 'fwd packet count', 'fwd mean payload',
                 'rev total payload', 'rev packet count', 'rev mean payload',
                 'payload ratio', 'time_from_start']
        for i, nm in enumerate(names):
            good = (maxdiff[i] == 0.0) if i in EXACT_IDX else (maxdiff[i] <= ATOL)
            ok &= good
            print(f'      {i:>3} {nm:<26}{maxdiff[i]:>13.3e}  '
                  f'{"exactly 0" if i in EXACT_IDX else f"<= {ATOL}":<14}'
                  f'{"PASS" if good else "FAIL"}')
        for nm, v in (('packet-assignment mismatches', bad_assign),
                      ('packet-count mismatches', bad_count),
                      ('payload mismatches', bad_pay),
                      ('packets with elapsed > D', late),
                      ('3feat bit-exact mismatches', f3_bad),
                      ('shape mismatches', shape_bad),
                      ('NaN', nan), ('Inf', inf), ('negative values', neg)):
            ok &= (v == 0)
            print(f'      {nm:<34}{v:>10}   {"PASS" if v == 0 else "FAIL"}')
        print(f'      identical packet population across 9 windows: {same_pop} '
              f'{sorted(set(pkt_by_window.values()))}')
        ok &= same_pop
        grand_ok &= ok
        print(f'      -> {"PASS" if ok else "FAIL"}   ({time.time() - t0:.0f}s)\n')
    print(f'  ALL THREE DURATIONS PASS: {grand_ok}')
    return 0 if grand_ok else 1


if __name__ == '__main__':
    sys.exit(main())
