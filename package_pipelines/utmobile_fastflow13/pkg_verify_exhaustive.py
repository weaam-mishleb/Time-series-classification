"""Exhaustive per-slot verification of the FastFlow-13 feature tables.

Replaces the inherited TCBench script, which required an undocumented `--root` argument
and rebuilt packet series from the official 34,378-flow pickle. Neither applies here: the
pickle is not part of this population, and `--root` now defaults to the package itself,
so the script runs with no arguments.

Where verify_features_independent.py checks aggregate invariants per window, this one
walks EVERY slot of EVERY flow in EVERY window and re-derives each of the eight features
from the primitives, using a plain Python loop rather than any vectorised grouping.

Checked per slot:
  idx 0,3  forward / reverse payload totals   : finite, non-negative
  idx 1,4  forward / reverse packet counts    : non-negative INTEGERS
  idx 2,5  mean payload                       : total / count, or exactly 0 when count 0
  idx 6    payload ratio                      : idx0 / idx3, or exactly 0 when idx3 == 0
  idx 7    time axis                          : slot_index * window_seconds, EXACTLY
  3feat    == 8feat[:, [1, 4, 6]]             : BIT-EXACT

Checked per flow: the packet count and payload total summed over slots are invariant
across all nine windows -- the strongest available evidence that integer slot assignment
neither dropped nor duplicated a packet.

Optional cross-check: set UTMPKG_RECON_DIR to the reconstruction reports directory to
additionally compare every value against features_<D>s.npz.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'lib'))
import utmpaths as P

FEAT3 = [1, 4, 6]
ATOL = 1e-9


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', default=P.ROOT,
                    help='package root (defaults to this package; no need to pass it)')
    ap.add_argument('--max-flows', type=int, default=0,
                    help='0 = every flow (the default); a positive value truncates')
    args = ap.parse_args()

    D = P.BIG_WINDOW_S
    pid = P.PKG_ID
    t0 = time.time()
    print(f'=== exhaustive slot verification: {P.VARIANT} ===')
    print(f'  root {args.root}')
    print(f'  every slot of every flow in all {len(P.WINDOWS)} windows\n')

    recon = os.environ.get('UTMPKG_RECON_DIR', '')
    Z = None
    if recon and os.path.exists(os.path.join(recon, f'features_{D}s.npz')):
        Z = np.load(os.path.join(recon, f'features_{D}s.npz'))
        print('  optional cross-check against the reconstruction: ENABLED\n')

    df = pd.read_parquet(P.parquet_path())
    bad = dict(count_not_integer=0, mean_fwd=0, mean_rev=0, ratio=0, axis=0,
               feat3=0, negative=0, nonfinite=0, meta_count=0, meta_payload=0,
               length=0, recon=0)
    cells = slots = 0
    per_flow = {}
    for w in P.WINDOWS:
        ms = int(w[:-2])
        T = (D * 1000) // ms + 1
        g = df[df.small_window_ms == ms].sort_values('row_id').reset_index(drop=True)
        wsec = ms / 1000.0
        for i, r in enumerate(g.itertuples()):
            a8 = np.asarray(r.feat_8, dtype=np.float64)
            a3 = np.asarray(r.feat_3, dtype=np.float64)
            if a8.size != T * 8 or a3.size != T * 3:
                bad['length'] += 1
                continue
            F = a8.reshape(T, 8)
            G = a3.reshape(T, 3)
            if args.max_flows and i >= args.max_flows:
                break
            tot_c = 0.0
            tot_p = 0.0
            for k in range(T):
                f0, f1, f2, f3, f4, f5, f6, f7 = F[k]
                if not np.isfinite(F[k]).all():
                    bad['nonfinite'] += 1
                if (F[k] < 0).any():
                    bad['negative'] += 1
                if f1 != int(f1) or f4 != int(f4):
                    bad['count_not_integer'] += 1
                exp2 = (f0 / f1) if f1 > 0 else 0.0
                exp5 = (f3 / f4) if f4 > 0 else 0.0
                exp6 = (f0 / f3) if f3 > 0 else 0.0
                if abs(exp2 - f2) > ATOL:
                    bad['mean_fwd'] += 1
                if abs(exp5 - f5) > ATOL:
                    bad['mean_rev'] += 1
                if abs(exp6 - f6) > ATOL:
                    bad['ratio'] += 1
                if f7 != k * wsec:
                    bad['axis'] += 1
                if G[k, 0] != f1 or G[k, 1] != f4 or G[k, 2] != f6:
                    bad['feat3'] += 1
                tot_c += f1 + f4
                tot_p += f0 + f3
                slots += 1
            if int(tot_c) != int(r.packets_in_window):
                bad['meta_count'] += 1
            if abs(tot_p - float(r.payload_in_window)) > 1e-6:
                bad['meta_payload'] += 1
            per_flow.setdefault(int(r.row_id), {})[ms] = (int(tot_c), round(tot_p, 3))
            cells += 1
        if Z is not None:
            R = Z[f'w{ms}']
            A = np.stack([np.asarray(v, dtype=np.float64).reshape(T, 8) for v in g.feat_8])
            if R.shape != A.shape or not np.array_equal(R, A):
                bad['recon'] += 1
        print(f'    {w:>7}  T={T:<4} flows {len(g):>6,}  slots {len(g) * T:>9,}  '
              f'{time.time() - t0:>6.0f}s', flush=True)

    invariant = sum(1 for v in per_flow.values() if len(set(v.values())) != 1)
    print(f'\n  flow x window cells verified : {cells:,}')
    print(f'  slots verified               : {slots:,}')
    print(f'  flows tracked across windows : {len(per_flow):,}')
    print()
    names = [('slot arrays with a wrong length', 'length'),
             ('non-finite values', 'nonfinite'),
             ('negative values', 'negative'),
             ('non-integer packet counts', 'count_not_integer'),
             ('forward mean payload mismatches', 'mean_fwd'),
             ('reverse mean payload mismatches', 'mean_rev'),
             ('payload ratio mismatches', 'ratio'),
             ('time-axis mismatches', 'axis'),
             ('3feat bit-exact mismatches', 'feat3'),
             ('packet count vs recorded metadata', 'meta_count'),
             ('payload total vs recorded metadata', 'meta_payload')]
    ok = True
    for label, key in names:
        ok &= bad[key] == 0
        print(f'    {label:<40}{bad[key]:>8}   {"PASS" if bad[key] == 0 else "FAIL"}')
    ok &= invariant == 0
    print(f'    {"flows whose packets/payload vary by window":<40}{invariant:>8}   '
          f'{"PASS" if invariant == 0 else "FAIL"}')
    if Z is not None:
        ok &= bad['recon'] == 0
        print(f'    {"windows differing from the reconstruction":<40}{bad["recon"]:>8}   '
              f'{"PASS" if bad["recon"] == 0 else "FAIL"}')
    print(f'\n  EXHAUSTIVE VERIFICATION {"PASSED" if ok else "FAILED"}   '
          f'({time.time() - t0:.0f}s)')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
