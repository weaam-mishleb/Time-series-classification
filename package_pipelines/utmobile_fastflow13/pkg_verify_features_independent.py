"""Independent verification of the FastFlow-13 feature tables.

Replaces the inherited TCBench verifier, which rebuilt features from the official
34,378-flow pickle. That pickle is deliberately NOT part of this population, so this
verifier reconstructs and cross-checks the integer-nanosecond feature tables instead.

It reads NOTHING outside the package and takes no required arguments, so it runs through
`./run_all.sh verify` unchanged.

What it checks, for every flow and every one of the nine windows:
  * T                == floor(D_ms / w_ms) + 1, and the stored array length matches
  * feature index 7  == slot_index * window_seconds, EXACTLY (integer-derived time axis)
  * forward + reverse packet counts   == packets_in_window recorded per row
  * forward + reverse payload totals  == payload_in_window recorded per row
  * mean payload (idx 2, 5)           == total / count where count > 0, else exactly 0
  * payload ratio  (idx 6)            == idx0 / idx3 where idx3 > 0, else exactly 0
  * 3feat                             == 8feat[:, [1, 4, 6]], BIT-EXACT
  * the packet population is IDENTICAL across all nine windows for every flow
  * eligibility: flow_length_ns > D * 10**9, a STRICT integer test
  * no NaN, no Inf, no negative value anywhere
  * population, capture count and class set agree with PACKAGE_ID.json and the split
  * recorded source hashes in reports/file_hashes.csv still match the files on disk

Optional cross-check: if UTMPKG_RECON_DIR points at the reconstruction reports directory,
every feature value is additionally compared against features_<D>s.npz element by element.
Absent that variable the structural verification above still runs in full.
"""
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'lib'))
import utmpaths as P

FEAT3 = [1, 4, 6]
FAIL = []


def check(name, ok, detail=''):
    print(f'  [{"OK " if ok else "FAIL"}] {name}' + (f'   {detail}' if detail else ''))
    if not ok:
        FAIL.append(name)


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()


def main():
    pid = P.PKG_ID
    D = P.BIG_WINDOW_S
    rec = json.load(open(P.split_path()))
    print(f'=== independent feature verification: {P.VARIANT} ===')
    print(f'  duration {D}s   windows {len(P.WINDOWS)}   expected flows {pid["n_flows"]:,}')
    print(f'  source: the package parquet only; no pickle, no external population\n')

    df = pd.read_parquet(P.parquet_path())
    check('parquet duration column matches the package',
          set(df.big_window_s.unique()) == {float(D)}, str(sorted(df.big_window_s.unique())))
    check('nine distinct small windows present',
          sorted(df.small_window_ms.unique()) == sorted(int(w[:-2]) for w in P.WINDOWS),
          str(sorted(df.small_window_ms.unique())))
    check('row count == flows x 9 windows',
          len(df) == pid['n_flows'] * len(P.WINDOWS),
          f"{len(df):,} == {pid['n_flows']:,} x {len(P.WINDOWS)}")
    check('flow population matches PACKAGE_ID',
          df.row_id.nunique() == pid['n_flows'], f'{df.row_id.nunique():,}')
    check('capture population matches the split',
          df.capture.nunique() == rec['n_captures'], f'{df.capture.nunique():,}')
    check('class set matches PACKAGE_ID',
          sorted(df.app.unique()) == sorted(pid['classes']),
          f'{df.app.nunique()} classes')
    check(f'eligibility is STRICT: every flow_length_ns > {D}e9',
          bool((df.flow_length_ns > D * 10 ** 9).all()),
          f'min {int(df.flow_length_ns.min()):,} ns')
    check('no flow sits exactly on the duration boundary',
          int((df.flow_length_ns == D * 10 ** 9).sum()) == 0)

    recon = os.environ.get('UTMPKG_RECON_DIR', '')
    Z = None
    if recon and os.path.exists(os.path.join(recon, f'features_{D}s.npz')):
        Z = np.load(os.path.join(recon, f'features_{D}s.npz'))
        print(f'\n  optional cross-check against the reconstruction: ENABLED')
    else:
        print(f'\n  optional cross-check against the reconstruction: not requested'
              f' (set UTMPKG_RECON_DIR to enable)')

    print('\n  per-window verification')
    print(f'    {"window":>7}{"T":>5}{"flows":>8}{"slots":>10}{"packets":>12}'
          f'{"max|diff| 3feat":>17}{"verdict":>9}')
    pkt_by_window, pay_by_window = {}, {}
    recon_maxdiff = 0.0
    allok = True
    for w in P.WINDOWS:
        ms = int(w[:-2])
        T_exp = (D * 1000) // ms + 1
        g = df[df.small_window_ms == ms].sort_values('row_id').reset_index(drop=True)
        F8 = np.stack([np.asarray(v, dtype=np.float64).reshape(T_exp, 8) for v in g.feat_8])
        F3 = np.stack([np.asarray(v, dtype=np.float64).reshape(T_exp, 3) for v in g.feat_3])
        ok = True
        ok &= bool((g['T'] == T_exp).all())
        # time axis is derived from the integer slot index, not accumulated floats
        axis = np.arange(T_exp, dtype=np.float64) * (ms / 1000.0)
        ok &= bool(np.array_equal(F8[:, :, 7], np.broadcast_to(axis, F8[:, :, 7].shape)))
        # counts and payloads must reproduce the per-row metadata exactly
        cnt = F8[:, :, 1].sum(axis=1) + F8[:, :, 4].sum(axis=1)
        pay = F8[:, :, 0].sum(axis=1) + F8[:, :, 3].sum(axis=1)
        ok &= bool(np.array_equal(cnt.astype(np.int64), g.packets_in_window.to_numpy()))
        ok &= bool(np.abs(pay - g.payload_in_window.to_numpy()).max() <= 1e-6)
        # derived features recomputed from the primitives they are built from
        for tot, c, mean in ((0, 1, 2), (3, 4, 5)):
            exp = np.divide(F8[:, :, tot], F8[:, :, c],
                            out=np.zeros_like(F8[:, :, tot]), where=F8[:, :, c] > 0)
            ok &= bool(np.abs(exp - F8[:, :, mean]).max() <= 1e-9)
        ratio = np.divide(F8[:, :, 0], F8[:, :, 3],
                          out=np.zeros_like(F8[:, :, 0]), where=F8[:, :, 3] > 0)
        ok &= bool(np.abs(ratio - F8[:, :, 6]).max() <= 1e-9)
        d3 = float(np.abs(F3 - F8[:, :, FEAT3]).max()) if F3.size else 0.0
        ok &= bool(np.array_equal(F3, F8[:, :, FEAT3]))
        ok &= not bool(np.isnan(F8).any() or np.isinf(F8).any() or (F8 < 0).any())
        if Z is not None:
            R = Z[f'w{ms}']
            if R.shape == F8.shape:
                recon_maxdiff = max(recon_maxdiff, float(np.abs(R - F8).max()))
            else:
                ok = False
        pkt_by_window[ms] = int(cnt.sum())
        pay_by_window[ms] = round(float(pay.sum()), 3)
        allok &= ok
        print(f'    {w:>7}{T_exp:>5}{len(g):>8,}{len(g) * T_exp:>10,}'
              f'{int(cnt.sum()):>12,}{d3:>17.3e}{"PASS" if ok else "FAIL":>9}')

    check('all nine windows internally consistent', allok)
    check('packet population identical across all nine windows',
          len(set(pkt_by_window.values())) == 1, str(sorted(set(pkt_by_window.values()))))
    check('payload total identical across all nine windows',
          len(set(pay_by_window.values())) == 1, str(sorted(set(pay_by_window.values()))))
    if Z is not None:
        check('every feature value matches the reconstruction',
              recon_maxdiff == 0.0, f'max |diff| {recon_maxdiff:.3e}')

    print('\n  recorded source hashes')
    hp = os.path.join(P.REPORTS, 'file_hashes.csv')
    if os.path.exists(hp):
        h = pd.read_csv(hp)
        drift = []
        for _, r in h.iterrows():
            fp = os.path.join(P.ROOT, r.file)
            if not os.path.exists(fp):
                drift.append(f'{r.file}: MISSING')
            elif sha256(fp) != r.sha256:
                drift.append(f'{r.file}: CHANGED')
        for _, r in h.iterrows():
            print(f'    {r.file:<62}{r.sha256[:16]}')
        check('recorded source hashes still match', not drift, '; '.join(drift[:3]))
    else:
        check('reports/file_hashes.csv present', False, 'missing')

    print(f'\n  FAILURES: {len(FAIL)}')
    for f in FAIL:
        print(f'    - {f}')
    return 0 if not FAIL else 1


if __name__ == '__main__':
    sys.exit(main())
