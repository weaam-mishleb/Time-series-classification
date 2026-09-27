"""Pre-flight checks for the FastFlow-13 B2 >=50 packages.

Nothing here trains and nothing here writes a checkpoint.

This replaces the TCBench validator. The TCBench version reproduced its population from
the official 34,378-flow pickle; this population was deliberately rebuilt from the raw
UTMobileNet2021 CSV archive instead, so the reproduction section here checks the
recorded population gate and the per-duration eligibility rule rather than a pickle.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('MPLBACKEND', 'Agg')

import json, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'lib'))
import utmpaths as P
P.install()

import numpy as np, pandas as pd
import capture14 as C

PATTERNS = ['/' + 'data/weaamm', '/' + 'media/Data', '/' + 'home/weaamm', '/' + 'tmp/claude']
FAIL = []


def check(name, ok, detail=''):
    print(f'  [{"OK " if ok else "FAIL"}] {name}' + (f'   {detail}' if detail else ''))
    if not ok:
        FAIL.append(name)


def main():
    pid = P.PKG_ID
    print('=== data provenance ===')
    check('official parquet present', os.path.exists(P.parquet_path()),
          os.path.basename(P.parquet_path()))
    bad = [f for r, d, fs in os.walk(P.ROOT) for f in fs
           if 'corrected' in f.lower() or 'EXPERIMENTAL' in f]
    check('no corrected/EXPERIMENTAL file in the package', not bad, '; '.join(bad[:3]))
    check('source pickle NOT used', pid.get('source_pkl_used') is False)
    check('source archive recorded', 'UTMobileNet2021' in pid.get('source_archive', ''),
          pid.get('source_archive', '')[:48])

    print('\n=== population gate (rebuilt from the raw CSV archive) ===')
    gate = pid['population_gate']
    check('gate total is 12,459 filtered biflows', gate['total_flows'] == 12459,
          str(gate['total_flows']))
    check('gate lists exactly 13 applications', len(gate['per_app']) == 13,
          str(len(gate['per_app'])))
    check('gate per-app counts sum to the total',
          sum(gate['per_app'].values()) == gate['total_flows'],
          str(sum(gate['per_app'].values())))
    check('excluded apps recorded', sorted(gate['excluded_apps']) ==
          ['dropbox', 'hulu', 'pandora', 'skype'], str(sorted(gate['excluded_apps'])))
    check('packet rule is >= 50 inclusive on the COMPLETE biflow',
          'packet_count >= 50' in pid['packet_rule'] and 'COMPLETE' in pid['packet_rule'],
          pid['packet_rule'][:56])
    check('no class-count filter applied', 'no class-count filter' in pid['packet_rule'])

    print('\n=== duration rule and boundary arithmetic ===')
    check(f'duration rule is STRICTLY > {P.BIG_WINDOW_S}',
          'STRICTLY' in pid['duration_rule'] and str(P.BIG_WINDOW_S) in pid['duration_rule'],
          pid['duration_rule'])
    check('timestamps are exact int64 nanoseconds',
          'int64 nanoseconds' in pid['timestamp_representation'],
          pid['timestamp_representation'])
    check('boundary rule documents the UPPER slot',
          'UPPER' in pid['boundary_rule'], pid['boundary_rule'][:64])
    check('boundary rule is integer, not float',
          'INTEGER' in pid['boundary_rule'].upper())

    print('\n=== split (reused, never regenerated) ===')
    rec = json.load(open(P.split_path()))
    check('split fingerprint', rec['split_fingerprint'] == P.SPLIT_FINGERPRINT,
          rec['split_fingerprint'])
    check('split seed recorded', rec['split_seed'] == P.SPLIT_SEED, f"seed {rec['split_seed']}")
    check('capture-level grouping', 'CAPTURE' in rec['protocol'].upper(), rec['protocol'][:48])
    tr, va = set(rec['train_idx']), set(rec['test_idx'])
    check('zero flow overlap', not (tr & va), f'{len(tr & va)}')
    f2c = pd.read_csv(P.flow_to_capture_path())
    ctr = set(f2c[f2c.row_id.isin(tr)].capture); cva = set(f2c[f2c.row_id.isin(va)].capture)
    check('zero capture leakage', not (ctr & cva), f'{len(ctr & cva)}')
    check('each flow maps to one capture', f2c.row_id.nunique() == len(f2c))
    check('each capture carries one label',
          int((f2c.groupby('capture').app.nunique() > 1).sum()) == 0)
    check('70/30 within half a percent',
          abs(rec['n_captures_train'] / (rec['n_captures_train'] + rec['n_captures_test'])
              - 0.7) < 0.005,
          f"captures {rec['n_captures_train']}/{rec['n_captures_test']}")
    tp, vp = rec['train_flows_per_class'], rec['test_flows_per_class']
    check('all 13 classes on both sides',
          all(tp.get(c, 0) > 0 and vp.get(c, 0) > 0 for c in rec['classes']),
          f"smallest val class {min(vp.values())}")
    check('flow counts add up', rec['n_flows_train'] + rec['n_flows_test'] == rec['n_flows'],
          f"{rec['n_flows_train']}+{rec['n_flows_test']}=={rec['n_flows']}")

    print('\n=== data shapes, every window and config ===')
    ok = True
    for cfg in P.FEATURE_CONFIGS:
        for w in P.WINDOWS:
            X, ids = C.load_window(w, cfg)
            good = (X.shape == (pid['n_flows'], P.T_BY_WINDOW[int(w[:-2])],
                                P.FEATURE_CONFIGS[cfg])
                    and np.isnan(X).sum() == 0 and np.isinf(X).sum() == 0)
            ok &= good
            if not good:
                check(f'{cfg} {w}', False, str(X.shape))
    check('all 18 (config, window) shapes correct, 0 NaN/Inf', ok,
          f"{pid['n_flows']} flows each")
    X3, _ = C.load_window('20ms', '3feat'); X8, _ = C.load_window('20ms', '8feat')
    check('3feat == 8feat[:, :, [1,4,6]]', np.array_equal(X3, X8[:, :, P.FEAT_3_INDICES]))
    check('no negative feature values', float(X8.min()) >= 0.0, f'min {float(X8.min())}')

    print('\n=== classes come from the config, never from a batch ===')
    check('PACKAGE_ID lists 13 classes', len(pid['classes']) == 13, str(len(pid['classes'])))
    check('split classes match PACKAGE_ID', list(rec['classes']) == list(pid['classes']))
    check('NUM_CLASS resolves to 13 from config', P.NUM_CLASS == 13, str(P.NUM_CLASS))

    print('\n=== prepare() and weights ===')
    d = C.prepare('250ms', '8feat')
    check('prepare returns the frozen split sizes',
          len(d['ytr']) == rec['n_flows_train'] and len(d['yva']) == rec['n_flows_test'],
          f"{len(d['ytr'])}/{len(d['yva'])}")
    check('standardization fitted on train only',
          abs(float(d['Xtr'].mean())) < 1e-9 and abs(float(d['Xtr'].std()) - 1) < 1e-9,
          f"train mean {float(d['Xtr'].mean()):.2e} std {float(d['Xtr'].std()):.4f}")
    w, cnt = C.class_weights(d['ytr'])
    check('sum(w_c * count_c) == n_train_flows',
          abs(float((w * cnt).sum()) - len(d['ytr'])) < 1e-6,
          f"{float((w * cnt).sum()):.1f} == {len(d['ytr'])}")

    print('\n=== PACKAGE_ID contract ===')
    check('num_classes == 13', pid['num_classes'] == 13, str(pid['num_classes']))
    check('weighted == true', pid['weighted'] is True)
    check('deterministic == true', pid['deterministic'] is True)
    check('aggregation == mean_probability', pid['aggregation'] == 'mean_probability')
    check('split fingerprint matches the split file',
          pid['split_fingerprint'] == rec['split_fingerprint'], pid['split_fingerprint'])
    check('big_window_seconds matches the package',
          pid['big_window_seconds'] == P.BIG_WINDOW_S, str(pid['big_window_seconds']))
    check('feature table documents both layers', len(pid.get('features', [])) == 8)
    check('runs == 324', pid['runs'] == 324, str(pid['runs']))
    check('population label does not overclaim FastFlow',
          'NOT claimed to be identical' in pid['population_note'])

    print('\n=== self-contained ===')
    bad = []
    for root, dirs, files in os.walk(P.ROOT):
        dirs[:] = [x for x in dirs if x != '__pycache__']
        for f in files:
            if not f.endswith(('.py', '.sh', '.md', '.txt')):
                continue
            txt = open(os.path.join(root, f), encoding='utf-8', errors='ignore').read()
            for pat in PATTERNS:
                for ln, line in enumerate(txt.splitlines(), 1):
                    if pat in line and not line.lstrip().startswith('#'):
                        bad.append(f'{os.path.relpath(os.path.join(root, f), P.ROOT)}:{ln}')
    check('no absolute host path in code/docs', not bad, '; '.join(bad[:4]))

    print(f'\n  FAILURES: {len(FAIL)}')
    for f in FAIL:
        print(f'    - {f}')
    return 0 if not FAIL else 1


if __name__ == '__main__':
    sys.exit(main())
