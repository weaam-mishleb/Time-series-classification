"""Create the three self-contained FastFlow-13 UTMobileNet packages.

Population: the gate-verified 12,459 biflows (B2 packet filter applied BEFORE biflow
construction, 13 FastFlow apps, packet_count >= 50 over the COMPLETE filtered biflow,
no class-count filter). Per duration only flows with complete duration STRICTLY greater
than D are eligible.

Timestamps and slot boundaries are EXACT INTEGER NANOSECONDS end to end; see
TIMESTAMP_AND_BOUNDARY_METHODOLOGY.md. This differs from the older float-based packages.

Split: stratified 70/30 at CAPTURE level (partition||fname), stratified by app, seed 42.
Each duration gets its OWN split computed on its OWN eligible population.

Nothing outside the three target directories is written.
"""
import hashlib
import json
import os
import shutil
import sys
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REP = os.path.join(ROOT, 'reports')
TEMPLATE = '/data/weaamm/utmobile_tcbench_3sec_capture14_weighted_det_6models_3seeds'
BASE = '/data/weaamm'
WINDOWS_MS = [20, 30, 40, 50, 75, 100, 150, 200, 250]
DURATIONS = [3, 5, 8]
FEAT3 = [1, 4, 6]
SPLIT_SEED = 42
TEST_SIZE = 0.3

FEATURE_MEANING = [
    (0, 'total_size_dir_-1', "total payload of FORWARD packets in the slot (pkts_dir == 1, "
        "same source IP as the flow's first packet)"),
    (1, 'packet_count_dir_-1', 'count of FORWARD packets in the slot (pkts_dir == 1)'),
    (2, 'avg_payload_dir_-1', 'mean payload of FORWARD packets in the slot (pkts_dir == 1)'),
    (3, 'total_size_dir_1', 'total payload of REVERSE packets in the slot (pkts_dir == 0)'),
    (4, 'packet_count_dir_1', 'count of REVERSE packets in the slot (pkts_dir == 0)'),
    (5, 'avg_payload_dir_1', 'mean payload of REVERSE packets in the slot (pkts_dir == 0)'),
    (6, 'upstream_downstream_ratio', 'total_payload_forward / total_payload_reverse '
        '(index 0 / index 3), and 0 when the denominator is 0. It is a PAYLOAD-SIZE '
        'ratio, not a packet-count ratio.'),
    (7, 'time_from_start', 'slot_index * window_seconds; identical for every flow, so it '
        'carries no per-sample information'),
]


def pkg_dir(D):
    return os.path.join(
        BASE, f'utmobile_fastflow13_b2_ge50_{D}sec_capture70_weighted_det_6models_3seeds')


def fingerprint(y):
    return hashlib.sha256(np.asarray(y).astype(np.int64).tobytes()).hexdigest()[:16]


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def build_split(meta, classes):
    """Stratified 70/30 over CAPTURES, stratified by the capture's app, seed 42."""
    cap = (meta.partition.astype(str) + '||' + meta.capture_id.astype(str)).to_numpy()
    meta = meta.assign(capture=cap)
    caps = meta.drop_duplicates('capture')[['capture', 'app']].sort_values('capture')
    cap_list = caps.capture.to_numpy()
    cap_app = caps.app.to_numpy()
    tr_caps, te_caps = train_test_split(
        cap_list, test_size=TEST_SIZE, random_state=SPLIT_SEED, stratify=cap_app)
    tr_set, te_set = set(tr_caps.tolist()), set(te_caps.tolist())
    assert not (tr_set & te_set), 'capture leakage'
    is_tr = meta.capture.isin(tr_set).to_numpy()
    train_idx = meta.flow_id.to_numpy()[is_tr].tolist()
    test_idx = meta.flow_id.to_numpy()[~is_tr].tolist()
    assert not (set(train_idx) & set(test_idx)), 'flow overlap'
    y_all = np.searchsorted(np.array(classes), meta.app.to_numpy())
    rec = dict(
        protocol=('stratified 70/30 at CAPTURE level (partition||fname), stratified by '
                  'app, seed 42'),
        split_seed=SPLIT_SEED,
        filter_rule=('FastFlow-style population: B2 packet filter applied BEFORE biflow '
                     'construction; 13 FastFlow applications; packet_count >= 50 '
                     '(INCLUSIVE) over the COMPLETE filtered biflow; NO class-count '
                     'filter; per duration, complete duration STRICTLY > D'),
        filter_source=('reproduced from the raw UTMobileNet2021 CSV archive, all four '
                       'partitions; the 34,378-flow pickle was NOT used'),
        boundary_rule=('slot = min(elapsed_ns // window_ns, T-1) in EXACT INTEGER '
                       'NANOSECONDS; a packet exactly on a boundary enters the UPPER '
                       'slot; a packet exactly at elapsed == D is included'),
        n_classes=len(classes), classes=list(classes),
        n_captures=int(len(cap_list)),
        n_captures_train=int(len(tr_caps)), n_captures_test=int(len(te_caps)),
        n_flows=int(len(meta)),
        n_flows_train=int(len(train_idx)), n_flows_test=int(len(test_idx)),
        train_captures_per_class=meta[is_tr].drop_duplicates('capture')
            .app.value_counts().to_dict(),
        test_captures_per_class=meta[~is_tr].drop_duplicates('capture')
            .app.value_counts().to_dict(),
        train_flows_per_class=meta[is_tr].app.value_counts().to_dict(),
        test_flows_per_class=meta[~is_tr].app.value_counts().to_dict(),
        capture_leakage=0, flow_overlap=0,
        split_fingerprint=fingerprint(y_all),
        train_idx=[int(x) for x in train_idx],
        test_idx=[int(x) for x in test_idx],
    )
    return rec, meta


def build_parquet(meta, z, D):
    rows = []
    for w in WINDOWS_MS:
        T = (D * 1000) // w + 1
        A = z[f'w{w}']
        assert A.shape == (len(meta), T, 8)
        for i, r in enumerate(meta.itertuples()):
            f8 = A[i]
            rows.append(dict(
                row_id=int(r.flow_id), app=r.app, partition=r.partition,
                fname=r.capture_id, capture=r.capture, ip_proto=int(r.proto),
                total_packets=int(r.packets_full), packets_in_window=int(r.packets_in_D),
                payload_in_window=float(r.payload_in_D),
                flow_length_seconds=float(r.duration_s),
                flow_length_ns=int(r.duration_ns),
                big_window_s=float(D), small_window_ms=int(w), T=int(T),
                n_features_8=8, n_features_3=3,
                feat_8=f8.reshape(-1).astype(np.float64),
                feat_3=f8[:, FEAT3].reshape(-1).astype(np.float64)))
    return pd.DataFrame(rows)


def write_utmpaths(dst, D):
    src = open(os.path.join(TEMPLATE, 'lib', 'utmpaths.py')).read()
    src = src.replace(
        "VARIANT = f'UTMobileNet21_capture14_weighted_det_{BIG_WINDOW_S}s'",
        "VARIANT = f'UTMobileNet21_fastflow13_b2_ge50_weighted_det_{BIG_WINDOW_S}s'")
    src = src.replace("TAG = f'UTM{BIG_WINDOW_S}sC14'",
                      "TAG = f'UTM{BIG_WINDOW_S}sFF13'")
    src = src.replace(
        "        DATA, f'utmobilenet21_capture14_first{BIG_WINDOW_S}s_9windows_8feat_3feat.parquet')",
        "        DATA, f'utmobilenet21_fastflow13_b2_ge50_first{BIG_WINDOW_S}s_"
        "9windows_8feat_3feat.parquet')")
    src = src.replace(
        "    return os.path.join(SPLITS, f'split_capture70_{BIG_WINDOW_S}s_14class.json')",
        "    return os.path.join(SPLITS, f'split_capture70_{BIG_WINDOW_S}s_13class.json')")
    src = src.replace(
        '''def source_pkl():
    """Official source pickle. Resolved from the environment first, then PACKAGE_ID.json,
    so no absolute host path is ever written into the code itself."""
    return os.environ.get('UTMPKG_SOURCE_PKL', PKG_ID['source_pkl_path'])''',
        '''def source_archive():
    """Raw UTMobileNet2021 CSV archive this package was reproduced from. There is no
    source pickle: the 34,378-flow pickle was deliberately NOT used."""
    return os.environ.get('UTMPKG_SOURCE_ARCHIVE', PKG_ID['source_archive_path'])''')
    src = src.replace('capture-14 weighted deterministic\npackages',
                      'FastFlow-13 B2 >=50 weighted deterministic\npackages')
    open(os.path.join(dst, 'lib', 'utmpaths.py'), 'w').write(src)


def main():
    t0 = time.time()
    z_all = {D: np.load(os.path.join(REP, f'features_{D}s.npz')) for D in DURATIONS}
    full = pd.read_parquet(os.path.join(REP, 'b2_flows_index.parquet'))
    gate = dict(total_flows=int(len(full)),
                per_app={k: int(v) for k, v in
                         full.app.value_counts().sort_index().items()},
                excluded_apps=['dropbox', 'hulu', 'pandora', 'skype'],
                total_packets=int(full.packets.sum()),
                note=('B2 packet filter applied BEFORE biflow construction; '
                      'packet_count >= 50 inclusive over the COMPLETE filtered biflow'))
    if gate['total_flows'] != 12459:
        print(f"  FATAL: population gate is {gate['total_flows']}, expected 12,459")
        return 1
    summary = []
    for D in DURATIONS:
        dst = pkg_dir(D)
        if os.path.exists(dst):
            print(f'  REFUSING: {dst} already exists'); return 1
        meta = pd.read_parquet(os.path.join(REP, f'meta_{D}s.parquet'))
        classes = sorted(meta.app.unique().tolist())
        if len(classes) != 13:
            print(f'  FATAL: {len(classes)} classes at {D}s, expected 13'); return 1

        rec, meta = build_split(meta, classes)
        for sub in ('lib', 'scripts', 'data', 'splits', 'logs', 'results',
                    'checkpoints', 'raw_results', 'reports/figures'):
            os.makedirs(os.path.join(dst, sub), exist_ok=True)
        for name in os.listdir(os.path.join(TEMPLATE, 'lib')):
            s = os.path.join(TEMPLATE, 'lib', name)
            if name == '__pycache__':
                continue
            d = os.path.join(dst, 'lib', name)
            shutil.copytree(s, d) if os.path.isdir(s) else shutil.copy2(s, d)
        for name in os.listdir(os.path.join(TEMPLATE, 'scripts')):
            shutil.copy2(os.path.join(TEMPLATE, 'scripts', name),
                         os.path.join(dst, 'scripts', name))
        shutil.copy2(os.path.join(TEMPLATE, 'requirements.txt'),
                     os.path.join(dst, 'requirements.txt'))
        shutil.copy2(os.path.join(TEMPLATE, 'run_all.sh'), os.path.join(dst, 'run_all.sh'))
        os.chmod(os.path.join(dst, 'run_all.sh'), 0o755)
        write_utmpaths(dst, D)
        shutil.copy2(os.path.join(ROOT, 'code', 'pkg_validate_package.py'),
                     os.path.join(dst, 'scripts', 'validate_package.py'))
        shutil.copy2(os.path.join(ROOT, 'code',
                                  'TIMESTAMP_AND_BOUNDARY_METHODOLOGY.md'),
                     os.path.join(dst, 'TIMESTAMP_AND_BOUNDARY_METHODOLOGY.md'))
        # literals hardcoded to the 14-class TCBench series
        pf = f'utmobilenet21_fastflow13_b2_ge50_first{{D}}s_9windows_8feat_3feat.parquet'
        for name, subs in (
            ('qa_check.py', [("check('n_classes always 14', set(ok.n_classes.unique()) == {14})",
                              "check(f'n_classes always {P.NUM_CLASS}', "
                              "set(ok.n_classes.unique()) == {P.NUM_CLASS})")]),
            ('verify_exhaustive.py', [
                ("f'utmobilenet21_capture14_first{D}s_9windows_8feat_3feat.parquet'",
                 "f'utmobilenet21_fastflow13_b2_ge50_first{D}s_9windows_8feat_3feat.parquet'")]),
            ('verify_features_independent.py', [
                ("f'utmobilenet21_capture14_first{D}s_9windows_8feat_3feat.parquet'",
                 "f'utmobilenet21_fastflow13_b2_ge50_first{D}s_9windows_8feat_3feat.parquet'")]),
            ('train_sweep.py', [('w_c = n_train_flows / (14 * train_flow_count_c)',
                                 'w_c = n_train_flows / (13 * train_flow_count_c)')]),
            ('smoke_test.py', [('produce (batch, 14)', 'produce (batch, NUM_CLASS)'),
                               ('create_model sees 14 labels',
                                'create_model sees every class')]),
        ):
            fp = os.path.join(dst, 'scripts', name)
            if not os.path.exists(fp):
                continue
            txt = open(fp).read()
            for a, b in subs:
                txt = txt.replace(a, b)
            open(fp, 'w').write(txt)
        # capture14.py is imported by name across the scripts; keep the name, retarget text
        cap = open(os.path.join(TEMPLATE, 'lib', 'capture14.py')).read()
        cap = cap.replace('capture-14 series', 'FastFlow-13 B2 >=50 series')
        cap = cap.replace('w_c = n_train_flows / (14 * train_flow_count_c)',
                          'w_c = n_train_flows / (13 * train_flow_count_c)')
        open(os.path.join(dst, 'lib', 'capture14.py'), 'w').write(cap)

        with open(os.path.join(dst, 'splits',
                               f'split_capture70_{D}s_13class.json'), 'w') as f:
            json.dump(rec, f, indent=1)
        meta[['flow_id', 'app', 'capture']].rename(
            columns={'flow_id': 'row_id'}).to_csv(
            os.path.join(dst, 'splits', 'flow_to_capture.csv'), index=False)

        df = build_parquet(meta, z_all[D], D)
        pq = os.path.join(dst, 'data',
                          f'utmobilenet21_fastflow13_b2_ge50_first{D}s_'
                          '9windows_8feat_3feat.parquet')
        df.to_parquet(pq, index=False)

        pid = dict(
            package=os.path.basename(dst), big_window_seconds=D, num_classes=13,
            packet_rule=('packet_count >= 50 (INCLUSIVE) over the COMPLETE B2-filtered '
                         'biflow; no class-count filter'),
            classes=classes, split_protocol='capture70', split_seed=SPLIT_SEED,
            split_fingerprint=rec['split_fingerprint'], weighted=True, deterministic=True,
            aggregation='mean_probability', eval_side='Validation',
            population_label=('FastFlow-style population with our TSLib capture-split and '
                              'capture-aggregation training protocol'),
            population_note=('The POPULATION follows FastFlow. The TRAINING PROTOCOL is '
                             'ours and is NOT claimed to be identical to FastFlow.'),
            source_archive='UTMobileNet2021.zip (all four partitions, raw CSVs)',
            source_archive_path='/media/Data/Datasets/UTMobileNet2021/Raw/UTMobileNet2021.zip',
            source_pkl_used=False,
            source_pkl_note=('The 34,378-flow pickle was deliberately NOT used; this '
                             'population was rebuilt from the raw CSV archive.'),
            timestamp_representation='int64 nanoseconds, exact, no floating point',
            boundary_rule=rec['boundary_rule'],
            duration_rule=f'complete flow duration STRICTLY greater than {D} s',
            population_gate=gate,
            n_flows=rec['n_flows'], n_captures=rec['n_captures'], runs=324,
            features=[dict(index=i, legacy_column_name=n, true_meaning=m)
                      for i, n, m in FEATURE_MEANING],
            feature_3feat_indices=FEAT3,
            feature_3feat_meaning='forward packet count, reverse packet count, '
                                  'payload-size ratio',
            direction_convention=(
                "Direction is FORWARD/REVERSE relative to the flow's FIRST packet. It is "
                'NOT client/server and NOT device/network. The legacy suffixes dir_-1 and '
                'dir_1 are inverted with respect to the pkts_dir value they hold; the '
                'values themselves are canonical and unchanged.'),
            feature_naming_status='legacy names retained unchanged for comparability',
        )
        with open(os.path.join(dst, 'PACKAGE_ID.json'), 'w') as f:
            json.dump(pid, f, indent=1)
        with open(os.path.join(dst, 'FEATURE_MANIFEST.json'), 'w') as f:
            json.dump(dict(n_features_8=8, n_features_3=3, feat_3_indices=FEAT3,
                           features=[dict(index=i, legacy_column_name=n,
                                          true_meaning=m)
                                     for i, n, m in FEATURE_MEANING],
                           timestamp_representation='int64 nanoseconds (exact)',
                           boundary_rule=rec['boundary_rule']), f, indent=1)

        summary.append(dict(D=D, dst=dst, flows=rec['n_flows'],
                            captures=rec['n_captures'],
                            tr_f=rec['n_flows_train'], te_f=rec['n_flows_test'],
                            tr_c=rec['n_captures_train'], te_c=rec['n_captures_test'],
                            fp=rec['split_fingerprint'], rows=len(df),
                            pq_sha=sha256_file(pq)))
        print(f'  [{D}s] {os.path.basename(dst)}')
        print(f'        flows {rec["n_flows"]:,} ({rec["n_flows_train"]:,} train / '
              f'{rec["n_flows_test"]:,} val)   captures {rec["n_captures"]:,} '
              f'({rec["n_captures_train"]:,}/{rec["n_captures_test"]:,})')
        print(f'        parquet rows {len(df):,}   fingerprint {rec["split_fingerprint"]}'
              f'   ({time.time() - t0:.0f}s)', flush=True)

    pd.DataFrame(summary).to_csv(os.path.join(REP, 'package_summary.csv'), index=False)
    print('\n  all three packages written')
    return 0


if __name__ == '__main__':
    sys.exit(main())
