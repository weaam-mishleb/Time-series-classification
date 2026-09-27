"""Build the UTMobileNet FastFlow-13 1s package from the verified reproduction artifacts.

Population rules (identical to the approved 3s/5s/8s family, only D changes):
  * all four partitions, 13 FastFlow apps (dropbox/hulu/pandora/skype excluded)
  * loopback removed at PACKET level (src or dst starting '127.')
  * sll.pkttype in {0,4}
  * TCP/UDP only with parsable addresses and ports
  * packet_count >= 50 over the COMPLETE filtered biflow
  * duration STRICTLY > 1 s; packets with elapsed <= 1 s
  * NO zero-padded admission of shorter flows
  * exact integer-nanosecond slot arithmetic (features() copied verbatim)

The 1s population is NOT paired with the 3s/5s/8s populations: it is a different, larger
eligible set, and it gets its own frozen capture-level split.
"""
import hashlib, json, os, shutil, sys, time
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split

REP = '/data/weaamm/utmobile_fastflow13_ge50_reproduction/reports'
TEMPLATE = ('/data/weaamm/utmobile_fastflow13_b2_ge50_5sec_capture70_weighted_det_'
            '6models_3seeds')
DST = ('/data/weaamm/utmobile_fastflow13_b2_ge50_1sec_capture70_weighted_det_'
       '6models_3seeds')
D = 1
D_NS = D * 10 ** 9
WINDOWS_MS = [20, 30, 40, 50, 75, 100, 150, 200, 250]
FEAT3 = [1, 4, 6]
SPLIT_SEED, TEST_SIZE = 42, 0.3

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


def features(t_ns, size, fwd, w_ns, T):
    """VERBATIM from the approved build_durations.py."""
    slot = np.minimum(t_ns // w_ns, T - 1).astype(np.int64)
    out = np.zeros((T, 8), dtype=np.float64)
    for mask, base in ((fwd, 0), (~fwd, 3)):
        if mask.any():
            s = slot[mask]
            tot = np.bincount(s, weights=size[mask], minlength=T)[:T]
            cnt = np.bincount(s, minlength=T)[:T].astype(np.float64)
            out[:, base] = tot; out[:, base + 1] = cnt
            out[:, base + 2] = np.divide(tot, cnt, out=np.zeros(T), where=cnt > 0)
    out[:, 6] = np.divide(out[:, 0], out[:, 3], out=np.zeros(T), where=out[:, 3] > 0)
    out[:, 7] = np.arange(T, dtype=np.float64) * (w_ns / 1e9)
    return out


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()


def main():
    if os.path.exists(DST):
        print(f'  REFUSING: {DST} already exists'); return 1
    t0 = time.time()
    idx = pd.read_parquet(f'{REP}/b2_flows_index.parquet')
    z = np.load(f'{REP}/b2_flow_packets_ns.npz')
    t_all, s_all, f_all, off = z['t_ns'], z['size'], z['fwd'], z['offsets']
    assert t_all.dtype == np.int64, 'timestamps must be int64 nanoseconds'
    idx['capture'] = idx.partition.astype(str) + '||' + idx.capture_id.astype(str)

    gate = dict(total_flows=int(len(idx)),
                per_app={k: int(v) for k, v in idx.app.value_counts().sort_index().items()},
                excluded_apps=['dropbox', 'hulu', 'pandora', 'skype'],
                total_packets=int(idx.packets.sum()),
                note=('B2 packet filter applied BEFORE biflow construction; '
                      'packet_count >= 50 inclusive over the COMPLETE filtered biflow'))
    if gate['total_flows'] != 12459:
        print(f"  FATAL population gate {gate['total_flows']} != 12459"); return 1
    n_eq = int((idx.duration_ns == D_NS).sum())
    elig = idx.index[idx.duration_ns > D_NS].to_numpy()          # STRICT >
    meta_src = idx.loc[elig].reset_index(drop=True)
    print(f'  gate {gate["total_flows"]:,} flows; duration EXACTLY {D}.0s: {n_eq}; '
          f'eligible (duration > {D}s): {len(meta_src):,}')
    classes = sorted(meta_src.app.unique().tolist())
    if len(classes) != 13:
        print(f'  FATAL {len(classes)} classes'); return 1

    # ---------------- features, all nine windows ----------------
    rows, feats = [], {w: [] for w in WINDOWS_MS}
    pkt_per_window = {w: 0 for w in WINDOWS_MS}
    float_mismatch = 0
    for fid in elig:
        a, b = off[fid], off[fid + 1]
        t_ns = t_all[a:b]
        keep = t_ns <= D_NS
        tu, sz, fw = t_ns[keep], s_all[a:b][keep], f_all[a:b][keep]
        r = idx.loc[fid]
        rows.append(dict(flow_id=int(fid), capture_id=r.capture_id, partition=r.partition,
                         app=r.app, proto=int(r.proto), capture=r.capture,
                         packets_full=int(r.packets), packets_in_D=int(keep.sum()),
                         payload_in_D=float(sz.sum()), fwd_in_D=int(fw.sum()),
                         rev_in_D=int((~fw).sum()), duration_s=float(r.duration_s),
                         duration_ns=int(r.duration_ns)))
        for w in WINDOWS_MS:
            w_ns = w * 1_000_000
            T = (D * 1000) // w + 1
            feats[w].append(features(tu, sz, fw, w_ns, T))
            pkt_per_window[w] += int(keep.sum())
            naive = np.minimum((tu / 1e9 / (w / 1000.0)).astype(np.int64), T - 1)
            exact = np.minimum(tu // w_ns, T - 1)
            float_mismatch += int((naive != exact).sum())
    meta = pd.DataFrame(rows)
    same_pop = len(set(pkt_per_window.values())) == 1
    print(f'  flows {len(meta):,}  captures {meta.capture.nunique():,}  '
          f'packets in D {int(meta.packets_in_D.sum()):,}')
    print(f'  identical packet population across all 9 windows: {same_pop} '
          f'({sorted(set(pkt_per_window.values()))})')
    print(f'  packets a naive float floor would misplace: {float_mismatch}')

    # ---------------- frozen capture-level split ----------------
    caps = meta.drop_duplicates('capture')[['capture', 'app']].sort_values('capture')
    tr_c, te_c = train_test_split(caps.capture.to_numpy(), test_size=TEST_SIZE,
                                  random_state=SPLIT_SEED, stratify=caps.app.to_numpy())
    tr_c, te_c = set(tr_c.tolist()), set(te_c.tolist())
    assert not (tr_c & te_c), 'capture leakage'
    is_tr = meta.capture.isin(tr_c).to_numpy()
    train_idx = [int(x) for x in meta.flow_id.to_numpy()[is_tr]]
    test_idx = [int(x) for x in meta.flow_id.to_numpy()[~is_tr]]
    assert not (set(train_idx) & set(test_idx)), 'flow overlap'
    y_all = np.searchsorted(np.array(classes), meta.app.to_numpy())
    fp = hashlib.sha256(y_all.astype(np.int64).tobytes()).hexdigest()[:16]
    tp = meta[is_tr].app.value_counts().to_dict(); vp = meta[~is_tr].app.value_counts().to_dict()
    if not all(tp.get(c, 0) > 0 and vp.get(c, 0) > 0 for c in classes):
        print('  FATAL: a class is missing from one side'); return 1
    rec = dict(
        protocol=('stratified 70/30 at CAPTURE level (partition||fname), stratified by '
                  'app, seed 42'),
        split_seed=SPLIT_SEED,
        filter_rule=('FastFlow-style population: B2 packet filter applied BEFORE biflow '
                     'construction; 13 FastFlow applications; packet_count >= 50 '
                     '(INCLUSIVE) over the COMPLETE filtered biflow; NO class-count '
                     'filter; complete duration STRICTLY > 1 s'),
        filter_source=('reproduced from the raw UTMobileNet2021 CSV archive, all four '
                       'partitions; the 34,378-flow pickle was NOT used'),
        boundary_rule=('slot = min(elapsed_ns // window_ns, T-1) in EXACT INTEGER '
                       'NANOSECONDS; a packet exactly on a boundary enters the UPPER '
                       'slot; a packet exactly at elapsed == D is included'),
        pairing_note=('This 1s population is NOT paired with the 3s/5s/8s populations. '
                      'It is a different, larger eligible set with its own frozen split.'),
        n_classes=13, classes=classes,
        n_captures=int(meta.capture.nunique()),
        n_captures_train=len(tr_c), n_captures_test=len(te_c),
        n_flows=int(len(meta)), n_flows_train=len(train_idx), n_flows_test=len(test_idx),
        train_captures_per_class=meta[is_tr].drop_duplicates('capture').app.value_counts().to_dict(),
        test_captures_per_class=meta[~is_tr].drop_duplicates('capture').app.value_counts().to_dict(),
        train_flows_per_class=tp, test_flows_per_class=vp,
        capture_leakage=0, flow_overlap=0, split_fingerprint=fp,
        train_idx=train_idx, test_idx=test_idx)

    # ---------------- assemble the package ----------------
    for sub in ('lib', 'scripts', 'data', 'splits', 'logs', 'results', 'checkpoints',
                'raw_results', 'reports/figures'):
        os.makedirs(os.path.join(DST, sub), exist_ok=True)
    for name in os.listdir(f'{TEMPLATE}/lib'):
        if name == '__pycache__':
            continue
        s = f'{TEMPLATE}/lib/{name}'; d = f'{DST}/lib/{name}'
        shutil.copytree(s, d) if os.path.isdir(s) else shutil.copy2(s, d)
    for name in os.listdir(f'{TEMPLATE}/scripts'):
        shutil.copy2(f'{TEMPLATE}/scripts/{name}', f'{DST}/scripts/{name}')
    for name in ('requirements.txt', 'run_all.sh', 'METHODOLOGY_CORRECTION.md',
                 'TIMESTAMP_AND_BOUNDARY_METHODOLOGY.md'):
        if os.path.exists(f'{TEMPLATE}/{name}'):
            shutil.copy2(f'{TEMPLATE}/{name}', f'{DST}/{name}')
    os.chmod(f'{DST}/run_all.sh', 0o755)
    # utmpaths: only the parquet/split filenames are duration-templated already
    src = open(f'{TEMPLATE}/lib/utmpaths.py').read()
    src = src.replace("f'split_capture70_{BIG_WINDOW_S}s_13class.json'",
                      "f'split_capture70_{BIG_WINDOW_S}s_13class.json'")
    open(f'{DST}/lib/utmpaths.py', 'w').write(src)

    with open(f'{DST}/splits/split_capture70_{D}s_13class.json', 'w') as f:
        json.dump(rec, f, indent=1)
    meta[['flow_id', 'app', 'capture']].rename(columns={'flow_id': 'row_id'}).to_csv(
        f'{DST}/splits/flow_to_capture.csv', index=False)

    out = []
    for w in WINDOWS_MS:
        T = (D * 1000) // w + 1
        A = np.stack(feats[w]).astype(np.float64)
        assert A.shape == (len(meta), T, 8)
        for i, r in enumerate(meta.itertuples()):
            f8 = A[i]
            out.append(dict(row_id=int(r.flow_id), app=r.app, partition=r.partition,
                            fname=r.capture_id, capture=r.capture, ip_proto=int(r.proto),
                            total_packets=int(r.packets_full),
                            packets_in_window=int(r.packets_in_D),
                            payload_in_window=float(r.payload_in_D),
                            flow_length_seconds=float(r.duration_s),
                            flow_length_ns=int(r.duration_ns),
                            big_window_s=float(D), small_window_ms=int(w), T=int(T),
                            n_features_8=8, n_features_3=3,
                            feat_8=f8.reshape(-1).astype(np.float64),
                            feat_3=f8[:, FEAT3].reshape(-1).astype(np.float64)))
    df = pd.DataFrame(out)
    pq = (f'{DST}/data/utmobilenet21_fastflow13_b2_ge50_first{D}s_'
          '9windows_8feat_3feat.parquet')
    df.to_parquet(pq, index=False)
    print(f'  parquet rows {len(df):,} -> {os.path.basename(pq)}')

    pid = dict(
        package=os.path.basename(DST), big_window_seconds=D, num_classes=13,
        packet_rule=('packet_count >= 50 (INCLUSIVE) over the COMPLETE B2-filtered '
                     'biflow; no class-count filter'),
        classes=classes, split_protocol='capture70', split_seed=SPLIT_SEED,
        split_fingerprint=fp, weighted=True, deterministic=True,
        aggregation='mean_probability', eval_side='Validation',
        population_label=('FastFlow-style population with our TSLib capture-split and '
                          'capture-aggregation training protocol'),
        population_note=('The POPULATION follows FastFlow. The TRAINING PROTOCOL is ours '
                         'and is NOT claimed to be identical to FastFlow.'),
        pairing_note=rec['pairing_note'],
        source_archive='UTMobileNet2021.zip (all four partitions, raw CSVs)',
        source_archive_path='/media/Data/Datasets/UTMobileNet2021/Raw/UTMobileNet2021.zip',
        source_pkl_used=False,
        source_pkl_note=('The 34,378-flow pickle was deliberately NOT used; this '
                         'population was rebuilt from the raw CSV archive.'),
        timestamp_representation='int64 nanoseconds, exact, no floating point',
        boundary_rule=rec['boundary_rule'],
        duration_rule=f'complete flow duration STRICTLY greater than {D} s',
        grad_clip='none (matches the final UTMobile protocol)',
        population_gate=gate, n_flows=rec['n_flows'], n_captures=rec['n_captures'],
        runs=324,
        features=[dict(index=i, legacy_column_name=nm, true_meaning=m)
                  for i, nm, m in FEATURE_MEANING],
        feature_3feat_indices=FEAT3,
        feature_3feat_meaning='forward packet count, reverse packet count, payload-size ratio',
        direction_convention=(
            "Direction is FORWARD/REVERSE relative to the flow's FIRST packet. It is NOT "
            'client/server and NOT device/network. The legacy suffixes dir_-1 and dir_1 '
            'are inverted with respect to the pkts_dir value they hold; the values '
            'themselves are canonical and unchanged.'),
        feature_naming_status='legacy names retained unchanged for comparability')
    json.dump(pid, open(f'{DST}/PACKAGE_ID.json', 'w'), indent=1)
    json.dump(dict(n_features_8=8, n_features_3=3, feat_3_indices=FEAT3,
                   features=[dict(index=i, legacy_column_name=nm, true_meaning=m)
                             for i, nm, m in FEATURE_MEANING],
                   timestamp_representation='int64 nanoseconds (exact)',
                   boundary_rule=rec['boundary_rule']),
              open(f'{DST}/FEATURE_MANIFEST.json', 'w'), indent=1)

    print(f'\n  split fingerprint {fp}')
    print(f'  flows    train {rec["n_flows_train"]:,} / val {rec["n_flows_test"]:,}')
    print(f'  captures train {rec["n_captures_train"]:,} / val {rec["n_captures_test"]:,}  '
          f'(train share {rec["n_captures_train"]/rec["n_captures"]:.4f})')
    print(f'  parquet sha256 {sha256_file(pq)[:32]}')
    print(f'  done in {time.time()-t0:.0f}s')
    return 0


if __name__ == '__main__':
    sys.exit(main())
