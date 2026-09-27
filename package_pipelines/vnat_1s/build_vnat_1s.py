"""Build the VNAT TCP-only duration-filtered 1s package.

Reads the release H5 READ-ONLY. Filter order, verbatim from the approved final builder:
  1. original VNAT H5 source
  2. remove VoIP
  3. keep Chat, Control, File Transfer, Streaming
  4. packet_count >= 10 over the COMPLETE flow
  5. TCP only (ip_proto == 6)
  6. duration >= 1 s, inclusive
  7. packets from the first second only (elapsed <= D, applied BEFORE slotting)

Short flows are NOT admitted and terminal zero-padding is NOT used. Empty slots inside a
retained flow occur naturally and are all-zero.

VPN: TCP-only removes every OpenVPN-over-UDP flow, so this package contains ZERO VPN flows.

SPLIT: the frozen master membership used by the 3s/5s/8s packages lived in a pilot
directory that has since been deleted, and flows in [1s, 3s) appear in no surviving
package, so their master-split side is unrecoverable. A NEW flow-level 70/30 split is
therefore generated and frozen here, stratified on TRAFFIC CLASS ONLY, seed 42. It is NOT
paired with the old per-duration splits.
"""
import hashlib, json, os, sys, time
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split

SRC = '/data/natan/datasets/vnat/VNAT_Dataframe_release_1.h5'
DST = '/data/weaamm/vnat_tcp_duration_1sec_weighted_det_6models_3seeds'
WINDOWS_MS = [20, 30, 40, 50, 75, 100, 150, 200, 250]
D_MS, MIN_PKTS, UP, TCP = 1000, 10, 1, 6
D = D_MS / 1000.0
CLASSES = ['Chat', 'Control', 'File Transfer', 'Streaming']
CLASS_ID = {c: i for i, c in enumerate(CLASSES)}
APP_TO_CLASS = {'vimeo': 'Streaming', 'netflix': 'Streaming', 'youtube': 'Streaming',
                'skype-chat': 'Chat', 'ssh': 'Control', 'rdp': 'Control',
                'sftp': 'File Transfer', 'rsync': 'File Transfer', 'scp': 'File Transfer'}


def features(rel, sizes, dirs, w_s, T):
    """VERBATIM from the approved VNAT builder."""
    slot = (rel // w_s).astype(np.int64)
    k = (slot >= 0) & (slot < T)
    assert k.all(), 'a packet inside D fell outside the slot grid'
    slot, sz, up = slot[k], sizes[k].astype(np.float64), dirs[k] == UP
    cu = np.bincount(slot[up], minlength=T).astype(np.float64)
    su = np.bincount(slot[up], weights=sz[up], minlength=T)
    cd = np.bincount(slot[~up], minlength=T).astype(np.float64)
    sd = np.bincount(slot[~up], weights=sz[~up], minlength=T)
    out = np.zeros((T, 8))
    out[:, 0], out[:, 1] = su, cu
    out[:, 2] = np.divide(su, cu, out=np.zeros(T), where=cu > 0)
    out[:, 3], out[:, 4] = sd, cd
    out[:, 5] = np.divide(sd, cd, out=np.zeros(T), where=cd > 0)
    out[:, 6] = np.divide(su, sd, out=np.zeros(T), where=sd > 0)
    out[:, 7] = np.arange(T) * w_s
    return out


def main():
    if os.path.exists(f'{DST}/data/labels.npy'):
        print('  REFUSING: data already present'); return 1
    t0 = time.time()
    print(f'  reading {SRC} (read-only)', flush=True)
    df = pd.read_hdf(SRC, key='data')
    print(f'    {len(df):,} rows in {time.time()-t0:.0f}s', flush=True)

    fn = df['file_names'].astype(str)
    parts = fn.str.replace(r'\.pcap$', '', regex=True).str.split('_')
    marker = parts.str[0].to_numpy()
    app = parts.str[1].str.lower().to_numpy()
    npk = df['sizes'].map(len).to_numpy()
    cls = pd.Series(app).map(APP_TO_CLASS).to_numpy()
    proto = np.array([int(c[4]) for c in df['connection']])

    print('\n  POPULATION FUNNEL')
    print(f'    1. rows in the H5                      {len(df):>8,}')
    print(f'    2. VoIP removed                        {int((app=="voip").sum()):>8,} removed')
    m_cls = pd.notna(pd.Series(cls)).to_numpy()
    print(f'    3. mapped to the four classes          {int(m_cls.sum()):>8,}')
    m_pkt = m_cls & (npk >= MIN_PKTS)
    print(f'    4. packet_count >= {MIN_PKTS} (complete flow)   {int(m_pkt.sum()):>8,}')
    m_tcp = m_pkt & (proto == TCP)
    n_vpn_removed = int((m_pkt & (proto != TCP) & (marker == 'vpn')).sum())
    n_vpn_kept = int((m_tcp & (marker == 'vpn')).sum())
    print(f'    5. TCP only (ip_proto == 6)            {int(m_tcp.sum()):>8,}'
          f'  ({int(m_pkt.sum()-m_tcp.sum())} non-TCP removed)')
    print(f'       VPN flows removed by TCP-only       {n_vpn_removed:>8,}')
    print(f'       VPN flows REMAINING                 {n_vpn_kept:>8,}  <-- must be 0')
    sel = np.where(m_tcp)[0]

    ts = [np.asarray(df['timestamps'].iloc[i], dtype=np.float64) for i in sel]
    sz = [np.asarray(df['sizes'].iloc[i], dtype=np.int64) for i in sel]
    dr = [np.asarray(df['directions'].iloc[i], dtype=np.int64) for i in sel]
    conn = [df['connection'].iloc[i] for i in sel]
    dur = np.array([float(t.max() - t.min()) for t in ts])
    fwd = np.array([int(d[int(np.argmin(t))]) for t, d in zip(ts, dr)])
    assert (fwd == UP).all(), 'first-packet direction is not uniformly 1'
    del df

    base = pd.DataFrame({
        'h5_row': sel, 'src_ip': [c[0] for c in conn], 'src_port': [int(c[1]) for c in conn],
        'dst_ip': [c[2] for c in conn], 'dst_port': [int(c[3]) for c in conn],
        'ip_proto': [int(c[4]) for c in conn], 'capture_file': fn.to_numpy()[sel],
        'app': app[sel], 'vpn_status': marker[sel], 'label': cls[sel],
        'n_packets': npk[sel], 'duration': dur,
        'n_up': [int((d == UP).sum()) for d in dr],
        'n_down': [int((d != UP).sum()) for d in dr]})

    take = np.where(dur >= D)[0]                       # `>=`, inclusive
    n_eq = int((dur == D).sum()); n_strict = int((dur > D).sum())
    m = base.iloc[take].reset_index(drop=True)
    m.insert(0, 'flow_id', np.arange(len(m)))
    n = len(m)
    print(f'\n    6. duration >= {D:g}s                     {n:>8,}')
    print(f'       flows with duration EXACTLY {D:g}s     {n_eq:>8}')
    print(f'       `> D` would give {n_strict}, `>= D` gives {n} -> '
          f'{"identical set" if n_strict == n else "DIFFERENT sets"}')
    print(f'       nearest above {float(dur[dur>=D].min()):.6f}s  '
          f'nearest below {float(dur[dur<D].max()):.6f}s')
    for c in CLASSES:
        print(f'       {c:<16}{int((m.label==c).sum()):>6}')
    problems = []
    if int((m.ip_proto != TCP).sum()): problems.append('non-TCP retained')
    if int((m.vpn_status == 'vpn').sum()): problems.append('VPN retained')
    if int((m.n_packets < MIN_PKTS).sum()): problems.append('flow with < 10 packets retained')
    if float(m.duration.min()) < D: problems.append('flow shorter than D retained')
    if problems:
        print('\n  STOPPING BEFORE ANY WRITE:'); [print('   ' + p) for p in problems]; return 2

    os.makedirs(f'{DST}/data', exist_ok=True); os.makedirs(f'{DST}/splits', exist_ok=True)
    y = m['label'].map(CLASS_ID).to_numpy().astype(np.int64)
    idx_all = np.arange(n)
    tr, va = train_test_split(idx_all, test_size=0.3, random_state=42, stratify=y)
    tr, va = np.sort(tr).astype(np.int64), np.sort(va).astype(np.int64)
    assert not (set(tr.tolist()) & set(va.tolist())), 'train/validation overlap'
    assert len(np.unique(y[tr])) == 4 and len(np.unique(y[va])) == 4, 'a class is missing'

    inwin = [(ts[j] - ts[j].min()) <= D for j in take]
    n_in = np.array([int(k.sum()) for k in inwin])
    for w in WINDOWS_MS:
        T, w_s = D_MS // w + 1, w / 1000.0
        X = np.zeros((n, T, 8), dtype=np.float64)
        for r, j in enumerate(take):
            rel = ts[j] - ts[j].min(); near = inwin[r]
            X[r] = features(rel[near], sz[j][near], dr[j][near], w_s, T)
        assert np.isfinite(X).all() and (X >= 0).all()
        assert abs(X[:, :, 1].sum() + X[:, :, 4].sum() - n_in.sum()) < 1e-6
        np.save(f'{DST}/data/X_{w}ms.npy', X)
        print(f'   X_{w}ms.npy  {X.shape}  '
              f'{os.path.getsize(f"{DST}/data/X_{w}ms.npy")/2**20:.1f} MB', flush=True)
        del X
    np.save(f'{DST}/data/labels.npy', y)
    np.save(f'{DST}/data/groups.npy', m['capture_file'].to_numpy())
    m.to_csv(f'{DST}/data/flows.csv', index=False)

    lab = m.label.to_numpy()
    fp = hashlib.sha256(np.concatenate([tr, va]).tobytes()).hexdigest()[:16]
    label_fp = hashlib.sha256(y.tobytes()).hexdigest()[:16]
    sj = dict(dataset='VNAT', variant='VNAT_1s_tcp_duration', big_window_s=1,
              protocol='random70_stratified_flowlevel_splitseed42',
              strategy=('flow level, stratified on TRAFFIC CLASS ONLY (never on VPN '
                        'status or application identity). NEWLY GENERATED for 1s and '
                        'frozen here.'),
              split_derivation=('NEW. The frozen master membership used by the 3s/5s/8s '
                                'packages lived in a pilot directory that was deleted as '
                                'superseded work, and flows in [1s,3s) appear in no '
                                'surviving package, so their side is unrecoverable. This '
                                'split is NOT paired with the old per-duration splits.'),
              split_seed=42, test_fraction=0.3, n_samples=n, n_train=int(len(tr)),
              n_test=int(len(va)), classes=4, class_names=CLASSES,
              train_per_class={c: int((lab[tr] == c).sum()) for c in CLASSES},
              test_per_class={c: int((lab[va] == c).sum()) for c in CLASSES},
              train_per_vpn={k: int(v) for k, v in
                             pd.Series(m.vpn_status.to_numpy()[tr]).value_counts().items()},
              test_per_vpn={k: int(v) for k, v in
                            pd.Series(m.vpn_status.to_numpy()[va]).value_counts().items()},
              train_per_app={k: int(v) for k, v in
                             pd.Series(m.app.to_numpy()[tr]).value_counts().items()},
              test_per_app={k: int(v) for k, v in
                            pd.Series(m.app.to_numpy()[va]).value_counts().items()},
              all_classes_both_sides=True,
              grouping='none - flow-level split; captures may appear on both sides by design',
              label_fingerprint=label_fp, split_fp=fp,
              used_by='all 9 windows, both feature configs, all 6 models, all 3 seeds',
              train_idx=[int(x) for x in tr], test_idx=[int(x) for x in va])
    json.dump(sj, open(f'{DST}/splits/split_random70_splitseed42.json', 'w'), indent=1)
    print(f'\n  split_fp {fp}   label_fp {label_fp}')
    print(f'  train {len(tr):,} / validation {len(va):,}  of {n:,}')
    print(f'  train per class {sj["train_per_class"]}')
    print(f'  test  per class {sj["test_per_class"]}')
    print(f'  vpn: train {sj["train_per_vpn"]}  test {sj["test_per_vpn"]}')
    json.dump(dict(total_flows=n, per_class={c: int((lab == c).sum()) for c in CLASSES},
                   vpn_flows=int((m.vpn_status == 'vpn').sum()),
                   nonvpn_flows=int((m.vpn_status == 'nonvpn').sum()),
                   boundary=dict(rule=f'duration >= {D:g}, inclusive',
                                 flows_exactly_equal_to_D=n_eq,
                                 count_with_strict_gt=n_strict, identical=n_strict == n)),
              open('/data/weaamm/vnat_1s_build/population_summary_1s.json', 'w'), indent=1)
    print(f'  done in {time.time()-t0:.0f}s')
    return 0


if __name__ == '__main__':
    sys.exit(main())
