"""Atomically (re)build a per-shard CESNET ledger from artifacts on disk.

A run counts OK only when its detailed result CSV exists AND its checkpoint loads.
Written to a temp file and os.replace'd, so a reader never sees a partial ledger.
"""
import glob, os, re, sys, tempfile
sys.path.insert(0, '/home/weaamm/Time-series-classifction/Time-Series-Library')
import pandas as pd, torch

C = '/data/weaamm/cesnet_clean'
MODELS = ['timesnet', 'informer', 'nst', 'autoformer', 'fedformer', 'timemixer']
WINDOWS = ['20ms', '30ms', '40ms', '50ms', '75ms', '100ms', '150ms', '200ms', '250ms']
W2SL = {'20ms':250,'30ms':167,'40ms':125,'50ms':100,'75ms':67,
        '100ms':50,'150ms':34,'200ms':25,'250ms':21}


def build(cfg, out):
    res = f'{C}/prod_results_{cfg}_seed0/Cesnet/direction'
    ckd = f'{C}/prod_ckpt_{cfg}_seed0/Cesnet/18'
    seen = {}
    for p in sorted(glob.glob(f'{res}/*_detailed.csv')):
        for _, r in pd.read_csv(p).iterrows():
            seen[(str(r['Model']).strip(),
                  re.sub(r'^dir_Cesnet_', '', str(r['Dataset']).strip()))] = (r, os.path.basename(p))
    rows = []
    for m in MODELS:
        for w in WINDOWS:
            sl = W2SL[w]
            cp = f'{ckd}/{m}_classifier_{sl}_best_val.pth'
            hit = seen.get((m, w))
            loads = False
            if os.path.exists(cp):
                try:
                    o = torch.load(cp, map_location='cpu', weights_only=False)
                    sd = o.state_dict() if hasattr(o, 'state_dict') else (
                        o.get('model_state_dict', o) if isinstance(o, dict) else None)
                    loads = bool(sd is not None and len(sd) > 0)
                except Exception:
                    loads = False
            ok = hit is not None and loads
            r = hit[0] if hit else None
            rows.append(dict(
                run_id=f'CESNET5s_dir_{cfg}_{m}_{w}_seed0', dataset='Cesnet',
                dataset_type='dir', big_window_s=5, feature_config=cfg, model=m,
                window=w, training_seed=0, seq_len=sl, num_class=18,
                val_acc=r['Val Accuracy'] if hit else '', val_f1=r['Val F1'] if hit else '',
                train_acc=r['Train Accuracy'] if hit else '',
                train_f1=r['Train F1'] if hit else '',
                training_time=r['Training Time'] if hit else '',
                result_csv=hit[1] if hit else '',
                checkpoint=os.path.relpath(cp, C) if os.path.exists(cp) else '',
                checkpoint_bytes=os.path.getsize(cp) if os.path.exists(cp) else 0,
                checkpoint_loads=loads, has_result_csv=hit is not None,
                status='OK' if ok else 'PENDING', error=''))
    df = pd.DataFrame(rows)
    assert df.run_id.nunique() == len(df) == 54, (df.run_id.nunique(), len(df))
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(out), suffix='.tmp')
    os.close(fd)
    df.to_csv(tmp, index=False)
    os.replace(tmp, out)
    return df


if __name__ == '__main__':
    cfg = sys.argv[1]
    df = build(cfg, f'{C}/ledger/ledger_{cfg}_seed0.csv')
    ok = int((df.status == 'OK').sum())
    print(f'  ledger_{cfg}_seed0.csv: {len(df)} rows, {df.run_id.nunique()} unique, '
          f'{ok} OK, {len(df)-ok} pending')
