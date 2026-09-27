"""ZeroRatio per window, per-app train/validation counts, and file hashes."""
import hashlib, json, os, sys
import numpy as np, pandas as pd

DUR = [3, 5, 8]
W = ['20ms', '30ms', '40ms', '50ms', '75ms', '100ms', '150ms', '200ms', '250ms']


def pkg(D):
    return ('/data/weaamm/utmobile_fastflow13_b2_ge50_'
            f'{D}sec_capture70_weighted_det_6models_3seeds')


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()


def main():
    zrows, arows, hrows = [], [], []
    for D in DUR:
        root = pkg(D)
        sys.path.insert(0, os.path.join(root, 'lib'))
        os.environ['UTMPKG_ROOT'] = root
        for m in ('utmpaths', 'capture14'):
            sys.modules.pop(m, None)
        import utmpaths as P; import capture14 as C
        C._CACHE.clear()
        rec = json.load(open(P.split_path()))
        for w in W:
            for cfg in ('8feat', '3feat'):
                X, _ = C.load_window(w, cfg)
                zrows.append(dict(duration_s=D, window=w, config=cfg,
                                  flows=X.shape[0], T=X.shape[1], F=X.shape[2],
                                  zero_ratio=round(float((X == 0).mean()), 6),
                                  all_zero_slots=int((X.sum(axis=2) == 0).sum()),
                                  min=float(X.min()), max=float(X.max())))
        tp, vp = rec['train_flows_per_class'], rec['test_flows_per_class']
        tc, vc = rec['train_captures_per_class'], rec['test_captures_per_class']
        for a in rec['classes']:
            arows.append(dict(duration_s=D, app=a,
                              train_flows=tp.get(a, 0), val_flows=vp.get(a, 0),
                              train_captures=tc.get(a, 0), val_captures=vc.get(a, 0),
                              total_flows=tp.get(a, 0) + vp.get(a, 0)))
        for rel in ('PACKAGE_ID.json', 'FEATURE_MANIFEST.json', 'manifest.json',
                    os.path.relpath(P.parquet_path(), root),
                    os.path.relpath(P.split_path(), root),
                    'splits/flow_to_capture.csv',
                    'TIMESTAMP_AND_BOUNDARY_METHODOLOGY.md'):
            p = os.path.join(root, rel)
            if os.path.exists(p):
                hrows.append(dict(duration_s=D, file=rel, bytes=os.path.getsize(p),
                                  sha256=sha256(p)))
        sys.path.remove(os.path.join(root, 'lib'))

    z = pd.DataFrame(zrows); a = pd.DataFrame(arows); h = pd.DataFrame(hrows)
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'reports')
    z.to_csv(os.path.join(out, 'acceptance_zero_ratio.csv'), index=False)
    a.to_csv(os.path.join(out, 'acceptance_per_app.csv'), index=False)
    h.to_csv(os.path.join(out, 'acceptance_hashes.csv'), index=False)
    for D in DUR:
        pd.DataFrame(z[z.duration_s == D]).to_csv(
            os.path.join(pkg(D), 'reports', 'zero_ratio_by_window.csv'), index=False)
        pd.DataFrame(a[a.duration_s == D]).to_csv(
            os.path.join(pkg(D), 'reports', 'per_app_train_val.csv'), index=False)
        pd.DataFrame(h[h.duration_s == D]).to_csv(
            os.path.join(pkg(D), 'reports', 'file_hashes.csv'), index=False)

    print('=== ZeroRatio, 8feat (fraction of zero values in the tensor) ===')
    p8 = z[z.config == '8feat'].pivot(index='window', columns='duration_s',
                                      values='zero_ratio').reindex(W)
    print(p8.to_string())
    print('\n=== ZeroRatio, 3feat ===')
    p3 = z[z.config == '3feat'].pivot(index='window', columns='duration_s',
                                      values='zero_ratio').reindex(W)
    print(p3.to_string())
    print('\n=== per-app train / validation flows ===')
    piv = a.pivot(index='app', columns='duration_s',
                  values=['train_flows', 'val_flows'])
    print(piv.to_string())
    print(f'\n  wrote acceptance_zero_ratio.csv / acceptance_per_app.csv / '
          f'acceptance_hashes.csv  ({len(h)} hashed files)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
