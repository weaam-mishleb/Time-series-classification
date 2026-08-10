"""
Single-batch smoke for CESNET: forward + loss + backward, three models, two windows.

No epoch, no sweep. Measures the thing that actually decides feasibility — peak GPU
memory at seq_len=1000 (the 5 ms window) versus seq_len=21 (250 ms).

GPU 0 only. Reads data read-only; writes one CSV to scratch.
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import TimeSeries_functions as ts

MODELS = ['timesnet', 'informer', 'nst']


def one(model_type, X, y, window, variant, batch_size):
    model, cfg = ts.create_model(X, y, config_type=model_type,
                                 dataset_name='Cesnet', small_window=window)
    model = model.cuda().train()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    xb = torch.tensor(X[:batch_size], dtype=torch.float32).cuda()
    yb = torch.tensor(y[:batch_size], dtype=torch.long).cuda()
    crit = nn.CrossEntropyLoss()

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = model(x_enc=xb, x_mark_enc=None, x_dec=None, x_mark_dec=None)
    loss = crit(out, yb)
    loss.backward()
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    grads = [p.grad for p in model.parameters() if p.grad is not None]
    gnan = any(bool(torch.isnan(g).any() or torch.isinf(g).any()) for g in grads)
    peak = torch.cuda.max_memory_allocated() / 2**20

    r = dict(variant=variant, window=window, seq_len=cfg.seq_len, model=model_type,
             batch_size=batch_size, num_class=cfg.num_class,
             params=sum(p.numel() for p in model.parameters()),
             out_shape=str(tuple(out.shape)),
             out_nan=bool(torch.isnan(out).any() or torch.isinf(out).any()),
             loss=float(loss.item()), loss_finite=bool(np.isfinite(loss.item())),
             grads_with_nan=gnan, n_grad_tensors=len(grads),
             peak_gpu_mib=round(peak, 1), fwd_bwd_s=round(dt, 3))
    del model, xb, yb, out, loss
    torch.cuda.empty_cache()
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', required=True, choices=['balanced', 'dir'])
    ap.add_argument('--windows', nargs='+', default=['250ms', '5ms'])
    ap.add_argument('--batch_size', type=int, default=256)
    ap.add_argument('--max_rows', type=int, default=None,
                    help='cap rows read (dir/5ms cannot be fully loaded in RAM)')
    args = ap.parse_args()

    assert torch.cuda.is_available(), 'no CUDA visible'
    print(f"device: {torch.cuda.get_device_name(0)}  "
          f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")

    rows = []
    for w in args.windows:
        if args.max_rows:
            # read a capped slice directly; prepare_datasets has no row limit
            sub = ('balanced_direction_datasets/balanced_dir_cesnet_timeseries_%s.csv'
                   if args.variant == 'balanced'
                   else 'direction_datasets/dir_cesnet_timeseries_%s.csv') % w
            p = os.path.join(ts.DATA_ROOT, 'Cesnet', 'data', sub)
            t0 = time.time()
            d = pd.read_csv(p, nrows=args.max_rows)
            y = d['APP'].to_numpy()
            Xf = d.drop(columns=['APP']).to_numpy(np.float64)
            T = Xf.shape[1] // 8
            X = Xf.reshape(len(d), T, 8)
            from sklearn.preprocessing import LabelEncoder
            y = LabelEncoder().fit_transform(y)
            print(f"\n{w}: capped read of {len(d):,} rows in {time.time()-t0:.0f}s -> {X.shape}")
        else:
            d = ts.prepare_datasets(dataset_name='Cesnet', small_windows=[w],
                                    dataset_type=args.variant,
                                    selected_feature_indices=[0, 1, 2, 3, 4, 5, 6, 7])
            if not d:
                print(f"{w}: NOT AVAILABLE"); continue
            k, (X, y) = list(d.items())[0]
            print(f"\n{w}: {k} {X.shape}")

        for m in MODELS:
            try:
                r = one(m, X, y, w, args.variant, args.batch_size)
                rows.append(r)
                print(f"  {m:<9} seq_len={r['seq_len']:<5} out={r['out_shape']:<12} "
                      f"loss={r['loss']:.4f} nan={r['out_nan']} grad_nan={r['grads_with_nan']} "
                      f"peak={r['peak_gpu_mib']:>8.1f} MiB  {r['fwd_bwd_s']:.3f}s")
            except RuntimeError as e:
                rows.append(dict(variant=args.variant, window=w, model=m,
                                 error=str(e)[:200]))
                print(f"  {m:<9} FAILED: {str(e)[:150]}")
                torch.cuda.empty_cache()
        del X, y

    work = os.environ.get('TS_CESNET_WORK', '/data/weaamm/cesnet_work')
    os.makedirs(work, exist_ok=True)
    out = os.path.join(work, f'cesnet_smoke_{args.variant}.csv')
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == '__main__':
    main()
