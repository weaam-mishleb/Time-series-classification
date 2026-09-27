"""
Inference cost benchmark for the UTMobileNet Task-1 models.

Measures forward-pass latency and throughput for TimesNet, Informer and NST at each
model's best observed window, on GPU and CPU. Nothing here trains, and nothing here
touches A2/B2, the frozen split, the checkpoints or the Task-1 results — the datasets
and checkpoints are opened read-only.

What is and is not timed
------------------------
Timed:     the model forward pass alone.
Not timed: CSV parsing, standardization, host-to-device transfer. The input batch is
           materialised on the target device once, before the warmup, so the measured
           interval contains no data movement.

GPU timing brackets every forward with torch.cuda.synchronize(), because CUDA kernel
launches are asynchronous and an unsynchronised timer measures launch overhead rather
than execution.

Model construction mirrors the training path exactly: TimeSeriesConfig with the same
(model_type, num_class, seq_len, num_features) the run used, then the same model class,
then the checkpoint's own model_state_dict with strict=True. The config carries no seed,
so architecture — and therefore latency — is independent of which seed produced the
weights; see --verify-arch.

Usage:
    CUDA_VISIBLE_DEVICES=0 python benchmark_inference.py --smoke
    CUDA_VISIBLE_DEVICES=0 python benchmark_inference.py
"""
import argparse
import json
import os
import platform
import subprocess
import time

import numpy as np
import pandas as pd
import torch

import TimeSeries_functions as ts
import splits
from models.TimesNet import Model as TimesNet
from models.Informer import Model as Informer
from models.Nonstationary_Transformer import Model as NST

# Output directory. Overridable so the same script can run on a machine that has no
# /data/weaamm — same pattern as TS_DATA_ROOT / TS_CHECKPOINTS_ROOT / TS_SPLITS_DIR.
# Affects only where the results CSV is written; no measurement path depends on it.
OUT = os.environ.get('TS_BENCH_OUT', '/data/weaamm/inference_benchmark')
CKPT_DIR = os.path.join(ts.CHECKPOINTS_ROOT, 'UTMobileNet', '15')
BUILDERS = {'timesnet': TimesNet, 'informer': Informer, 'nst': NST}

# Best observed window per model from the Task-1 sweep (8 features, mean over seeds).
TARGETS = [
    # model,      window,  label
    ('timesnet', '200ms', 'TimesNet'),
    ('informer', '250ms', 'Informer'),
    ('nst',      '200ms', 'NST'),
]
BATCH_SIZES = [1, 32, 128]
N_WARMUP = 20
N_ITERS = 200
FEATURES_8 = [0, 1, 2, 3, 4, 5, 6, 7]


# =====================================================================================
# Data: real A2 validation flows, standardized with train-only statistics
# =====================================================================================
def val_tensor(window):
    """Return the standardized validation split for `window` as a float32 tensor.

    Reproduces prepare_data's transform exactly: the frozen a2grouped split is applied
    first, then mu/sigma are computed on the TRAIN rows only and applied to validation.
    No validation statistic ever reaches the training arm, so there is no leakage.
    """
    d = ts.prepare_datasets(dataset_name='UTMobileNet', small_windows=[window],
                            dataset_type='dir', selected_feature_indices=FEATURES_8)
    if not d:
        raise SystemExit(f"no data for window {window} under {ts.DATA_ROOT}")
    X, y = list(d.values())[0]
    train_idx, val_idx = splits.get_or_create_split(
        y, dataset_name='UTMobileNet', num_class=15,
        tag=os.environ.get('TS_SPLIT_TAG', 'a2grouped'), verbose=False)
    X_train, X_val = X[train_idx], X[val_idx]
    mu = X_train.mean(axis=(0, 1), keepdims=True)
    sigma = X_train.std(axis=(0, 1), keepdims=True)
    sigma = np.where(sigma < 1e-8, 1.0, sigma)
    X_val = (X_val - mu) / sigma
    return torch.tensor(np.ascontiguousarray(X_val), dtype=torch.float32), len(train_idx)


# =====================================================================================
# Model
# =====================================================================================
def build_model(model_type, seq_len, num_features=8, num_class=15, ckpt=None,
                device='cpu'):
    """Rebuild a trained classifier exactly as the training path did, then load weights."""
    cfg = ts.TimeSeriesConfig(
        model_type=model_type, num_class=num_class, seq_len=seq_len,
        num_features=num_features,
        classes_names=[f'Class_{i}' for i in range(num_class)],
        dataset_name='UTMobileNet')
    cfg.use_gpu = (device == 'cuda')
    model = BUILDERS[model_type](cfg)

    info = {'missing': [], 'unexpected': [], 'epoch': None, 'val_acc': None}
    if ckpt:
        blob = torch.load(ckpt, map_location='cpu', weights_only=False)
        sd = blob['model_state_dict'] if 'model_state_dict' in blob else blob
        missing, unexpected = model.load_state_dict(sd, strict=True)
        info['missing'], info['unexpected'] = list(missing), list(unexpected)
        info['epoch'] = blob.get('epoch')
        bm = blob.get('best_val_acc') or {}
        info['val_acc'] = bm.get('val_acc')
        info['val_f1'] = bm.get('val_f1')
    model = model.to(device).eval()
    info['params'] = sum(p.numel() for p in model.parameters())
    return model, cfg, info


def forward(model, x):
    """The exact call signature the training loop uses."""
    return model(x_enc=x, x_mark_enc=None, x_dec=None, x_mark_dec=None)


# =====================================================================================
# Timing
# =====================================================================================
def bench(model, X_val, batch_size, device, n_warmup=N_WARMUP, n_iters=N_ITERS):
    """Time n_iters forward passes on a batch already resident on `device`."""
    n = X_val.shape[0]
    idx = np.arange(batch_size) % n                      # deterministic, real rows
    batch = X_val[idx].to(device).contiguous()           # transfer happens BEFORE timing
    cuda = (device == 'cuda')

    with torch.no_grad():
        for _ in range(n_warmup):
            out = forward(model, batch)
        if cuda:
            torch.cuda.synchronize()

        lat = np.empty(n_iters, dtype=np.float64)
        for i in range(n_iters):
            if cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = forward(model, batch)
            if cuda:
                torch.cuda.synchronize()
            lat[i] = (time.perf_counter() - t0) * 1000.0   # ms

    bad = bool(torch.isnan(out).any() or torch.isinf(out).any())
    peak = (torch.cuda.max_memory_allocated() / 2**20) if cuda else float('nan')
    return {
        'out_shape': tuple(out.shape), 'nan_or_inf': bad,
        'latency_mean_ms': lat.mean(), 'latency_std_ms': lat.std(ddof=1),
        'latency_p50_ms': np.percentile(lat, 50),
        'latency_p95_ms': np.percentile(lat, 95),
        'latency_p99_ms': np.percentile(lat, 99),
        'latency_min_ms': lat.min(), 'latency_max_ms': lat.max(),
        'per_flow_latency_ms': lat.mean() / batch_size,
        'throughput_flows_per_s': batch_size / (lat.mean() / 1000.0),
        'peak_gpu_mem_mib': peak,
    }


# =====================================================================================
def device_info():
    cpu = platform.processor() or 'unknown'
    try:
        for line in open('/proc/cpuinfo'):
            if line.startswith('model name'):
                cpu = line.split(':', 1)[1].strip()
                break
    except OSError:
        pass
    gpu = 'none'
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        gpu = f'{p.name} ({p.total_memory/2**30:.1f} GiB)'
    return {'cpu': cpu, 'cpu_threads': torch.get_num_threads(), 'gpu': gpu,
            'torch': torch.__version__, 'cuda': torch.version.cuda,
            'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}


def seq_len_for(window):
    """Read T from the dataset itself rather than assuming the mapping."""
    p = os.path.join(ts.DATA_ROOT, 'UTMobileNet', 'direction_dataset', window,
                     '5s', '15 classes', 'features.csv')
    with open(p) as fh:
        ncol = len(fh.readline().split(','))
    assert ncol % 8 == 0, f'{p}: {ncol} columns is not a multiple of 8'
    return ncol // 8


def ckpt_path(model_type, seq_len):
    return os.path.join(CKPT_DIR, f'{model_type}_classifier_{seq_len}_best_val.pth')


def verify_arch(model_type, seq_len):
    """Architecture must not depend on the seed: TimeSeriesConfig takes no seed.

    Rebuilding under three different torch seeds must give identical parameter counts
    and identical tensor shapes; only the values differ. That is what makes a single
    surviving checkpoint a valid stand-in for latency purposes.
    """
    sigs = []
    for s in (0, 1, 2):
        torch.manual_seed(s)
        m, _, _ = build_model(model_type, seq_len)
        sigs.append(tuple(sorted((k, tuple(v.shape)) for k, v in m.state_dict().items())))
    return all(x == sigs[0] for x in sigs), sum(
        np.prod(sh) for _, sh in sigs[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true',
                    help='Informer/250ms only, both devices, batch 1 — validates the harness')
    ap.add_argument('--n_warmup', type=int, default=N_WARMUP)
    ap.add_argument('--n_iters', type=int, default=N_ITERS)
    ap.add_argument('--batch_sizes', nargs='+', type=int, default=BATCH_SIZES)
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    di = device_info()
    print(json.dumps(di, indent=1))
    devices = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])
    targets = [t for t in TARGETS if t[0] == 'informer'] if args.smoke else TARGETS
    batch_sizes = [1] if args.smoke else args.batch_sizes

    rows = []
    t_start = time.time()
    for model_type, window, label in targets:
        sl = seq_len_for(window)
        ck = ckpt_path(model_type, sl)
        Xv, n_train = val_tensor(window)
        same_arch, _ = verify_arch(model_type, sl)
        print(f"\n=== {label} | {window} | seq_len={sl} | val flows={Xv.shape[0]} "
              f"| arch seed-independent: {same_arch}")
        for device in devices:
            model, cfg, info = build_model(model_type, sl, ckpt=ck, device=device)
            if info['missing'] or info['unexpected']:
                raise SystemExit(f"{label}: checkpoint mismatch "
                                 f"missing={info['missing']} unexpected={info['unexpected']}")
            if device == 'cuda':
                torch.cuda.reset_peak_memory_stats()
            for bs in batch_sizes:
                r = bench(model, Xv, bs, device, args.n_warmup, args.n_iters)
                rows.append(dict(
                    model=label, model_type=model_type, window=window, seq_len=sl,
                    num_features=8, num_class=15, feature_config='8feat', dataset='A2',
                    device=device, batch_size=bs, params=info['params'],
                    n_warmup=args.n_warmup, n_iters=args.n_iters,
                    checkpoint=ck, ckpt_epoch=info['epoch'],
                    ckpt_val_acc=info['val_acc'], ckpt_val_f1=info.get('val_f1'),
                    arch_seed_independent=same_arch,
                    n_val_flows=int(Xv.shape[0]), out_shape=str(r.pop('out_shape')),
                    **r, cpu=di['cpu'], cpu_threads=di['cpu_threads'], gpu=di['gpu'],
                    torch_version=di['torch'], cuda_version=di['cuda']))
                print(f"  {device:<4} bs={bs:<4} mean={r['latency_mean_ms']:8.3f} ms  "
                      f"p95={r['latency_p95_ms']:8.3f}  per-flow={r['per_flow_latency_ms']:7.4f} ms  "
                      f"thr={r['throughput_flows_per_s']:9.1f} flows/s  "
                      f"NaN/Inf={r['nan_or_inf']}")
            del model
            if device == 'cuda':
                torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    tag = '_smoke' if args.smoke else ''
    raw = os.path.join(OUT, f'inference_cost_raw{tag}.csv')
    df.to_csv(raw, index=False)
    print(f"\nwrote {raw}  ({len(df)} measurements) in "
          f"{time.time() - t_start:.1f}s")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
