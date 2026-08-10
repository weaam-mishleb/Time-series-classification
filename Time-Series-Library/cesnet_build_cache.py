"""
Single-pass cache builder for the full CESNET direction dataset.

The source CSVs live on an NFS mount that has been delivering roughly 12-25 MB/s under
load, and the previous approach walked each file three times (wc, cut, sort). This reads
every file exactly once, sequentially, one window at a time — never two NFS readers at
once — and in that single pass it:

  * writes a float32 memmap of shape (n, T, 8) to local disk,
  * validates row count, column count, label vector, NaN/Inf,
  * computes a 128-bit content hash per row for BOTH feature configurations,
      3feat = columns [1, 4, 6]   (packet_count_up, packet_count_down, ratio_up_down)
      8feat = all 8 columns
    so the duplicate grouping can be built later without touching NFS again.

float32 halves the footprint versus the float64 the loader uses, and the values are
packet counts and byte sums well inside float32's exact-integer range (2^24), so nothing
is lost. Total cache: ~75 GB for all 11 windows.

Source files are opened read-only and never modified.
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd

SRC = '/media/Data/Datasets/chanan_data/Cesnet/data/direction_datasets'
CACHE = os.environ.get('TS_CESNET_CACHE', '/data/weaamm/cesnet_full_cache')
# Easy windows first, so usable data exists early; 5 ms (21.7 GB) last.
WINDOWS = ['250ms', '200ms', '150ms', '100ms', '75ms', '50ms', '40ms', '30ms',
           '20ms', '10ms', '5ms']
FEAT_3 = [1, 4, 6]
CHUNK = 20000


def hash_rows(a):
    """128-bit blake2b per row -> (n, 16) uint8. Collision probability is negligible."""
    a = np.ascontiguousarray(a)
    out = np.empty((a.shape[0], 16), dtype=np.uint8)
    for i in range(a.shape[0]):
        out[i] = np.frombuffer(
            hashlib.blake2b(a[i].tobytes(), digest_size=16).digest(), dtype=np.uint8)
    return out


def fingerprint(y):
    return hashlib.sha256(np.asarray(y).astype(np.int64).tobytes()).hexdigest()[:16]


def build(window, log):
    src = os.path.join(SRC, f'dir_cesnet_timeseries_{window}.csv')
    dst = os.path.join(CACHE, window)
    os.makedirs(dst, exist_ok=True)
    done_flag = os.path.join(dst, 'COMPLETE.json')
    if os.path.exists(done_flag):
        log(f'{window}: already complete, skipping')
        return json.load(open(done_flag))

    size_gb = os.path.getsize(src) / 2**30
    header = pd.read_csv(src, nrows=0)
    cols = len(header.columns)
    T, rem = divmod(cols - 1, 8)
    assert rem == 0, f'{window}: (cols-1) not divisible by 8'
    log(f'{window}: {size_gb:.1f} GB, {cols} cols -> T={T}')

    X = None
    labels, h8, h3 = [], [], []
    n = 0
    n_nan = n_inf = n_neg = 0
    t0 = time.time()
    bytes_done = 0

    for ci, ch in enumerate(pd.read_csv(src, chunksize=CHUNK)):
        y = ch['APP'].to_numpy()
        arr = ch.drop(columns=['APP']).to_numpy(np.float32)
        assert arr.shape[1] == T * 8, f'{window}: chunk width {arr.shape[1]} != {T*8}'
        arr = arr.reshape(len(ch), T, 8)

        n_nan += int(np.isnan(arr).sum())
        n_inf += int(np.isinf(arr).sum())
        n_neg += int((arr < 0).sum())

        if X is None:
            # total rows are known only after the pass; grow the memmap in one go by
            # using the row count of the 250 ms file, which is validated below.
            X = np.lib.format.open_memmap(
                os.path.join(dst, 'X.npy'), mode='w+', dtype=np.float32,
                shape=(EXPECTED_ROWS, T, 8))
        X[n:n + len(ch)] = arr
        labels.append(y)
        h8.append(hash_rows(arr.reshape(len(ch), -1)))
        h3.append(hash_rows(arr[:, :, FEAT_3].reshape(len(ch), -1)))
        n += len(ch)

        bytes_done += arr.nbytes / 2          # rough CSV-bytes proxy
        if ci % 5 == 0:
            el = time.time() - t0
            rate = n / el if el else 0
            eta = (EXPECTED_ROWS - n) / rate if rate else 0
            log(f'  {window}: {n:>9,}/{EXPECTED_ROWS:,} rows  '
                f'{100*n/EXPECTED_ROWS:5.1f}%  {rate:,.0f} rows/s  '
                f'elapsed {el/60:.1f}m  eta {eta/60:.1f}m')

    assert n == EXPECTED_ROWS, f'{window}: got {n} rows, expected {EXPECTED_ROWS}'
    X.flush()
    del X

    y = np.concatenate(labels)
    np.save(os.path.join(dst, 'y_raw.npy'), y)
    np.save(os.path.join(dst, 'hash8.npy'), np.concatenate(h8))
    np.save(os.path.join(dst, 'hash3.npy'), np.concatenate(h3))

    rec = dict(window=window, T=T, rows=n, cols=cols, src_gb=round(size_gb, 2),
               classes=int(pd.Series(y).nunique()),
               label_fingerprint=fingerprint(y),
               nan=n_nan, inf=n_inf, negative=n_neg,
               seconds=round(time.time() - t0, 1))
    with open(done_flag, 'w') as fh:
        json.dump(rec, fh, indent=1)
    log(f'{window}: DONE in {rec["seconds"]/60:.1f} min  '
        f'fp={rec["label_fingerprint"]} nan={n_nan} inf={n_inf} neg={n_neg}')
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--windows', nargs='+', default=WINDOWS)
    ap.add_argument('--expected_rows', type=int, default=1069192)
    args = ap.parse_args()

    global EXPECTED_ROWS
    EXPECTED_ROWS = args.expected_rows

    os.makedirs(CACHE, exist_ok=True)
    logf = open(os.path.join(CACHE, 'build.log'), 'a', buffering=1)

    def log(m):
        line = f'[{time.strftime("%F %T")}] {m}'
        print(line, flush=True)
        logf.write(line + '\n')

    log(f'=== cache build start: {len(args.windows)} windows -> {CACHE} ===')
    t0 = time.time()
    recs = []
    for w in args.windows:
        recs.append(build(w, log))
        pd.DataFrame(recs).to_csv(os.path.join(CACHE, 'manifest.csv'), index=False)

    fps = {r['label_fingerprint'] for r in recs}
    log(f'=== all done in {(time.time()-t0)/60:.1f} min ===')
    log(f'label fingerprint identical across windows: {len(fps) == 1} {fps}')
    log(f'total NaN {sum(r["nan"] for r in recs)}  Inf {sum(r["inf"] for r in recs)}  '
        f'negative {sum(r["negative"] for r in recs)}')


if __name__ == '__main__':
    sys.exit(main())
