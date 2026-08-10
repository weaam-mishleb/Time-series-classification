"""
Frozen, duplicate-aware 80/20 split for the FULL CESNET direction dataset.

Why this exists
---------------
The dataset has no flow identifier and its rows are not independent: 26.5% of the
1,069,192 rows are byte-identical to another row in at least one window. Reconstructing
the split the earlier 3feat results were measured with -- the pre-splits.py inline
train_test_split(test_size=0.2, random_state=32, stratify=y) -- shows 25.3% of its
validation rows are literal copies, same features AND same label, of a training row.
Those numbers are therefore optimistic by an unknown margin, and are kept under the
tag `legacy_leaky_split` rather than reused.

Grouping
--------
Two rows that are byte-identical in ANY (window, feature-config) pair are treated as the
same underlying flow, and the relation is closed transitively with union-find. Both 3feat
[1,4,6] and 8feat [0..7] are included, over all nine windows, so one split serves both
arms and no group can be separated in either.

Fold choice
-----------
StratifiedGroupKFold gives group integrity and class balance at once. All five folds are
evaluated and the one whose per-class validation fractions deviate least from 20% is
selected -- a decision made on labels alone, never on model output.
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

CACHE = os.environ.get('TS_CESNET_CACHE', '/data/weaamm/cesnet_full_cache')
SPLITS_DIR = os.environ.get(
    'TS_SPLITS_DIR',
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'splits'))
WINDOWS = ['20ms', '30ms', '40ms', '50ms', '75ms', '100ms', '150ms', '200ms', '250ms']
TAG = 'cleangrouped'
SEED = 32
N_SPLITS = 5


class DSU:
    def __init__(self, n):
        self.p = np.arange(n, dtype=np.int64)

    def find(self, x):
        p = self.p
        while p[x] != x:
            p[x] = p[p[x]]
            x = p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def union_hashfile(dsu, path, log):
    """Union every set of rows sharing a 128-bit content hash."""
    h = np.load(path)
    # lexsort on the 16 hash bytes groups identical rows without building a dict
    order = np.lexsort([h[:, i] for i in range(15, -1, -1)])
    hs = h[order]
    same = np.all(hs[1:] == hs[:-1], axis=1)
    n_pairs = 0
    for i in np.flatnonzero(same):
        dsu.union(int(order[i]), int(order[i + 1]))
        n_pairs += 1
    log(f'    {os.path.basename(path)}: {n_pairs:,} duplicate adjacencies')
    return n_pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--force', action='store_true', help='overwrite an existing split file')
    args = ap.parse_args()

    def log(m):
        print(f'[{time.strftime("%H:%M:%S")}] {m}', flush=True)

    y_raw = np.load(os.path.join(CACHE, WINDOWS[0], 'y_raw.npy'))
    n = len(y_raw)
    classes = np.array(sorted(set(y_raw.tolist())))
    y = np.searchsorted(classes, y_raw)          # same ordering LabelEncoder produces
    num_class = len(classes)
    log(f'{n:,} rows, {num_class} classes')

    dsu = DSU(n)
    log('building duplicate groups over 9 windows x 2 feature configs:')
    for w in WINDOWS:
        fp = np.load(os.path.join(CACHE, w, 'y_raw.npy'))
        assert np.array_equal(fp, y_raw), f'{w}: label order differs from {WINDOWS[0]}'
        for hf in ('hash3.npy', 'hash8.npy'):
            union_hashfile(dsu, os.path.join(CACHE, w, hf), log)

    root = np.array([dsu.find(i) for i in range(n)], dtype=np.int64)
    _, groups_idx, counts = np.unique(root, return_inverse=True, return_counts=True)
    n_groups = len(counts)
    log(f'groups: {n_groups:,}   multi-row: {int((counts > 1).sum()):,}   '
        f'largest: {counts.max():,}   rows in multi-row groups: '
        f'{int(counts[counts > 1].sum()):,} ({100*counts[counts>1].sum()/n:.2f}%)')

    # ---- evaluate all folds, pick the most class-balanced -----------------------------
    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    best = None
    for k, (tr, va) in enumerate(sgkf.split(np.zeros(n), y, groups=root)):
        fr = np.array([(y[va] == c).sum() / (y == c).sum() for c in range(num_class)])
        dev = float(np.abs(fr - 1 / N_SPLITS).max())
        log(f'  fold {k}: val {len(va):,} ({len(va)/n:.2%})  '
            f'per-class val frac {fr.min():.3f}-{fr.max():.3f}  max deviation {dev:.4f}')
        if best is None or dev < best[0]:
            best = (dev, k, np.sort(tr), np.sort(va))
    dev, fold, train_idx, val_idx = best
    log(f'selected fold {fold} (max per-class deviation {dev:.4f}) — chosen on labels only')

    # ---- invariants -------------------------------------------------------------------
    assert len(train_idx) + len(val_idx) == n
    assert not np.intersect1d(train_idx, val_idx).size
    straddle = np.intersect1d(np.unique(root[train_idx]), np.unique(root[val_idx]))
    assert straddle.size == 0, f'{straddle.size} duplicate groups span both sides'
    assert set(y[train_idx]) == set(y[val_idx]) == set(range(num_class))

    per_class = [{'label_encoded': int(c), 'label_raw': int(classes[c]),
                  'train': int((y[train_idx] == c).sum()),
                  'val': int((y[val_idx] == c).sum()),
                  'val_frac': float((y[val_idx] == c).sum() /
                                    ((y == c).sum()))} for c in range(num_class)]

    rec = {
        'dataset_name': 'Cesnet',
        'dataset_variant': 'direction_datasets (full)',
        'num_class': num_class, 'n_samples': n,
        'val_split': len(val_idx) / n, 'random_state': SEED, 'tag': TAG,
        'strategy': (f'StratifiedGroupKFold(n_splits={N_SPLITS}, shuffle=True, '
                     f'random_state={SEED}); fold {fold} of {N_SPLITS} selected as the one '
                     'with the smallest maximum per-class validation-fraction deviation '
                     'from 20%, decided on labels alone. Groups are the transitive closure '
                     'of exact row duplicates over 9 windows (20-250 ms) x 2 feature '
                     'configs (3feat=[1,4,6], 8feat=[0..7]).'),
        'selected_fold': fold, 'fold_max_class_deviation': dev,
        'windows_used': WINDOWS,
        'feature_configs_used': {'3feat': [1, 4, 6], '8feat': list(range(8))},
        'label_fingerprint': hashlib.sha256(y.astype(np.int64).tobytes()).hexdigest()[:16],
        'label_encoding': 'LabelEncoder over sorted unique APP ids',
        'classes_raw': [int(c) for c in classes],
        'n_groups': int(n_groups),
        'n_groups_multi_row': int((counts > 1).sum()),
        'largest_group': int(counts.max()),
        'rows_in_multi_row_groups': int(counts[counts > 1].sum()),
        'supersedes': ('legacy_leaky_split — the pre-splits.py inline '
                       'train_test_split(test_size=0.2, random_state=32, stratify=y), '
                       'whose validation set contains 25.3% literal copies of training rows'),
        'per_class': per_class,
        'train_idx': [int(i) for i in train_idx],
        'val_idx': [int(i) for i in val_idx],
    }

    path = os.path.join(SPLITS_DIR, f'Cesnet_{num_class}cls_{n}samples_{TAG}.json')
    if os.path.exists(path) and not args.force:
        sys.exit(f'REFUSING to overwrite {path} (use --force)')
    with open(path, 'w') as fh:
        json.dump(rec, fh)
    log(f'wrote {path}')
    log(f'  train {len(train_idx):,}  val {len(val_idx):,} ({len(val_idx)/n:.2%})')
    for p in per_class:
        log(f"    raw={p['label_raw']:>3} enc={p['label_encoded']:>2}  "
            f"train={p['train']:>7,}  val={p['val']:>6,}  val_frac={p['val_frac']:.4f}")


if __name__ == '__main__':
    sys.exit(main())
