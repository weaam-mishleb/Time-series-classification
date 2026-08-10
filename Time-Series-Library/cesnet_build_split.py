"""
Build the frozen, duplicate-aware 80/20 split for the balanced CESNET dataset.

The dataset carries no flow identifier, so provenance has to be reconstructed: two rows
that are byte-identical in ANY of the 11 window sizes are treated as the same flow, and
the relation is closed transitively across windows (see cesnet_dup_audit.py). Every such
group is kept whole on one side of the split — otherwise a validation row could be an
exact copy of a training row, which is leakage of the most direct kind.

StratifiedGroupKFold gives group-integrity and class balance simultaneously; fold 0 of a
5-fold split is the validation set, i.e. 20%. seed 32 matches the project convention.

Writes one new JSON under splits/. Never overwrites, never touches raw data.
"""
import hashlib
import json
import os
import sys

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

WORK = os.environ.get('TS_CESNET_WORK', '/data/weaamm/cesnet_work')
SPLITS_DIR = os.environ.get(
    'TS_SPLITS_DIR',
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'splits'))
TAG = 'dupgrouped'
SEED = 32
N_SPLITS = 5          # fold 0 -> 20% validation


def fingerprint(y):
    return hashlib.sha256(np.asarray(y).astype(np.int64).tobytes()).hexdigest()[:16]


def main():
    root = np.load(f'{WORK}/cesnet_group_root.npy')
    labels_raw = np.load(f'{WORK}/cesnet_labels.npy')
    n = len(labels_raw)

    # The pipeline trains on LabelEncoder output, so the split fingerprint must be taken
    # on the encoded labels, not on the raw APP ids, or splits.py will reject the file.
    classes = np.array(sorted(set(labels_raw.tolist())))
    y = np.searchsorted(classes, labels_raw)          # identical to LabelEncoder ordering
    num_class = len(classes)

    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    train_idx, val_idx = next(iter(sgkf.split(np.zeros(n), y, groups=root)))
    train_idx = np.sort(train_idx)
    val_idx = np.sort(val_idx)

    # ---- invariants: refuse to write a split that leaks -------------------------------
    assert len(train_idx) + len(val_idx) == n, 'split does not cover every row'
    assert not (set(train_idx) & set(val_idx)), 'train and val overlap'
    gtr, gva = set(root[train_idx]), set(root[val_idx])
    straddle = gtr & gva
    assert not straddle, f'{len(straddle)} duplicate groups span both sides'
    assert set(y[train_idx]) == set(y[val_idx]) == set(range(num_class)), \
        'a class is missing from one side'

    per_class = []
    for c in range(num_class):
        tr = int((y[train_idx] == c).sum())
        va = int((y[val_idx] == c).sum())
        per_class.append({'label_encoded': c, 'label_raw': int(classes[c]),
                          'train': tr, 'val': va, 'val_frac': va / (tr + va)})

    rec = {
        'dataset_name': 'Cesnet',
        'dataset_variant': 'balanced_direction_datasets',
        'num_class': num_class,
        'n_samples': n,
        'val_split': len(val_idx) / n,
        'random_state': SEED,
        'tag': TAG,
        'strategy': ('StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=32), '
                     'fold 0 as validation. Groups are the transitive closure of exact '
                     'row duplicates over all 11 window sizes (5,10,20,30,40,50,75,100,'
                     '150,200,250 ms). No duplicate group spans train and validation.'),
        'label_fingerprint': fingerprint(y),
        'label_encoding': 'LabelEncoder over sorted unique APP ids',
        'classes_raw': [int(c) for c in classes],
        'n_groups': int(len(set(root))),
        'n_groups_multi_row': int(sum(1 for g in set(root)
                                      if int((root == g).sum()) > 1)),
        'note': ('CESNET carries no flow identifier, so groups are reconstructed from '
                 'byte-identical duplicate rows. Applies to every window size, because '
                 'all 11 files hold the same 18,000 flows in the same row order '
                 '(verified: identical label fingerprint in all 11).'),
        'per_class': per_class,
        'train_idx': [int(i) for i in train_idx],
        'val_idx': [int(i) for i in val_idx],
    }

    path = os.path.join(SPLITS_DIR, f'Cesnet_{num_class}cls_{n}samples_{TAG}.json')
    if os.path.exists(path):
        sys.exit(f'REFUSING to overwrite existing split: {path}')
    os.makedirs(SPLITS_DIR, exist_ok=True)
    with open(path, 'w') as fh:
        json.dump(rec, fh, indent=1)

    print(f'wrote {path}')
    print(f'  train {len(train_idx):,}  val {len(val_idx):,}  '
          f'({len(val_idx)/n:.2%} validation)')
    print(f'  groups {rec["n_groups"]:,} ({rec["n_groups_multi_row"]} multi-row), '
          f'0 straddling')
    print(f'  label fingerprint {rec["label_fingerprint"]}')
    print('\n  per class:')
    for p in per_class:
        print(f"    raw={p['label_raw']:>3} enc={p['label_encoded']:>2}  "
              f"train={p['train']:>4}  val={p['val']:>4}  val_frac={p['val_frac']:.3f}")


if __name__ == '__main__':
    main()
