"""
Fair Option-A vs Option-B comparison: one population, one split, two row orders.

Option A (direction fix only) and Option B (direction fix + background filtering) do not
contain the same set of flows, and each writes rows in its own os.walk order. Comparing
them naively would confound three different things: the preprocessing change we want to
measure, a population difference, and a split difference.

This module removes the last two:

  1. Population -- the comparison runs on the INTERSECTION of the two datasets, keyed on
     `source_file` (the originating capture filename), which is a stable flow identity and
     is independent of row order and of preprocessing.

  2. Split -- ONE grouped split is built on that intersection, keyed on `capture_group`
     (app, action, date). The group key is derived from the filename, so it is bit-identical
     in both options; preprocessing cannot move a flow between groups.

  3. Row order -- the single split is then mapped into each dataset's own row order, so
     train_idx_A and train_idx_B address the SAME physical flows in different arrays.

Everything is verified rather than assumed: containment, label agreement, group agreement,
coverage, disjointness, and no capture group straddling on either side.

Read-only with respect to the datasets. Writes split files only under --write.
No GPU, no models.

Usage:
    python build_ab_comparison.py                 # report only
    python build_ab_comparison.py --write         # also freeze the split files
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

from grouped_split import build_grouped_split

A_ROOT = os.environ.get('TS_A_ROOT', '/data/weaamm/utmobile_A')
B_ROOT = os.environ.get('TS_B_ROOT', '/data/weaamm/utmobile_B')
OUT_ROOT = os.environ.get('TS_AB_SPLIT_ROOT', '/data/weaamm/results/UTMobileNet/ab_split')
WINDOWS = [20, 30, 40, 50, 75, 100, 150, 200, 250]
KEY = 'source_file'


def groups_path(root, win):
    return os.path.join(root, 'direction_dataset', f'{win}ms', '5s', '15 classes',
                        'groups.csv')


def load(root, win):
    p = groups_path(root, win)
    if not os.path.exists(p):
        return None
    g = pd.read_csv(p)
    assert g['row'].tolist() == list(range(len(g))), f"{p}: 'row' is not 0..n-1"
    assert g[KEY].is_unique, f"{p}: duplicate {KEY}"
    return g


def fingerprint(seq):
    return hashlib.sha256(
        '\n'.join(map(str, seq)).encode()).hexdigest()[:16]


def build_window(win, verbose=True):
    """Return a dict describing the common population and the shared split for one window."""
    gA, gB = load(A_ROOT, win), load(B_ROOT, win)
    if gA is None or gB is None:
        return {'window': win, 'status': 'missing',
                'have_A': gA is not None, 'have_B': gB is not None}

    sA, sB = set(gA[KEY]), set(gB[KEY])
    common = sA & sB
    only_A, only_B = sA - sB, sB - sA

    # --- the common population, ordered canonically so the result is reproducible --------
    comm = (gA[gA[KEY].isin(common)]
            .sort_values(KEY, kind='mergesort')
            .reset_index(drop=True))
    posA = {k: i for i, k in enumerate(gA[KEY])}
    posB = {k: i for i, k in enumerate(gB[KEY])}

    # --- agreement of every label/group-bearing field, on the common flows ---------------
    bA = gA.set_index(KEY).loc[comm[KEY]]
    bB = gB.set_index(KEY).loc[comm[KEY]]
    disagree = {c: int((bA[c].to_numpy() != bB[c].to_numpy()).sum())
                for c in ('label', 'class_dir', 'app', 'action', 'date', 'capture_group')}

    # --- one split, built on the common population only ---------------------------------
    sp = comm.copy()
    sp['row'] = np.arange(len(sp))
    if 'group_ambiguous' not in sp.columns:
        # The regenerated pipeline writes source_file directly, so no flow is ambiguous.
        sp['group_ambiguous'] = False
    train_c, val_c, report = build_grouped_split(sp)

    train_files = sp.loc[train_c, KEY].tolist()
    val_files = sp.loc[val_c, KEY].tolist()

    idx = {}
    for tag, pos in (('A', posA), ('B', posB)):
        idx[f'train_{tag}'] = np.array([pos[f] for f in train_files], dtype=np.int64)
        idx[f'val_{tag}'] = np.array([pos[f] for f in val_files], dtype=np.int64)

    # --- invariants ---------------------------------------------------------------------
    checks = {}
    checks['B_subset_of_A'] = (len(only_B) == 0)
    checks['no_field_disagreement'] = all(v == 0 for v in disagree.values())
    checks['coverage'] = (len(train_files) + len(val_files) == len(comm))
    checks['disjoint'] = (len(set(train_files) & set(val_files)) == 0)
    for tag in ('A', 'B'):
        g = {'A': gA, 'B': gB}[tag]
        tr, va = idx[f'train_{tag}'], idx[f'val_{tag}']
        checks[f'{tag}_files_match'] = (
            g.loc[tr, KEY].tolist() == train_files and g.loc[va, KEY].tolist() == val_files)
        checks[f'{tag}_no_straddle'] = not (
            set(g.loc[tr, 'capture_group']) & set(g.loc[va, 'capture_group']))
        checks[f'{tag}_all_classes'] = (
            g.loc[tr, 'class_dir'].nunique() == g.loc[va, 'class_dir'].nunique()
            == comm['class_dir'].nunique())
    # the decisive one: identical flows, hence identical labels, in both index orders
    checks['labels_identical'] = (
        gA.loc[idx['train_A'], 'label'].tolist() == gB.loc[idx['train_B'], 'label'].tolist()
        and gA.loc[idx['val_A'], 'label'].tolist() == gB.loc[idx['val_B'], 'label'].tolist())

    return {'window': win, 'status': 'ok',
            'n_A': len(gA), 'n_B': len(gB), 'n_common': len(comm),
            'only_A': sorted(only_A), 'only_B': sorted(only_B),
            'disagree': disagree, 'checks': checks,
            'n_train': len(train_files), 'n_val': len(val_files),
            'val_frac': len(val_files) / len(comm),
            'n_groups': int(comm['capture_group'].nunique()),
            'fp_train': fingerprint(train_files), 'fp_val': fingerprint(val_files),
            'report': report, 'idx': idx,
            'train_files': train_files, 'val_files': val_files}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--windows', nargs='+', type=int, default=WINDOWS)
    ap.add_argument('--write', action='store_true', help='freeze the split files to disk')
    args = ap.parse_args()

    results = []
    for win in args.windows:
        r = build_window(win)
        results.append(r)
        if r['status'] != 'ok':
            print(f"{win:>4}ms  SKIPPED (A={r['have_A']} B={r['have_B']})")
            continue
        bad = [k for k, v in r['checks'].items() if not v]
        flag = 'OK' if not bad else 'FAILED: ' + ','.join(bad)
        print(f"{win:>4}ms  A={r['n_A']:5d}  B={r['n_B']:5d}  common={r['n_common']:5d}  "
              f"only_A={len(r['only_A']):3d}  train={r['n_train']:5d} val={r['n_val']:4d} "
              f"({r['val_frac']:.1%})  groups={r['n_groups']}  {flag}")

        if args.write:
            d = os.path.join(OUT_ROOT, f'{win}ms')
            os.makedirs(d, exist_ok=True)
            for k, v in r['idx'].items():
                np.save(os.path.join(d, f'{k}.npy'), v)
            pd.DataFrame({'source_file': r['train_files'] + r['val_files'],
                          'split': ['train'] * r['n_train'] + ['val'] * r['n_val'],
                          'row_A': np.concatenate([r['idx']['train_A'], r['idx']['val_A']]),
                          'row_B': np.concatenate([r['idx']['train_B'], r['idx']['val_B']]),
                          }).to_csv(os.path.join(d, 'split_map.csv'), index=False)
            r['report'].to_csv(os.path.join(d, 'per_class.csv'), index=False)
            with open(os.path.join(d, 'meta.json'), 'w') as f:
                json.dump({k: r[k] for k in ('window', 'n_A', 'n_B', 'n_common', 'only_A',
                                             'only_B', 'n_train', 'n_val', 'val_frac',
                                             'n_groups', 'fp_train', 'fp_val', 'checks')},
                          f, indent=1)

    ok = [r for r in results if r['status'] == 'ok']
    if ok:
        print(f"\nsplit fingerprints (identical across windows only if the population is):")
        for r in ok:
            print(f"  {r['window']:>4}ms  train {r['fp_train']}  val {r['fp_val']}")
    failed = [r for r in ok if not all(r['checks'].values())]
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
