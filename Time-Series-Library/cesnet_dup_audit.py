"""
Duplicate audit across all 11 balanced CESNET windows, and construction of the
duplicate-union grouping needed for a leak-free split.

Read-only with respect to the data. Writes nothing here; the split file is written by
a separate script only after this audit is reviewed.

Two rows that are byte-identical in ANY window are almost certainly the same underlying
flow, so they must never be separated by a train/validation boundary. Groups are formed
per window and then merged across windows with union-find, which is the transitive
closure the instruction asks for.
"""
import hashlib
import json
import os
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

R = '/media/Data/Datasets/chanan_data/Cesnet/data/balanced_direction_datasets'
W = ['5ms', '10ms', '20ms', '30ms', '40ms', '50ms', '75ms', '100ms', '150ms',
     '200ms', '250ms']
# Intermediate artefacts (group assignment, labels). Not committed — regenerable.
OUT = os.environ.get('TS_CESNET_WORK', '/data/weaamm/cesnet_work')


class DSU:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def row_hashes(X):
    """sha1 per row of the raw feature bytes — exact equality, no tolerance."""
    Xc = np.ascontiguousarray(X)
    return [hashlib.sha1(Xc[i].tobytes()).hexdigest() for i in range(Xc.shape[0])]


def main():
    os.makedirs(OUT, exist_ok=True)
    labels = None
    dsu = None
    per_window = []

    for w in W:
        d = pd.read_csv(f'{R}/balanced_dir_cesnet_timeseries_{w}.csv')
        y = d['APP'].to_numpy()
        X = d.drop(columns=['APP']).to_numpy(np.float64)
        if labels is None:
            labels = y
            dsu = DSU(len(y))
        else:
            assert np.array_equal(labels, y), f'{w}: label order differs'

        h = row_hashes(X)
        buckets = defaultdict(list)
        for i, hh in enumerate(h):
            buckets[hh].append(i)

        dup_groups = {k: v for k, v in buckets.items() if len(v) > 1}
        n_dup_rows = sum(len(v) for v in dup_groups.values())

        # features identical AND label identical
        hb = defaultdict(list)
        for i, hh in enumerate(h):
            hb[(hh, int(y[i]))].append(i)
        dup_fl = {k: v for k, v in hb.items() if len(v) > 1}

        # same features, DIFFERENT labels -> irreducible label conflict
        conflicts = {k: sorted({int(labels[i]) for i in v})
                     for k, v in dup_groups.items()
                     if len({int(labels[i]) for i in v}) > 1}
        n_conflict_rows = sum(len(dup_groups[k]) for k in conflicts)

        for v in dup_groups.values():
            for j in v[1:]:
                dsu.union(v[0], j)

        per_window.append(dict(
            window=w, T=X.shape[1] // 8,
            dup_groups=len(dup_groups), dup_rows=n_dup_rows,
            dup_rows_pct=100 * n_dup_rows / len(y),
            max_group=max((len(v) for v in dup_groups.values()), default=0),
            dup_feat_label_groups=len(dup_fl),
            label_conflict_groups=len(conflicts),
            label_conflict_rows=n_conflict_rows))
        print(f"  {w:>6}: T={X.shape[1]//8:>3}  dup_groups={len(dup_groups):>5}  "
              f"dup_rows={n_dup_rows:>5} ({100*n_dup_rows/len(y):5.2f}%)  "
              f"max_group={max((len(v) for v in dup_groups.values()), default=0):>3}  "
              f"label_conflicts={len(conflicts):>4} groups / {n_conflict_rows:>4} rows",
              flush=True)

    pd.DataFrame(per_window).to_csv(f'{OUT}/cesnet_dup_per_window.csv', index=False)

    # ---- union across all windows ---------------------------------------------------
    root = np.array([dsu.find(i) for i in range(len(labels))])
    groups = defaultdict(list)
    for i, r in enumerate(root):
        groups[r].append(i)
    sizes = Counter(len(v) for v in groups.values())
    multi = {k: v for k, v in groups.items() if len(v) > 1}

    print(f"\n=== union of duplicate groups across all 11 windows ===")
    print(f"  rows                      : {len(labels):,}")
    print(f"  distinct groups           : {len(groups):,}")
    print(f"  groups with >1 row        : {len(multi):,}")
    print(f"  rows inside a multi-group : {sum(len(v) for v in multi.values()):,}")
    print(f"  largest group             : {max(len(v) for v in groups.values())}")
    print(f"  group-size histogram      : {dict(sorted(sizes.items()))}")

    # groups that mix labels after the union
    mixed = {k: v for k, v in multi.items() if len({int(labels[i]) for i in v}) > 1}
    print(f"  groups spanning >1 label  : {len(mixed):,} "
          f"({sum(len(v) for v in mixed.values()):,} rows)")

    per_class = pd.DataFrame([
        {'label': lab,
         'rows': int((labels == lab).sum()),
         'groups': len({root[i] for i in np.where(labels == lab)[0]})}
        for lab in sorted(set(labels.tolist()))])
    per_class['rows_per_group'] = per_class.rows / per_class.groups
    print("\n=== per class ===")
    print(per_class.to_string(index=False))
    per_class.to_csv(f'{OUT}/cesnet_dup_per_class.csv', index=False)

    np.save(f'{OUT}/cesnet_group_root.npy', root)
    np.save(f'{OUT}/cesnet_labels.npy', labels)
    print(f"\nsaved group assignment -> {OUT}/cesnet_group_root.npy")


if __name__ == '__main__':
    main()
