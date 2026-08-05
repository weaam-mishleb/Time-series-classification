"""
Group-aware train/validation split — Step D.

The frozen random split puts 48 of 52 capture groups on both sides of the boundary:
flows from one recording session (same app, same action, same day, often seconds apart)
appear in both train and validation. That is the leading remaining explanation for the
~4.4-point train/val gap that the B1 selection fix did not close.

This module builds a split in which **no capture group is ever divided**. A whole
session lands on one side or the other, so a validation flow never has a sibling in
training.

Selection is exhaustive, not random. Every capture group belongs to exactly one class
(the group key starts with the app, and the app determines the class), so the choice
decomposes per class: for each class, enumerate every way of assigning its groups to the
two sides and keep whichever comes closest to the target validation fraction. Classes
hold 2-8 groups, so the largest enumeration is 2^8 - 2 = 254 cases. The result is
deterministic and needs no seed.

Constraints enforced:
  * a group is never split                        (the point of the exercise)
  * every class appears on both sides             (>=1 group each way, asserted)
  * rows flagged `group_ambiguous` go to train    (see below)

On ambiguity: a handful of near-empty flows are byte-identical to each other. Where such
twins sit in the same class under different capture groups, which file produced which row
is unrecoverable. Every row sharing a contested content hash is forced to train — pinning
only one of the pair would leave its exact duplicate in validation, which is precisely the
contamination the grouped split exists to remove.
"""
import argparse
import glob
import hashlib
import json
import os
import re
from itertools import combinations

import numpy as np
import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(_THIS_DIR)
GROUPS_DIR = os.path.join(REPO, 'groups')
SPLITS_DIR = os.environ.get('TS_SPLITS_DIR', os.path.join(REPO, 'splits'))

TARGET_VAL_FRACTION = 0.20
MIN_VAL_ROWS = 20      # below this a per-class F1 is noise
MIN_TRAIN_ROWS = 50    # below this the class is not learnable at all


def _choose_val_groups(sizes, candidates, target, min_val_rows, min_train_rows):
    """
    Pick the subset of one class's groups whose row share is closest to `target`.

    `sizes` maps group name -> row count over every group in the class; `candidates` is
    the subset eligible for validation. The fraction is measured against the class total,
    not against the candidates, so forcing a group to train shrinks the achievable
    validation share rather than silently rescaling the target.

    Plain "closest to target" is not enough here, because group sizes within a class are
    wildly uneven — spotify is [126, 126, 1] and twitter [152, 134, 1, 1]. Nearest-to-20%
    then selects the 1-row group, giving a validation set that cannot measure anything: a
    single error would move that class's recall from 100% to 0%.

    Two floors, applied in priority order:

      1. `min_train_rows` (hard). The class must stay learnable. hangout is [108, 8, 1];
         forcing 20+ validation rows there means training on 9, and the model would simply
         never learn the class — far worse than a coarse validation estimate.
      2. `min_val_rows` (relaxed when nothing satisfies it). Keeps per-class metrics
         meaningful wherever the group structure permits.

    Ties break toward fewer groups, then alphabetically, so the outcome is reproducible
    without a seed.
    """
    names = sorted(candidates)
    total = sum(sizes.values())
    forced_to_train = len(sizes) > len(names)
    max_r = len(names) if forced_to_train else len(names) - 1

    def search(val_floor):
        best = None
        for r in range(1, max_r + 1):
            for combo in combinations(names, r):
                v = sum(sizes[g] for g in combo)
                if v < val_floor or total - v < min_train_rows:
                    continue
                key = (abs(v / total - target), r, combo)
                if best is None or key < best[0]:
                    best = (key, combo, v / total)
        return best

    best = search(min_val_rows)
    relaxed = False
    if best is None:
        # No subset clears the validation floor while keeping the class learnable.
        # Training data wins: take the best split that satisfies the train floor alone.
        best = search(1)
        relaxed = True
    if best is None:
        raise ValueError(
            f"no group assignment leaves {min_train_rows} training rows "
            f"(class total {total}, groups {sorted(sizes.values(), reverse=True)})")
    return list(best[1]), best[2], relaxed


def build_grouped_split(groups_df, target=TARGET_VAL_FRACTION,
                        min_val_rows=MIN_VAL_ROWS, min_train_rows=MIN_TRAIN_ROWS):
    """Return (train_idx, val_idx, report) with no capture group spanning the two."""
    g = groups_df
    pinned = g['group_ambiguous'].to_numpy(dtype=bool)

    # A pinned row sits in train, so its group must go to train whole — otherwise that
    # group would span both sides, which is the precise condition this split removes.
    # The row's true group is unknown but is one of the groups holding a contested twin,
    # and every such row is flagged, so forcing all of their groups covers every candidate.
    forced_train_groups = set(g.loc[pinned, 'capture_group'])

    val_rows, report = [], []
    for cls in sorted(g['class_dir'].unique()):
        m = (g['class_dir'] == cls).to_numpy()
        # Pinned rows are removed from the accounting entirely: they cannot go to val, so
        # counting them would bias each group's apparent size.
        eligible = m & ~pinned
        sub = g[eligible]
        sizes = sub.groupby('capture_group').size().to_dict()
        candidates = {k for k in sizes if k not in forced_train_groups}

        if len(sizes) < 2:
            raise ValueError(
                f"class {cls} has {len(sizes)} eligible capture group(s); a grouped split "
                "needs at least 2 so the class can appear on both sides.")
        if not candidates:
            raise ValueError(
                f"class {cls} has every capture group forced to train by ambiguous rows; "
                "it cannot appear in validation.")

        chosen, frac, relaxed = _choose_val_groups(
            sizes, candidates, target, min_val_rows, min_train_rows)
        picked = sub[sub['capture_group'].isin(chosen)]
        val_rows.extend(picked['row'].tolist())

        report.append({
            'class': cls,
            'groups_total': len(sizes),
            'groups_val': len(chosen),
            'rows_total': int(m.sum()),
            'rows_val': len(picked),
            'val_frac': frac,
            'pinned': int((m & pinned).sum()),
            'relaxed': relaxed,
            'val_groups': ','.join(chosen),
        })

    n = len(g)
    val_idx = np.array(sorted(val_rows), dtype=np.int64)
    train_idx = np.array(sorted(set(range(n)) - set(val_rows)), dtype=np.int64)

    # --- invariants: fail loudly rather than train on a broken split -------------------
    assert len(train_idx) + len(val_idx) == n, "split does not cover every row"
    assert not (set(train_idx) & set(val_idx)), "train and val overlap"
    assert not set(g.loc[val_idx, 'row']) & set(np.where(pinned)[0].tolist()), \
        "an ambiguous row reached validation"

    tr_groups = set(g.loc[train_idx, 'capture_group'])
    va_groups = set(g.loc[val_idx, 'capture_group'])
    straddling = tr_groups & va_groups
    assert not straddling, f"{len(straddling)} capture groups span both sides: {straddling}"

    tr_cls = set(g.loc[train_idx, 'class_dir'])
    va_cls = set(g.loc[val_idx, 'class_dir'])
    assert len(tr_cls) == len(va_cls) == g['class_dir'].nunique(), \
        f"class coverage broken: {len(tr_cls)} train / {len(va_cls)} val classes"

    return train_idx, val_idx, pd.DataFrame(report)


def _fingerprint(y):
    return hashlib.sha256(np.asarray(y).astype(np.int64).tobytes()).hexdigest()[:16]


def build_control_split(groups_df, val_idx_grouped, seed=32):
    """
    Random split matched to the grouped split, class for class — the B2 control.

    Moving to the grouped split changes three things at once: capture sessions stop
    spanning the boundary, training loses 314 rows, and the validation class priors shift
    by up to 7 points. A drop measured against B1 would blend all three and could not be
    attributed to any one of them.

    This control reproduces the last two exactly — identical per-class validation counts,
    hence identical training size and identical priors — while drawing rows at random
    within each class instead of by whole capture group. Sessions therefore span the
    boundary here, as they did in B0 and B1.

        B2 vs control   -> the leakage effect alone, everything else held fixed
        B1 vs control   -> the cost of the smaller, reweighted validation set alone

    The ambiguous rows are pinned to train here too, so the two splits differ in exactly
    one respect: whether the partition respects capture sessions.
    """
    g = groups_df
    rng = np.random.default_rng(seed)
    pinned = g['group_ambiguous'].to_numpy(dtype=bool)
    val_counts = g.loc[val_idx_grouped, 'class_dir'].value_counts().to_dict()

    val_rows = []
    for cls in sorted(g['class_dir'].unique()):
        pool = g.index[(g['class_dir'] == cls).to_numpy() & ~pinned].to_numpy()
        k = val_counts.get(cls, 0)
        if k > len(pool):
            raise ValueError(f"class {cls}: need {k} val rows, only {len(pool)} eligible")
        val_rows.extend(rng.choice(pool, size=k, replace=False).tolist())

    val_idx = np.array(sorted(val_rows), dtype=np.int64)
    train_idx = np.array(sorted(set(range(len(g))) - set(val_rows)), dtype=np.int64)

    assert len(train_idx) + len(val_idx) == len(g), "control split does not cover all rows"
    assert not (set(train_idx) & set(val_idx)), "control split overlaps"
    assert len(val_idx) == len(val_idx_grouped), \
        f"control val size {len(val_idx)} != grouped {len(val_idx_grouped)}"
    ctrl = g.loc[val_idx, 'class_dir'].value_counts().to_dict()
    assert ctrl == val_counts, "control per-class counts do not match the grouped split"
    return train_idx, val_idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', type=float, default=TARGET_VAL_FRACTION)
    ap.add_argument('--min-val-rows', type=int, default=MIN_VAL_ROWS)
    ap.add_argument('--min-train-rows', type=int, default=MIN_TRAIN_ROWS)
    ap.add_argument('--write', action='store_true',
                    help='write the split JSONs; omit for a dry run')
    ap.add_argument('--detail-window', type=int, default=50,
                    help='window whose per-class table is printed')
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(GROUPS_DIR, 'UTMobileNet_dir_*ms_groups.csv')),
                   key=lambda s: int(re.search(r'_(\d+)ms', s).group(1)))
    summary = []

    for p in paths:
        win = int(re.search(r'_(\d+)ms', p).group(1))
        g = pd.read_csv(p)
        train_idx, val_idx, rep = build_grouped_split(
            g, args.target, args.min_val_rows, args.min_train_rows)
        n = len(g)
        summary.append((win, n, len(train_idx), len(val_idx), len(val_idx) / n,
                        g['capture_group'].nunique(),
                        len(set(g.loc[val_idx, 'capture_group']))))

        if win == args.detail_window:
            detail = rep

        if args.write:
            os.makedirs(SPLITS_DIR, exist_ok=True)
            out = os.path.join(SPLITS_DIR,
                               f'UTMobileNet_15cls_{n}samples_grouped20.json')
            with open(out, 'w') as fh:
                json.dump({
                    'dataset_name': 'UTMobileNet', 'num_class': 15, 'n_samples': n,
                    'val_split': args.target, 'random_state': None, 'tag': 'grouped20',
                    'strategy': 'grouped_by_capture_session',
                    'group_key': '(app, action, date)',
                    'label_fingerprint': _fingerprint(g['label'].to_numpy()),
                    'note': ('No capture group spans train and val. Rows flagged '
                             'group_ambiguous are pinned to train. Deterministic: groups '
                             'chosen by exhaustive per-class enumeration, no seed.'),
                    'train_idx': [int(i) for i in train_idx],
                    'val_idx': [int(i) for i in val_idx],
                }, fh, indent=1)
            print(f'wrote {os.path.basename(out)}')

            c_train, c_val = build_control_split(g, val_idx)
            out_c = os.path.join(SPLITS_DIR,
                                 f'UTMobileNet_15cls_{n}samples_control20.json')
            with open(out_c, 'w') as fh:
                json.dump({
                    'dataset_name': 'UTMobileNet', 'num_class': 15, 'n_samples': n,
                    'val_split': args.target, 'random_state': 32, 'tag': 'control20',
                    'strategy': 'random_within_class_matched_to_grouped',
                    'group_key': None,
                    'label_fingerprint': _fingerprint(g['label'].to_numpy()),
                    'note': ('Control for the grouped split: identical per-class val '
                             'counts, so training size and class priors match exactly, '
                             'but rows are drawn at random within each class. Capture '
                             'sessions DO span the boundary here, as in B0/B1. The only '
                             'difference from grouped20 is session separation.'),
                    'train_idx': [int(i) for i in c_train],
                    'val_idx': [int(i) for i in c_val],
                }, fh, indent=1)
            print(f'wrote {os.path.basename(out_c)}')

    print(f"\n{'=' * 78}\nGROUPED SPLIT — overall (target val fraction "
          f"{args.target:.0%})\n{'=' * 78}")
    print(f"{'win':>7} {'rows':>6} {'train':>7} {'val':>6} {'val%':>7} "
          f"{'groups':>7} {'val grp':>8}")
    for w, n, tr, va, f, ng, nvg in summary:
        print(f"{w:>5}ms {n:>6} {tr:>7} {va:>6} {f:>6.1%} {ng:>7} {nvg:>8}")

    print(f"\n{'=' * 78}\nPER-CLASS BALANCE @ {args.detail_window}ms\n{'=' * 78}")
    print(f"{'class':<14}{'grp':>5}{'grp_val':>8}{'rows':>7}{'val':>6}{'val%':>7}"
          f"{'pin':>5}  val groups")
    for _, r in detail.iterrows():
        print(f"{r['class']:<14}{r['groups_total']:>5}{r['groups_val']:>8}"
              f"{r['rows_total']:>7}{r['rows_val']:>6}{r['val_frac']:>6.1%}"
              f"{r['pinned']:>5}  {'RELAXED ' if r['relaxed'] else ''}{r['val_groups']}")
    print(f"\n{'TOTAL':<14}{detail['groups_total'].sum():>5}"
          f"{detail['groups_val'].sum():>8}{detail['rows_total'].sum():>7}"
          f"{detail['rows_val'].sum():>6}"
          f"{detail['rows_val'].sum() / detail['rows_total'].sum():>6.1%}"
          f"{detail['pinned'].sum():>5}")
    if not args.write:
        print("\n(dry run — no files written; re-run with --write to freeze)")


if __name__ == '__main__':
    main()
