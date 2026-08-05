"""
Emit groups.csv for the UTMobile direction datasets — Step C.

Re-runs assembly over the existing `timeseries/direction_dataset/` intermediates so that
every row of features.csv gains a capture-group label, which Step D needs for a grouped
train/val split.

Safety: the live features.csv files are root-owned artefacts dated Oct 2025 and are the
byte-provenance of the B0 and B1 measurements. This script never writes to them. It
assembles into a scratch directory and then *proves* the new features.csv is numerically
identical to the live one before accepting groups.csv as aligned.

That proof matters more than it looks. The frozen split indexes rows of the live
features.csv, and rows are grouped by class, so a reordering *within* a class would leave
the label vector — and therefore splits.py's label_fingerprint — completely unchanged
while silently repointing every index. Content comparison is the only check that catches
it.

Usage:
    python build_groups.py --windows 75            # single-window trial
    python build_groups.py                         # all nine windows
    python build_groups.py --skip-verify           # emit without proving alignment (not advised)
"""
import argparse
import hashlib
import os
import sys
import time
from collections import defaultdict, deque

import numpy as np
import pandas as pd

from TimeseriesCreate import create_direction_dataset_for_utmobile

DATA_ROOT = os.environ.get('TS_DATA_ROOT', '/media/Data/Datasets/chanan_data')
UT = os.path.join(DATA_ROOT, 'UTMobileNet')
SCRATCH = os.environ.get('TS_GROUPS_SCRATCH', '/data/weaamm/results/UTMobileNet/groups_build')

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
GROUPS_OUT = os.path.join(os.path.dirname(_THIS_DIR), 'groups')

# Reverse-alphabetical. This order defines class_label 0..14 and MUST match the order the
# live features.csv was built with, or every label shifts.
CLASS_ORDER = ['youtube', 'twitter', 'spotify', 'reddit', 'pinterest', 'netflix',
               'messenger', 'instagram', 'hulu', 'hangout', 'google maps',
               'google drive', 'gmail', 'facebook', 'dropbox']

ALL_WINDOWS = [20, 30, 40, 50, 75, 100, 150, 200, 250]


def live_dir(win):
    return os.path.join(UT, 'direction_dataset', f'{win}ms', '5s', '15 classes')


def _row_hashes(A):
    return [hashlib.sha256(np.ascontiguousarray(r).tobytes()).hexdigest() for r in A]


def align_groups_to_live(scratch, live):
    """
    Align the freshly built groups.csv onto the live features.csv row order.

    os.walk does not promise a stable traversal order, and in practice it does not give
    the order the Oct-2025 assembly used: only ~12% of rows land in the same position.
    The row *contents* are identical though, so the live order can be recovered exactly by
    joining on row content — the join is constrained to within-class, which is sound
    because the label vectors match element-for-element.

    A few rows are byte-identical to each other (near-empty flows, ~69 of 536 cells
    non-zero). Where such twins sit in the same class but come from different captures,
    which physical file produced which row is genuinely unrecoverable. Those rows are
    flagged `group_ambiguous` rather than silently assigned, so Step D can keep them out
    of validation instead of trusting a coin flip.

    Returns (ok, message, groups_df_in_live_order).
    """
    new_X = pd.read_csv(os.path.join(scratch, 'features.csv')).to_numpy(dtype=np.float64)
    old_X = pd.read_csv(os.path.join(live, 'features.csv')).to_numpy(dtype=np.float64)
    new_y = pd.read_csv(os.path.join(scratch, 'labels.csv')).to_numpy().ravel()
    old_y = pd.read_csv(os.path.join(live, 'labels.csv')).to_numpy().ravel()
    groups = pd.read_csv(os.path.join(scratch, 'groups.csv'))

    if new_X.shape != old_X.shape:
        return False, f"shape {new_X.shape} != live {old_X.shape}", None
    if not np.array_equal(new_y, old_y):
        return False, f"labels differ ({int((new_y != old_y).sum())} rows)", None

    hn, ho = _row_hashes(new_X), _row_hashes(old_X)

    # Fast path: assembly reproduced the live order outright.
    if hn == ho:
        groups = groups.copy()
        groups['group_ambiguous'] = False
        return True, f"{len(new_X)} rows identical and already in live order", groups

    perm = np.full(len(old_X), -1, dtype=np.int64)
    ambiguous = np.zeros(len(old_X), dtype=bool)

    for cls in np.unique(old_y):
        new_pos = np.where(new_y == cls)[0]
        old_pos = np.where(old_y == cls)[0]

        buckets = defaultdict(deque)
        for j in new_pos:
            buckets[hn[j]].append(j)
        # A hash with several candidates inside one class is only a problem when those
        # candidates disagree about the capture group.
        contested = {k for k, v in buckets.items()
                     if len(v) > 1 and groups.iloc[list(v)]['capture_group'].nunique() > 1}

        for i in old_pos:
            bucket = buckets.get(ho[i])
            if not bucket:
                return False, (f"class {cls}: no content match for live row {i} — "
                               "contents differ, not just order"), None
            perm[i] = bucket.popleft()
            ambiguous[i] = ho[i] in contested

    if (perm < 0).any():
        return False, f"{int((perm < 0).sum())} live rows unmatched", None
    if len(set(perm.tolist())) != len(perm):
        return False, "permutation is not a bijection", None

    out = groups.iloc[perm].reset_index(drop=True)
    out['row'] = np.arange(len(out))
    out['group_ambiguous'] = ambiguous

    # Independent confirmation that the permutation really reproduces the live matrix.
    if not np.array_equal(new_X[perm], old_X):
        return False, "permutation did not reproduce live features.csv", None

    n_amb = int(ambiguous.sum())
    return True, (f"{len(old_X)} rows reordered onto live order "
                  f"({n_amb} flagged ambiguous)"), out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--windows', nargs='+', type=int, default=ALL_WINDOWS)
    ap.add_argument('--skip-verify', action='store_true')
    args = ap.parse_args()

    os.makedirs(GROUPS_OUT, exist_ok=True)
    results, t_all = [], time.time()

    for win in args.windows:
        print(f"\n{'=' * 70}\nWINDOW {win}ms\n{'=' * 70}")
        src = os.path.join(UT, 'timeseries', 'direction_dataset', str(win))
        input_dirs = [os.path.join(src, c) for c in CLASS_ORDER]

        missing = [d for d in input_dirs if not os.path.isdir(d)]
        if missing:
            print(f"SKIP {win}ms — missing class dirs: {missing[:3]}")
            results.append((win, 'SKIP', 'missing intermediates'))
            continue

        scratch = os.path.join(SCRATCH, f'{win}ms')
        os.makedirs(scratch, exist_ok=True)
        t0 = time.time()
        create_direction_dataset_for_utmobile(input_dirs, scratch,
                                              window_size=5, balance_classes=False)
        print(f"assembled in {time.time() - t0:.0f}s")

        if args.skip_verify:
            ok, msg = True, 'verification skipped'
            aligned = pd.read_csv(os.path.join(scratch, 'groups.csv'))
        else:
            ok, msg, aligned = align_groups_to_live(scratch, live_dir(win))

        if ok:
            dest = os.path.join(GROUPS_OUT, f'UTMobileNet_dir_{win}ms_groups.csv')
            aligned.to_csv(dest, index=False)
            print(f"✅ ALIGNED  {msg}\n   groups -> {dest}")
            results.append((win, 'OK', msg))
        else:
            print(f"🔴 MISALIGNED  {msg}\n   groups.csv NOT published for {win}ms")
            results.append((win, 'FAIL', msg))

    print(f"\n{'=' * 70}\nSUMMARY  ({time.time() - t_all:.0f}s)\n{'=' * 70}")
    for win, status, msg in results:
        print(f"{win:>5}ms  {status:<5}  {msg}")
    return 0 if all(s != 'FAIL' for _, s, _ in results) else 1


if __name__ == '__main__':
    sys.exit(main())
