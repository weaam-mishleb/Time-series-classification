"""
Regenerate the UTMobile processed dataset with corrected direction detection.

Writes to a NEW tree (TS_V2_ROOT) and never touches the October-2025 originals, so the
pre-fix dataset stays available for comparison. Emits a full before/after filtering
report — nothing is dropped silently.

Stage 1: raw packet CSVs -> per-flow timeseries intermediates (one tree per window)
Stage 2: intermediates   -> features.csv / labels.csv / groups.csv

Usage:
    python regenerate_utmobile.py                  # all 9 windows, both stages
    python regenerate_utmobile.py --windows 50     # single window
    python regenerate_utmobile.py --stage 2        # assembly only
"""
import argparse
import json
import os
import sys
import time
from datetime import timedelta

import pandas as pd

import TimeseriesCreate as tc

RAW = os.environ.get(
    'TS_RAW_ROOT',
    '/media/Data/Datasets/chanan_data/UTMobileNet/Deterministic Automated Data')
V2 = os.environ.get('TS_V2_ROOT', '/data/weaamm/utmobile_v2')

# ms -> seconds. These are the nine windows that exist in the current dataset.
WINDOWS = {20: 0.020, 30: 0.030, 40: 0.040, 50: 0.050, 75: 0.075,
           100: 0.100, 150: 0.150, 200: 0.200, 250: 0.250}

# Reverse-alphabetical: this order defines class_label 0..14 and MUST match the
# order the frozen splits were built with.
CLASS_ORDER = ['youtube', 'twitter', 'spotify', 'reddit', 'pinterest', 'netflix',
               'messenger', 'instagram', 'hulu', 'hangout', 'google maps',
               'google drive', 'gmail', 'facebook', 'dropbox']


def inter_dir(win):
    return os.path.join(V2, 'timeseries', 'direction_dataset', str(win))


def asm_dir(win):
    return os.path.join(V2, 'direction_dataset', f'{win}ms', '5s', '15 classes')


def stage1(win, sec, report_dir):
    """Raw CSVs -> intermediates, collecting per-flow filtering statistics."""
    out = inter_dir(win)
    os.makedirs(out, exist_ok=True)
    sink = tc.set_filter_stats_sink([])
    t0 = time.time()
    tc.pipeline(input_dir=RAW, output_dir=out, window_size=sec)
    tc.set_filter_stats_sink(None)

    df = pd.DataFrame(sink)
    if len(df):
        df['flow'] = df['input_file'].map(os.path.basename)
        df['app'] = df['input_file'].map(lambda p: os.path.basename(os.path.dirname(p)))
        df.to_csv(os.path.join(report_dir, f'filter_stats_{win}ms.csv'), index=False)
    print(f"[stage1] {win}ms done in {timedelta(seconds=int(time.time() - t0))} "
          f"({len(df)} flows processed)")
    return df


def stage2(win):
    """Intermediates -> features.csv / labels.csv / groups.csv."""
    src = inter_dir(win)
    input_dirs = [os.path.join(src, c) for c in CLASS_ORDER]
    missing = [d for d in input_dirs if not os.path.isdir(d)]
    if missing:
        print(f"[stage2] {win}ms SKIPPED - missing class dirs: {missing[:3]}")
        return False
    out = asm_dir(win)
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    tc.create_direction_dataset_for_utmobile(input_dirs, out, window_size=5,
                                             balance_classes=False)
    print(f"[stage2] {win}ms done in {timedelta(seconds=int(time.time() - t0))}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--windows', nargs='+', type=int, default=sorted(WINDOWS))
    ap.add_argument('--stage', type=int, choices=[1, 2], default=None,
                    help='run only this stage; default runs both')
    args = ap.parse_args()

    mode = os.environ.get('TS_DIRECTION_MODE', 'pkttype')
    report_dir = os.path.join(V2, 'reports')
    os.makedirs(report_dir, exist_ok=True)

    print(f"{'=' * 70}\nREGENERATE UTMobile  |  direction mode = {mode}\n{'=' * 70}")
    print(f"  raw    -> {RAW}")
    print(f"  output -> {V2}   (originals untouched)")
    print(f"  windows: {args.windows}\n")

    t_all = time.time()
    for win in args.windows:
        sec = WINDOWS[win]
        print(f"\n{'-' * 70}\nWINDOW {win}ms ({sec}s)\n{'-' * 70}")
        if args.stage in (None, 1):
            stage1(win, sec, report_dir)
        if args.stage in (None, 2):
            stage2(win)

    print(f"\n{'=' * 70}\nALL DONE in {timedelta(seconds=int(time.time() - t_all))}"
          f"\n{'=' * 70}")


if __name__ == '__main__':
    sys.exit(main())
