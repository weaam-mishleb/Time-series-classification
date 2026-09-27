"""Measure the true decimal precision of every raw timestamp in every CSV.

The archive has no `frame.time_epoch` column; the only timestamp field is `frame.time`,
a formatted string such as 'Apr 30, 2019 07:51:15.381121000 CDT'. This scans the
fractional part of EVERY timestamp in ALL capture files and reports:

  * how many fractional digits each value carries
  * how many values have a non-zero digit finer than one microsecond
  * how many finer than one nanosecond (impossible here, but checked)
  * the finest non-zero decimal place observed anywhere

If any value is finer than a microsecond, microsecond integers are not a lossless
representation and the implementation must move to nanoseconds or Decimal.
"""
import collections
import io
import os
import re
import sys
import time
import zipfile

import pandas as pd

ARCHIVE = '/media/Data/Datasets/UTMobileNet2021/Raw/UTMobileNet2021.zip'
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'reports')
FRAC = re.compile(r'\.(\d+)')


def main():
    t0 = time.time()
    zf = zipfile.ZipFile(ARCHIVE)
    names = sorted(n for n in zf.namelist() if n.endswith('.csv'))
    print(f'scanning {len(names)} capture files for timestamp precision', flush=True)

    digit_len = collections.Counter()
    finest_place = 0            # 1..9, the finest decimal position with a non-zero digit
    n_sub_us = 0                # values with a non-zero digit beyond 1e-6
    n_total = 0
    n_missing = 0
    examples = []
    for i, n in enumerate(names, 1):
        with zf.open(n) as fh:
            d = pd.read_csv(io.TextIOWrapper(fh, 'utf-8', 'replace'),
                            usecols=lambda c: c == 'frame.time', low_memory=False)
        if 'frame.time' not in d.columns:
            n_missing += 1
            continue
        s = d['frame.time'].astype(str)
        frac = s.str.extract(FRAC)[0]
        bad = frac.isna().sum()
        n_missing += int(bad)
        frac = frac.dropna()
        n_total += len(frac)
        digit_len.update(frac.str.len().value_counts().to_dict())
        tail = frac.str[6:]                       # positions 7,8,9 = sub-microsecond
        nz = tail.str.strip('0').str.len() > 0
        k = int(nz.sum())
        if k:
            n_sub_us += k
            if len(examples) < 10:
                examples.extend(s[frac[nz].index[:3]].tolist())
        # finest non-zero decimal place anywhere
        for f in frac.unique():
            st = f.rstrip('0')
            if len(st) > finest_place:
                finest_place = len(st)
        if i % 500 == 0:
            print(f'  {i}/{len(names)}  {n_total:,} timestamps  {time.time() - t0:.0f}s',
                  flush=True)

    print(f'\n  timestamps scanned            : {n_total:,}')
    print(f'  values with no parsable fraction: {n_missing}')
    print(f'  fractional-digit lengths       : {dict(digit_len)}')
    print(f'  finest non-zero decimal place  : 1e-{finest_place} s')
    print(f'  values finer than 1 microsecond: {n_sub_us:,}')
    if examples:
        print(f'  examples: {examples[:6]}')
    verdict = ('MICROSECOND is lossless for this archive'
               if n_sub_us == 0 and finest_place <= 6
               else 'SUB-MICROSECOND PRESENT -> microsecond integers would LOSE precision')
    print(f'\n  VERDICT: {verdict}')
    pd.DataFrame([dict(timestamps=n_total, missing_fraction=n_missing,
                       digit_lengths=str(dict(digit_len)),
                       finest_decimal_place=finest_place,
                       values_finer_than_microsecond=n_sub_us,
                       verdict=verdict)]).to_csv(
        os.path.join(OUT, 'timestamp_precision.csv'), index=False)
    print(f'  wrote {OUT}/timestamp_precision.csv   ({time.time() - t0:.0f}s)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
