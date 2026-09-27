# Package evidence record

The trained packages themselves live outside this repository, under
`/data/weaamm/`. They are far too large to version: per-package feature
tensors run to tens of megabytes (`.parquet`, `.npy`, `.npz`), checkpoints to
several gigabytes (`.pth`), and the CESNET sweep logs alone exceed 60 MB.

What is committed here is the small, **non-regenerable** part of each package:

| file | why it is kept |
|---|---|
| `PACKAGE_ID.json` | the full protocol contract: population rules, split fingerprint, class list, loss, optimizer, boundary arithmetic |
| `FEATURE_MANIFEST.json` | both feature layers (legacy names vs. true meaning) |
| `split_*.json` | the **frozen** partition — cannot be regenerated once the seed/population context is lost |
| `ledger_*.csv` | one row per training run: every reported number comes from here |
| `ledger_grouped.csv` | the capture-grouped VNAT re-runs behind the paper's Table III |
| `QA_REPORT.txt`, `zero_ratio*.csv`, `per_app_train_val.csv`, `file_hashes.csv` | acceptance evidence |
| `*.xlsx` | the published report workbook |

## Contents

- `utmobile_fastflow13_{1,3,5,8}s` — UTMobileNet2021, FastFlow-13 population,
  B2 packet filter, `packet_count >= 50`, capture-level 70/30 split (seed 42),
  weighted cross-entropy, 13 classes. 324 runs each at 3/5/8 s; 180 at 1 s
  (the 1 s grid ships five windows only — see `PACKAGE_ID.window_exclusion_1s`).
- `vnat_tcp_duration_{1,3,5,8}s` — VNAT, TCP-only and duration-filtered,
  flow-level 70/30 split stratified on traffic class only. Contains **zero VPN
  flows**: every VNAT VPN capture is OpenVPN over UDP, so TCP-only removes them all.
- `cesnet_clean` — CESNET-QUIC22 (18 classes, 1,069,192 flows), 5 s horizon,
  seed 0. 3feat complete (54/54); 8feat partial.

## Reproduction code

`../package_pipelines/` holds the scripts that built and verified all of the
above, including the exact-integer timestamp/boundary methodology and the
independent feature verifiers.
