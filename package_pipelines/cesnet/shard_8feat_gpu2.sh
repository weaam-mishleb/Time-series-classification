#!/usr/bin/env bash
# CESNET 5s dir, seed 0, 8feat SHARD -- physical GPU 2 ONLY. 54 runs.
# Fully isolated from the GPU-0 3feat shard:
#   results     prod_results_8feat_seed0   (separate directory)
#   checkpoints prod_ckpt_8feat_seed0      (separate; .pth names carry no cfg/seed)
#   progress    prod_logs_8feat_seed0/progress_Cesnet_dir.txt
#   ledger      ledger/ledger_8feat_seed0.csv  (atomic temp+rename)
# The two shards therefore cannot train the same run_id.
set -uo pipefail
C=/data/weaamm/cesnet_clean
LIB=/home/weaamm/Time-series-classifction/Time-Series-Library
cd "$LIB"
export CUDA_VISIBLE_DEVICES=2 MPLBACKEND=Agg TS_SEED=0 TS_SPLIT_TAG=cleangrouped
stamp(){ echo "[$(date '+%F %T')] $*"; }
stamp "8FEAT SHARD START - physical GPU 2 - 6 models x 9 windows x seed 0 = 54 runs"
TS_FEATURES=0,1,2,3,4,5,6,7 \
TS_RESULTS_ROOT=$C/prod_results_8feat_seed0 \
TS_CHECKPOINTS_ROOT=$C/prod_ckpt_8feat_seed0 \
python3 run_utmobile_sweep.py --dataset Cesnet --dataset_type dir \
  --models timesnet informer nst autoformer fedformer timemixer \
  --windows 20ms 30ms 40ms 50ms 75ms 100ms 150ms 200ms 250ms \
  --epochs 150 --resume --log_dir $C/prod_logs_8feat_seed0
rc=$?
stamp "sweep rc=$rc - rebuilding shard ledger"
python3 $C/build_shard_ledger.py 8feat
stamp "8FEAT SHARD COMPLETE. Inference/report/QA NOT run."
