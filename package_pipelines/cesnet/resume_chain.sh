#!/usr/bin/env bash
# CESNET 5s dir, seed 0. GPU-0 SHARD = 3feat ONLY (54 runs: 26 reused + 28 to run).
# The 8feat shard runs separately on GPU 2. This script must NEVER touch 8feat.
# 26 runs verified OK and reused; 28 remaining here (nst/20ms + 3 new models x 9 windows).
# Protocol frozen from the completed runs: TS_SEED=0, plain CrossEntropyLoss (no class
# weights), Adam lr 1e-4, 150 epochs, EarlyStopping, TS_SPLIT_TAG=cleangrouped.
set -uo pipefail
C=/data/weaamm/cesnet_clean
LIB=/home/weaamm/Time-series-classifction/Time-Series-Library
cd "$LIB"
export CUDA_VISIBLE_DEVICES=0 MPLBACKEND=Agg TS_SEED=0 TS_SPLIT_TAG=cleangrouped
stamp(){ echo "[$(date '+%F %T')] $*"; }

run(){ local cfg=$1 feats=$2; shift 2
  TS_FEATURES=$feats \
  TS_RESULTS_ROOT=$C/prod_results_${cfg}_seed0 \
  TS_CHECKPOINTS_ROOT=$C/prod_ckpt_${cfg}_seed0 \
  python3 run_utmobile_sweep.py --dataset Cesnet --dataset_type dir \
    --epochs 150 --resume --log_dir $C/prod_logs_${cfg}_seed0 "$@"
  return $?; }

stamp "STEP 1/3: finish the original three models, 3feat (resume skips the 26 done)"
run 3feat 1,4,6 --models timesnet informer nst \
    --windows 20ms 30ms 40ms 50ms 75ms 100ms 150ms 200ms 250ms
stamp "STEP 1 rc=$?"

stamp "STEP 2/3: the three newly added models, 3feat"
run 3feat 1,4,6 --models autoformer fedformer timemixer \
    --windows 20ms 30ms 40ms 50ms 75ms 100ms 150ms 200ms 250ms
stamp "STEP 2 rc=$?"

stamp "3FEAT SHARD COMPLETE (GPU 0). The 8feat shard runs separately on GPU 2."
