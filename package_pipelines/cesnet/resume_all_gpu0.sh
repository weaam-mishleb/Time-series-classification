#!/usr/bin/env bash
# CESNET 5s dir, seed 0. ALL remaining work on physical GPU 0, strictly SEQUENTIAL.
# Grid: 6 models x {3feat,8feat} x 9 windows x seed 0 = 108. 26 reused, 82 to run.
# Sequential by design: running the 3feat and 8feat shards in parallel made both
# processes parse the same 5.3 GB window CSV at once (~17.5 GB RSS each) and filled swap.
# Protocol frozen from the 26 completed runs: TS_SEED=0, plain CrossEntropyLoss with NO
# class weights, Adam lr 1e-4, 150 epochs, EarlyStopping, TS_SPLIT_TAG=cleangrouped.
# 3feat and 8feat keep separate results/checkpoints/progress roots, because the progress
# file is keyed "window|model" with no feature-config field.
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
    --epochs 150 --resume --log_dir $C/prod_logs_${cfg}_seed0 "$@"; }

W="20ms 30ms 40ms 50ms 75ms 100ms 150ms 200ms 250ms"

stamp "STEP 1/3  3feat, original three models (resume skips the 26 already done)"
run 3feat 1,4,6 --models timesnet informer nst --windows $W
stamp "STEP 1 rc=$?"

stamp "STEP 2/3  3feat, the three newly added models"
run 3feat 1,4,6 --models autoformer fedformer timemixer --windows $W
stamp "STEP 2 rc=$?"

stamp "STEP 3/3  8feat, all six models"
run 8feat 0,1,2,3,4,5,6,7 --models timesnet informer nst autoformer fedformer timemixer --windows $W
stamp "STEP 3 rc=$?"

stamp "rebuilding shard ledgers"
python3 $C/build_shard_ledger.py 3feat
python3 $C/build_shard_ledger.py 8feat
stamp "ALL CESNET TRAINING COMPLETE. Inference/report/QA NOT run."
