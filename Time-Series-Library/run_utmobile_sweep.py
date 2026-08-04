"""
Full UTMobileNet sweep: every model x every window size.

calibrate.py measured a single configuration to get a per-epoch cost. This script is
the actual Baseline stage A run: it walks all 6 models across all UTMobileNet window
sizes with the standard settings (num_epochs=150, EarlyStopping as configured inside
train_classifier).

Each (window, model) pair is run in its own call to evaluate_models_on_datasets, so a
pair that blows up - OOM, a model that rejects a given seq_len - is logged and the
sweep carries on. Every pair also writes its own results CSV under RESULTS_ROOT, which
means a sweep that dies halfway still leaves finished work on disk.

Windows are loaded once per window and reused across the 6 models, so the CSV parsing
cost is paid 1x per window rather than 6x.

Example:
    CUDA_VISIBLE_DEVICES=2 python run_utmobile_sweep.py
"""
import argparse
import os
import time
import traceback
from datetime import datetime, timedelta

import TimeSeries_functions as ts

# All window sizes UTMobileNet is built for. Note that the 'dir' (direction_dataset)
# variant does not ship 5ms/10ms - only the plain 'dataset' variant has all 11. Missing
# windows are logged as SKIPPED rather than treated as failures.
ALL_WINDOWS = ["5ms", "10ms", "20ms", "30ms", "40ms", "50ms",
               "75ms", "100ms", "150ms", "200ms", "250ms"]

ALL_MODELS = ["timesnet", "informer", "autoformer", "fedformer", "timemixer", "nst"]


class Tee:
    """Write log lines to stdout and to the log file at once, flushing as we go.

    Flushing matters here: the sweep runs for hours and the log is the only way to see
    where it got to from another shell.
    """

    def __init__(self, path):
        self.fh = open(path, 'a', buffering=1)

    def __call__(self, msg=""):
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{stamp}] {msg}" if msg else ""
        print(line, flush=True)
        self.fh.write(line + "\n")

    def raw(self, text):
        """Log a multi-line blob (a traceback) without stamping every line."""
        print(text, flush=True)
        self.fh.write(text + "\n")

    def close(self):
        self.fh.close()


def load_done_pairs(path):
    """Read the pairs already marked complete, for --resume."""
    if not os.path.exists(path):
        return set()
    with open(path) as fh:
        return {line.strip() for line in fh if line.strip()}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', default='UTMobileNet')
    parser.add_argument('--dataset_type', default='dir',
                        help="'dir' for the direction features (9 windows), "
                             "anything else for the plain dataset (11 windows)")
    parser.add_argument('--windows', nargs='+', default=ALL_WINDOWS,
                        help='window sizes to sweep; defaults to all 11')
    parser.add_argument('--models', nargs='+', default=ALL_MODELS,
                        choices=ALL_MODELS,
                        help='models to sweep; defaults to all 6')
    parser.add_argument('--epochs', type=int, default=150,
                        help='max epochs per run; EarlyStopping usually ends it sooner')
    parser.add_argument('--feature_name', default='vec[upstream,downstream,ratio]')
    parser.add_argument('--use_class_weights', action='store_true')
    parser.add_argument('--log_dir', default=None,
                        help='where to write the sweep log (default: RESULTS_ROOT/<dataset>/sweep_logs)')
    parser.add_argument('--resume', action='store_true',
                        help='skip (window, model) pairs already recorded as done in the progress file')
    args = parser.parse_args()

    log_dir = args.log_dir or os.path.join(ts.RESULTS_ROOT, args.dataset, 'sweep_logs')
    os.makedirs(log_dir, exist_ok=True)

    started = datetime.now()
    log_path = os.path.join(log_dir, f"sweep_{started.strftime('%Y%m%d_%H%M%S')}.log")
    progress_path = os.path.join(log_dir, f"progress_{args.dataset}_{args.dataset_type}.txt")

    log = Tee(log_path)
    done_pairs = load_done_pairs(progress_path) if args.resume else set()

    log("=== UTMobileNet sweep ===")
    log(f"dataset={args.dataset} type={args.dataset_type} epochs={args.epochs}")
    log(f"models  ({len(args.models)}): {', '.join(args.models)}")
    log(f"windows ({len(args.windows)}): {', '.join(args.windows)}")
    log(f"total pairs: {len(args.models) * len(args.windows)}")
    log(f"log       -> {log_path}")
    log(f"progress  -> {progress_path}")
    log(f"results   -> {ts.RESULTS_ROOT}")
    log(f"checkpts  -> {ts.CHECKPOINTS_ROOT}")
    log(f"data root -> {ts.DATA_ROOT}")
    if args.resume and done_pairs:
        log(f"resuming: {len(done_pairs)} pair(s) already done, will be skipped")
    log()

    ts.print_gpu_info()

    outcomes = []  # (window, model, status, elapsed)
    sweep_start = time.time()

    for window in args.windows:
        log("=" * 70)
        log(f"WINDOW {window}")
        log("=" * 70)

        # Load the window once and hand the same dict to all 6 models.
        try:
            load_start = time.time()
            datasets = ts.prepare_datasets(
                dataset_name=args.dataset,
                small_windows=[window],
                dataset_type=args.dataset_type,
                selected_feature_indices=[1, 4, 6],  # upstream, downstream, ratio
            )
            log(f"load took {timedelta(seconds=int(time.time() - load_start))}")
        except Exception as e:
            log(f"LOAD FAILED for window {window}: {e}")
            log.raw(traceback.format_exc())
            for model in args.models:
                outcomes.append((window, model, 'LOAD FAILED', 0))
            continue

        if not datasets:
            # prepare_datasets prints its own "Files not found" line and returns {}.
            log(f"SKIPPED window {window}: no data under {ts.DATA_ROOT} for this window")
            for model in args.models:
                outcomes.append((window, model, 'SKIPPED (no data)', 0))
            continue

        for key, (X, y) in datasets.items():
            log(f"loaded '{key}': X={X.shape} y={y.shape}")

        for model in args.models:
            pair = f"{window}|{model}"
            if pair in done_pairs:
                log(f"--- {window} / {model}: already done, skipping ---")
                outcomes.append((window, model, 'SKIPPED (resume)', 0))
                continue

            log("-" * 70)
            log(f"RUN {window} / {model}")
            log("-" * 70)
            run_start = time.time()
            try:
                results = ts.evaluate_models_on_datasets(
                    datasets_dict=datasets,
                    model_types=[model],
                    use_class_weights=args.use_class_weights,
                    dataset_name=args.dataset,
                    feature_name=args.feature_name,
                    num_epochs=args.epochs,
                )
                elapsed = time.time() - run_start
                log(f"OK {window} / {model} in {timedelta(seconds=int(elapsed))}")
                log.raw(results.to_string(index=False))
                outcomes.append((window, model, 'OK', elapsed))

                with open(progress_path, 'a') as fh:
                    fh.write(pair + "\n")

            except Exception as e:
                elapsed = time.time() - run_start
                log(f"FAILED {window} / {model} after {timedelta(seconds=int(elapsed))}: "
                    f"{type(e).__name__}: {e}")
                log.raw(traceback.format_exc())
                outcomes.append((window, model, f'FAILED ({type(e).__name__})', elapsed))
                # Deliberately no re-raise: the next pair gets its shot.

    total = time.time() - sweep_start
    log()
    log("=" * 70)
    log(f"SWEEP DONE in {timedelta(seconds=int(total))}")
    log("=" * 70)

    ok = sum(1 for _, _, status, _ in outcomes if status == 'OK')
    log(f"{ok}/{len(outcomes)} pair(s) completed")
    log(f"{'window':<8} {'model':<12} {'status':<24} elapsed")
    for window, model, status, elapsed in outcomes:
        log(f"{window:<8} {model:<12} {status:<24} {timedelta(seconds=int(elapsed))}")
    log()
    log(f"per-run CSVs are under {os.path.join(ts.RESULTS_ROOT, args.dataset, 'direction')}")
    log(f"full log: {log_path}")
    log.close()


if __name__ == '__main__':
    main()
