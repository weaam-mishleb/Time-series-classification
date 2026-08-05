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

Two things the 2026-08-04 sweep taught us, both handled below:

  * evaluate_models_on_datasets catches its own exceptions, prints them, and returns a
    DataFrame with the string "Error" in every metric column. A returned frame is
    therefore NOT proof of success - the frame has to be inspected. That run reported
    54/54 "OK" while only 7 pairs had produced real numbers.
  * Because the library printed those failures to stdout/stderr rather than raising,
    none of it reached the sweep log, which looked completely clean. stdout and stderr
    are now mirrored into the log file.

Example:
    CUDA_VISIBLE_DEVICES=2 python run_utmobile_sweep.py
"""
import argparse
import collections
import gc
import os
import sys
import time
import traceback
from datetime import datetime, timedelta

import torch

import TimeSeries_functions as ts

# All window sizes UTMobileNet is built for. Note that the 'dir' (direction_dataset)
# variant does not ship 5ms/10ms - only the plain 'dataset' variant has all 11. Missing
# windows are logged as SKIPPED rather than treated as failures.
ALL_WINDOWS = ["5ms", "10ms", "20ms", "30ms", "40ms", "50ms",
               "75ms", "100ms", "150ms", "200ms", "250ms"]

ALL_MODELS = ["timesnet", "informer", "autoformer", "fedformer", "timemixer", "nst"]

# Batches per dataloader under --fast-dev-run. Two rather than one so the training loop
# actually steps the optimizer more than once and per-batch averaging is exercised.
FAST_DEV_BATCHES = 2

# Recent output lines, kept so a failed run can report why it failed. The library
# prints its exception and returns normally, so the reason exists only in the output
# stream - there is no exception object left for us to inspect.
RECENT_OUTPUT = collections.deque(maxlen=400)


class TeeStream:
    """File-like wrapper mirroring a stream to the console and to the log file.

    Installed over sys.stdout/sys.stderr so that everything the library prints - the
    "Error processing ..." lines and their tracebacks above all - lands in the log file
    instead of only on a console nobody is attached to any more.
    """

    def __init__(self, stream, fh):
        self.stream = stream
        self.fh = fh
        self._partial = ""

    def write(self, data):
        self.stream.write(data)
        # tqdm redraws its bar with a bare \r many times a second. Mirroring those would
        # bloat the log into hundreds of MB over a 150-epoch sweep, so keep only chunks
        # that terminate a line.
        if not (data.startswith("\r") and not data.endswith("\n")):
            self.fh.write(data)
        self._capture(data)
        return len(data)

    def _capture(self, data):
        self._partial += data
        while "\n" in self._partial:
            line, self._partial = self._partial.split("\n", 1)
            line = line.strip()
            if line:
                RECENT_OUTPUT.append(line)

    def flush(self):
        self.stream.flush()
        self.fh.flush()

    def isatty(self):
        return self.stream.isatty()

    def fileno(self):
        # tqdm and friends probe this; delegate to the real stream.
        return self.stream.fileno()


def make_logger(fh):
    """Return a log(msg) that timestamps a line and sends it through the tee."""

    def log(msg=""):
        if msg:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)
        else:
            print("", flush=True)

    return log


def gpu_free_gb():
    """Free / total memory on the visible CUDA device, in GiB. (None, None) if no GPU."""
    if not torch.cuda.is_available():
        return None, None
    free, total = torch.cuda.mem_get_info()
    return free / 1024 ** 3, total / 1024 ** 3


def inspect_results(df):
    """Decide whether a returned frame represents a real run.

    evaluate_models_on_datasets returns a DataFrame either way, writing "Error" (or
    "Missing", from its length-padding path) into every metric column when the run threw.
    """
    if df is None or len(df) == 0:
        return False, "empty results frame"

    metric_cols = [c for c in df.columns if c not in ('Model', 'Dataset')]
    if not metric_cols:
        return False, "results frame has no metric columns"

    bad_rows = df[metric_cols].isin(["Error", "Missing"]).any(axis=1)
    if bad_rows.all():
        return False, "metrics reported as Error"
    if bad_rows.any():
        return False, f"{int(bad_rows.sum())}/{len(df)} row(s) reported as Error"
    return True, ""


def failure_reason(model):
    """Recover the library's error message for `model` from the captured output."""
    marker = f"with {model}: "
    for line in reversed(RECENT_OUTPUT):
        if "Error processing" in line and marker in line:
            reason = line.split(marker, 1)[1].strip()
            return reason[:300]
    return ""


def classify(reason):
    """Short tag for the summary table, so the failure modes are countable."""
    low = reason.lower()
    if "out of memory" in low or "outofmemory" in low:
        return "CUDA OOM"
    if "too many open files" in low or "errno 24" in low:
        return "FD EXHAUSTION"
    if not reason:
        return "FAILED"
    return "FAILED"


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
    parser.add_argument('--min_free_gb', type=float, default=6.0,
                        help='refuse to start if the visible GPU has less free memory than '
                             'this; the server has no scheduler, so a co-tenant job can '
                             'leave too little VRAM. 0 disables the check.')
    parser.add_argument('--max_consecutive_failures', type=int, default=6,
                        help='abort the sweep after this many failures in a row rather '
                             'than burning through every remaining pair. 0 disables.')
    parser.add_argument('--fast-dev-run', '--debug', dest='fast_dev_run',
                        action='store_true',
                        help='sanity check: 1 configuration, 1 epoch, '
                             f'{FAST_DEV_BATCHES} batches per dataloader. Exercises the '
                             'whole path - load, build, train, validate, evaluate, write '
                             'CSV - in seconds, so a typo does not surface 3 hours into a '
                             'real sweep. Writes nothing to the progress file.')
    parser.add_argument('--limit_batches', type=int, default=None,
                        help='cap every dataloader at N batches (implied by --fast-dev-run)')
    args = parser.parse_args()

    # --fast-dev-run is a smoke test, so it overrides the sweep's breadth and depth.
    # Applied before anything is logged, so the log shows what actually ran.
    if args.fast_dev_run:
        args.models = args.models[:1]
        args.epochs = 1
        if args.limit_batches is None:
            args.limit_batches = FAST_DEV_BATCHES
        # Resuming a smoke test would mean skipping the one pair it is meant to run.
        args.resume = False
        # The window list is deliberately NOT truncated to its first entry. The 'dir'
        # variant ships no 5ms/10ms data, so windows[:1] would pick 5ms, skip it for lack
        # of data, and exit having verified nothing. Instead the loop below stops after
        # the first pair that actually runs, which lands on the first window with data.

    log_dir = args.log_dir or os.path.join(ts.RESULTS_ROOT, args.dataset, 'sweep_logs')
    os.makedirs(log_dir, exist_ok=True)

    started = datetime.now()
    log_path = os.path.join(log_dir, f"sweep_{started.strftime('%Y%m%d_%H%M%S')}.log")
    progress_path = os.path.join(log_dir, f"progress_{args.dataset}_{args.dataset_type}.txt")

    fh = open(log_path, 'a', buffering=1)
    sys.stdout = TeeStream(sys.__stdout__, fh)
    sys.stderr = TeeStream(sys.__stderr__, fh)
    log = make_logger(fh)

    try:
        done_pairs = load_done_pairs(progress_path) if args.resume else set()

        if args.fast_dev_run:
            log("*** FAST-DEV-RUN: sanity check only, results are meaningless ***")
        log("=== UTMobileNet sweep ===")
        log(f"dataset={args.dataset} type={args.dataset_type} epochs={args.epochs}")
        if args.limit_batches:
            log(f"limit_batches={args.limit_batches} (each dataloader capped)")
        log(f"models  ({len(args.models)}): {', '.join(args.models)}")
        log(f"windows ({len(args.windows)}): {', '.join(args.windows)}")
        log(f"total pairs: {len(args.models) * len(args.windows)}")
        log(f"log       -> {log_path}")
        log(f"progress  -> {progress_path}"
            f"{' (not recorded for a fast-dev-run)' if args.fast_dev_run else ''}")
        log(f"results   -> {ts.RESULTS_ROOT}")
        log(f"checkpts  -> {ts.CHECKPOINTS_ROOT}")
        log(f"data root -> {ts.DATA_ROOT}")
        log(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
        if args.resume and done_pairs:
            log(f"resuming: {len(done_pairs)} pair(s) already done, will be skipped")
        log()

        ts.print_gpu_info()

        # No scheduler on this box: a co-tenant job holding most of the VRAM will turn
        # every run into a CUDA OOM. Check once up front rather than discovering it 50
        # failures later.
        free_gb, total_gb = gpu_free_gb()
        if free_gb is None:
            log("WARNING: no CUDA device visible - runs will fall back to CPU")
        else:
            log(f"GPU memory: {free_gb:.2f} GiB free of {total_gb:.2f} GiB")
            if args.min_free_gb > 0 and free_gb < args.min_free_gb:
                log(f"ABORTING: only {free_gb:.2f} GiB free, need {args.min_free_gb:.2f} GiB.")
                log("Another process is most likely holding the GPU. Check 'nvidia-smi', "
                    "pick a free device via CUDA_VISIBLE_DEVICES, or lower --min_free_gb "
                    "to override.")
                return 1
        log()

        outcomes = []  # (window, model, status, elapsed, reason)
        consecutive_failures = 0
        aborted = False
        sweep_start = time.time()

        for window in args.windows:
            if aborted:
                break

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
                traceback.print_exc()
                for model in args.models:
                    outcomes.append((window, model, 'LOAD FAILED', 0, str(e)[:300]))
                continue

            if not datasets:
                # prepare_datasets prints its own "Files not found" line and returns {}.
                log(f"SKIPPED window {window}: no data under {ts.DATA_ROOT} for this window")
                for model in args.models:
                    outcomes.append((window, model, 'SKIPPED (no data)', 0, ''))
                continue

            for key, (X, y) in datasets.items():
                log(f"loaded '{key}': X={X.shape} y={y.shape}")

            for model in args.models:
                pair = f"{window}|{model}"
                if pair in done_pairs:
                    log(f"--- {window} / {model}: already done, skipping ---")
                    outcomes.append((window, model, 'SKIPPED (resume)', 0, ''))
                    continue

                log("-" * 70)
                free_gb, _ = gpu_free_gb()
                free_note = f" | GPU free {free_gb:.2f} GiB" if free_gb is not None else ""
                log(f"RUN {window} / {model}{free_note}")
                log("-" * 70)

                run_start = time.time()
                results = None
                try:
                    results = ts.evaluate_models_on_datasets(
                        datasets_dict=datasets,
                        model_types=[model],
                        use_class_weights=args.use_class_weights,
                        dataset_name=args.dataset,
                        feature_name=args.feature_name,
                        num_epochs=args.epochs,
                        limit_batches=args.limit_batches,
                    )
                    elapsed = time.time() - run_start

                    # A returned frame is not success - the library swallows its own
                    # exceptions and reports them inside the frame.
                    ok, why = inspect_results(results)
                    if ok:
                        log(f"OK {window} / {model} in {timedelta(seconds=int(elapsed))}")
                        print(results.to_string(index=False), flush=True)
                        outcomes.append((window, model, 'OK', elapsed, ''))
                        consecutive_failures = 0
                        # A fast-dev-run pair trained for 1 epoch on 2 batches. Recording
                        # it would make a later --resume skip it as finished, so the pair
                        # would never actually be trained - the exact silent false success
                        # this script exists to catch. Smoke tests stay out of the ledger.
                        if not args.fast_dev_run:
                            with open(progress_path, 'a') as pfh:
                                pfh.write(pair + "\n")
                    else:
                        reason = failure_reason(model) or why
                        status = classify(reason)
                        log(f"{status} {window} / {model} after "
                            f"{timedelta(seconds=int(elapsed))}: {reason}")
                        print(results.to_string(index=False), flush=True)
                        outcomes.append((window, model, status, elapsed, reason))
                        consecutive_failures += 1
                        # Deliberately NOT recorded in the progress file, so --resume
                        # retries it instead of skipping it as finished.

                except Exception as e:
                    elapsed = time.time() - run_start
                    reason = f"{type(e).__name__}: {e}"[:300]
                    status = classify(reason)
                    log(f"{status} {window} / {model} after "
                        f"{timedelta(seconds=int(elapsed))}: {reason}")
                    # Full traceback, through the tee, so the log keeps the stack rather
                    # than just the one-line message.
                    traceback.print_exc()
                    sys.stderr.flush()
                    outcomes.append((window, model, status, elapsed, reason))
                    consecutive_failures += 1
                    # Not written to the progress file, so --resume retries this pair.
                    # No re-raise: the next pair gets its shot.

                finally:
                    # Runs on the OOM path too. evaluate_models_on_datasets cleans up its
                    # own model and loaders, but the results frame and any CUDA blocks
                    # freed inside it are only reclaimed once we drop our reference and
                    # ask the allocator to release its cache - otherwise the reserved
                    # memory carries into the next pair and makes an OOM more likely.
                    del results
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                if args.fast_dev_run:
                    # One configuration is the whole point; whether it passed or failed
                    # is reported by the summary below.
                    log()
                    log("fast-dev-run: one configuration attempted, stopping.")
                    aborted = True
                    break

                if (args.max_consecutive_failures > 0
                        and consecutive_failures >= args.max_consecutive_failures):
                    log()
                    log(f"ABORTING: {consecutive_failures} consecutive failures. "
                        "Something environmental is wrong (GPU held by another process, "
                        "or leaked file descriptors) - continuing would only produce more "
                        "bad rows. Fix the cause and re-run with --resume.")
                    aborted = True
                    break

            # Reached on both paths out of the model loop - normal completion and the
            # abort break. The window's X/y arrays are the largest objects the sweep
            # holds (CESNET reaches several GB), and without this they would stay alive
            # while the next window is being parsed, doubling peak host memory.
            for _key in list(datasets):
                datasets[_key] = None
            datasets.clear()
            del datasets, X, y
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        total = time.time() - sweep_start
        # fast-dev-run sets `aborted` to stop after one pair; that is a planned stop,
        # not a failure, so it must not read as ABORTED or force a non-zero exit.
        really_aborted = aborted and not args.fast_dev_run
        log()
        log("=" * 70)
        if args.fast_dev_run:
            headline = 'FAST-DEV-RUN DONE'
        else:
            headline = 'SWEEP ABORTED' if really_aborted else 'SWEEP DONE'
        log(f"{headline} in {timedelta(seconds=int(total))}")
        log("=" * 70)

        attempted = [o for o in outcomes if not o[2].startswith('SKIPPED')]
        ok = [o for o in outcomes if o[2] == 'OK']
        log(f"{len(ok)}/{len(attempted)} attempted pair(s) produced real metrics "
            f"({len(outcomes) - len(attempted)} skipped)")

        by_status = collections.Counter(o[2] for o in outcomes)
        for status, count in by_status.most_common():
            log(f"  {status:<24} {count}")
        log()

        log(f"{'window':<8} {'model':<12} {'status':<18} {'elapsed':<10} reason")
        for window, model, status, elapsed, reason in outcomes:
            log(f"{window:<8} {model:<12} {status:<18} "
                f"{str(timedelta(seconds=int(elapsed))):<10} {reason}")
        log()
        log(f"per-run CSVs are under {os.path.join(ts.RESULTS_ROOT, args.dataset, 'direction')}")
        log(f"full log: {log_path}")

        if args.fast_dev_run:
            if ok:
                log("sanity check PASSED - the full path runs end to end.")
            else:
                log("sanity check FAILED - fix this before starting a real sweep.")
        return 1 if (really_aborted or not ok) else 0

    finally:
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        fh.close()


if __name__ == '__main__':
    sys.exit(main())
