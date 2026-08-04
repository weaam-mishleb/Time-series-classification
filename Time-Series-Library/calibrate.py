"""
Single-run calibration entry point.

main_program.py hardcodes its dataset/window/model choices and exposes no CLI, so
there was no way to time one configuration without editing it. This script drives the
same functions (prepare_datasets -> evaluate_models_on_datasets) for exactly one
dataset/window/model combination, with the epoch count exposed as a flag.

Purpose is measurement: run it with --train_epochs 1 to get a real per-epoch time on
this machine before committing to a full sweep.

Example:
    CUDA_VISIBLE_DEVICES=2 python calibrate.py \
        --dataset UTMobileNet --window 50ms --model timesnet --train_epochs 1
"""
import argparse
import time
from datetime import timedelta

import TimeSeries_functions as ts


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', default='UTMobileNet',
                        choices=['UTMobileNet', 'Cesnet', 'VisQuic', 'QuicText'])
    parser.add_argument('--dataset_type', default='dir',
                        help="'dir' for direction features, 'balanced' for the balanced CESNET set")
    parser.add_argument('--window', default='50ms',
                        help='small window size, e.g. 5ms / 50ms / 250ms')
    parser.add_argument('--model', default='timesnet',
                        choices=['nst', 'informer', 'timesnet', 'autoformer', 'fedformer', 'timemixer'])
    parser.add_argument('--train_epochs', type=int, default=1,
                        help='max epochs; early stopping may end the run sooner')
    parser.add_argument('--feature_name', default='vec[upstream,downstream,ratio]')
    parser.add_argument('--use_class_weights', action='store_true')
    args = parser.parse_args()

    print(f"=== calibration run ===")
    print(f"dataset={args.dataset} type={args.dataset_type} window={args.window} "
          f"model={args.model} epochs={args.train_epochs}")
    print(f"checkpoints -> {ts.CHECKPOINTS_ROOT}")
    print(f"results     -> {ts.RESULTS_ROOT}")
    print(f"data root   -> {ts.DATA_ROOT}")

    ts.print_gpu_info()

    load_start = time.time()
    datasets = ts.prepare_datasets(
        dataset_name=args.dataset,
        small_windows=[args.window],
        dataset_type=args.dataset_type,
        selected_feature_indices=[1, 4, 6],  # upstream, downstream, ratio
    )
    print(f"\nDataset load took {timedelta(seconds=int(time.time() - load_start))}")

    if not datasets:
        raise SystemExit(
            f"No dataset found for {args.dataset}/{args.window} under {ts.DATA_ROOT}. "
            "Nothing to run - check the window name and the data root."
        )

    for key, (X, y) in datasets.items():
        print(f"loaded '{key}': X={X.shape} y={y.shape}")

    run_start = time.time()
    results = ts.evaluate_models_on_datasets(
        datasets_dict=datasets,
        model_types=[args.model],
        use_class_weights=args.use_class_weights,
        dataset_name=args.dataset,
        feature_name=args.feature_name,
        num_epochs=args.train_epochs,
    )
    total = time.time() - run_start

    print(f"\n=== done in {timedelta(seconds=int(total))} "
          f"for {args.train_epochs} epoch(s) ===")
    print(results)


if __name__ == '__main__':
    main()
