"""Calibrate kernel weights on calibration CSVs, then validate both optima."""

import argparse
import itertools
import os

import pandas as pd

from config.paths import MAE_CRPS_RESULTS_DIR, RAW_MAE_CRPS_RESULTS_DIR
from config.trading_strategies_calibration_config import (
    bands_grid_config,
    median_grid_config,
)
from Forecasting_results_analysis.dynamic_reweighting_forecast_diagnostics import (
    evaluate_parameter_grid,
)


MODEL_CONFIGS = [
    ("_hist_insample_None_False_None", "MULTI_prediction"),
    ("_weather_scenarios_None_False_None", "MULTI_prediction"),
    ("_hist_insample_None_True_dual_coeff", "MULTI_prediction"),
    ("_weather_scenarios_None_True_dual_coeff", "MULTI_prediction"),
    ("_____None____", "benchmark_prediction"),
]


def parse_args():
    """Parse grid, model-selection, process-count, and output-file options."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid_source", default="median", choices=["median", "bands"])
    parser.add_argument("--processes", default=32, type=int)
    parser.add_argument("--special_results_directory")
    parser.add_argument(
        "--model_setting",
        choices=[model_setting for model_setting, _ in MODEL_CONFIGS],
    )
    parser.add_argument(
        "--calibration_file",
        default=None,
    )
    parser.add_argument(
        "--validation_file",
        default=None,
    )
    parser.add_argument(
        "--validation_t0_file",
        default=None,
    )
    args = parser.parse_args()
    suffix = "_bands" if args.grid_source == "bands" else ""
    args.calibration_file = args.calibration_file or (
        f"calibration_dynamic_reweighting_forecast{suffix}.csv"
    )
    args.validation_file = args.validation_file or (
        f"validation_dynamic_reweighting_forecast{suffix}.csv"
    )
    args.validation_t0_file = args.validation_t0_file or (
        f"validation_dynamic_reweighting_forecast_by_t0{suffix}.csv"
    )
    return args


def parameter_grid(grid_source):
    """Return every kernel ``(method, p, lambda)`` tuple for a named grid."""
    config = bands_grid_config if grid_source == "bands" else median_grid_config
    return [
        ("kernel", float(p), float(lambda_))
        for p, lambda_ in itertools.product(config["p_list"], config["lambda_list"])
    ]


def result_path(filename, special_results_directory=None):
    """Resolve relative result filenames inside the MAE/CRPS output directory."""
    if os.path.isabs(filename):
        return filename
    directory = MAE_CRPS_RESULTS_DIR
    if special_results_directory:
        directory = os.path.join(special_results_directory, RAW_MAE_CRPS_RESULTS_DIR)
    return os.path.join(directory, filename)


def parameter_key(method, p, lambda_):
    """Create a stable, hashable key for resume checks."""
    def number(value):
        """Normalize CSV numbers and missing values for key comparison."""
        return None if pd.isna(value) else float(value)

    return method, number(p), number(lambda_)


def completed_parameters(path, model_setting, column_name):
    """Read the parameter keys already calibrated for one model."""
    if not os.path.exists(path):
        return set()
    results = pd.read_csv(path)
    rows = results[
        (results["run_type"] == "calibration")
        & (results["model_setting"] == model_setting)
        & (results["model"] == column_name)
    ]
    return {
        parameter_key(row.weights, row.param2, row.param3)
        for row in rows.itertuples()
    }


def append_results(results, path):
    """Append new calibration rows, writing a header only for a new file."""
    append = os.path.exists(path)
    results.to_csv(path, mode="a" if append else "w", header=not append, index=False)


def replace_model_results(results, path):
    """Replace selected models while preserving other validation rows."""
    if os.path.exists(path):
        previous = pd.read_csv(path)
        previous = previous[
            ~previous["model_setting"].isin(results["model_setting"].unique())
        ]
        results = pd.concat([previous, results], ignore_index=True)
    results.to_csv(path, index=False)


def best_kernel_parameters(calibration, model_setting, column_name):
    """Select separate kernel parameters minimizing MAE and CRPS."""
    rows = calibration[
        (calibration["run_type"] == "calibration")
        & (calibration["model_setting"] == model_setting)
        & (calibration["model"] == column_name)
        & (calibration["weights"] == "kernel")
    ]
    if rows.empty:
        raise ValueError(f"No kernel calibration results for {model_setting}")

    best_mae = rows.loc[rows["mae_weighted"].idxmin()]
    best_crps = rows.loc[rows["crps_weighted"].idxmin()]
    return [
        ("mae", ("kernel", float(best_mae.param2), float(best_mae.param3))),
        ("crps", ("kernel", float(best_crps.param2), float(best_crps.param3))),
    ]


def main():
    """Start or resume calibration, select both optima, and run their validation."""
    args = parse_args()
    calibration_path = result_path(args.calibration_file, args.special_results_directory)
    validation_path = result_path(args.validation_file, args.special_results_directory)
    validation_t0_path = result_path(args.validation_t0_file, args.special_results_directory)
    for path in [calibration_path, validation_path, validation_t0_path]:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    parameters = parameter_grid(args.grid_source)
    model_configs = [
        config
        for config in MODEL_CONFIGS
        if args.model_setting is None or config[0] == args.model_setting
    ]

    for model_setting, column_name in model_configs:
        completed = completed_parameters(calibration_path, model_setting, column_name)
        pending = [
            parameter
            for parameter in parameters
            if parameter_key(*parameter) not in completed
        ]
        if not pending:
            print(f"Calibration already complete: {model_setting}", flush=True)
            continue

        print(f"Calibrating {model_setting}: {len(pending)} settings", flush=True)
        results = evaluate_parameter_grid(
            model_setting,
            column_name,
            pending,
            run_type="calibration",
            processes=args.processes,
            special_results_directory=args.special_results_directory,
        )
        append_results(results, calibration_path)

    calibration = pd.read_csv(calibration_path)
    validation_results = []
    validation_t0_results = []
    for model_setting, column_name in model_configs:
        selected = best_kernel_parameters(calibration, model_setting, column_name)
        print(f"Validating {model_setting}: MAE-best and CRPS-best", flush=True)
        results, per_t0 = evaluate_parameter_grid(
            model_setting,
            column_name,
            [parameter for _, parameter in selected],
            run_type="test",
            processes=args.processes,
            return_per_t0=True,
            special_results_directory=args.special_results_directory,
        )
        results.insert(0, "selected_by", [metric for metric, _ in selected])
        per_t0.insert(
            0,
            "selected_by",
            [metric for metric, _ in selected for _ in range(30)],
        )
        validation_results.append(results)
        validation_t0_results.append(per_t0)

    replace_model_results(
        pd.concat(validation_results, ignore_index=True),
        validation_path,
    )
    replace_model_results(
        pd.concat(validation_t0_results, ignore_index=True),
        validation_t0_path,
    )
    print(f"Calibration results: {calibration_path}")
    print(f"Validation results: {validation_path}")
    print(f"Validation by t0: {validation_t0_path}")


if __name__ == "__main__":
    main()
