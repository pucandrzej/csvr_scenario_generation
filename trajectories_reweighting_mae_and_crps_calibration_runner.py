"""Run compact MAE/CRPS calibration grid through the diagnostic script."""

import argparse
import itertools
import os
import subprocess
import sys

from config.paths import MAE_CRPS_RESULTS_DIR
from config.trading_strategies_calibration_config import (
    bands_grid_config,
    median_grid_config,
)


MODEL_CONFIGS = [
    ("_hist_insample_None_False_None", "MULTI_prediction"),
    ("_weather_scenarios_None_False_None", "MULTI_prediction"),
    ("_hist_insample_None_True_dual_coeff", "MULTI_prediction"),
    ("_weather_scenarios_None_True_dual_coeff", "MULTI_prediction"),
    ("_____None____", "benchmark_prediction"),
]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid_source", default="median", choices=["median", "bands"])
    parser.add_argument("--weights_method", default="all", choices=["all", "mae", "kernel"])
    parser.add_argument("--run_type", default="calibration")
    parser.add_argument("--processes", default="32")
    parser.add_argument("--model_setting", default=None)
    parser.add_argument("--underlying_model_column", default=None)
    parser.add_argument("--delivery_index", default=None)
    parser.add_argument("--daily_file", default=None)
    parser.add_argument("--p_values", default=None)
    parser.add_argument("--lambda_values", default=None)
    parser.add_argument("--max_grid_rows", default=None, type=int)
    parser.add_argument(
        "--output_file",
        default="calibration_dynamic_reweighting_forecast_diagnostics.csv",
    )
    return parser.parse_args()


def _parse_float_list(values):
    if values is None:
        return None
    return [float(value) for value in values.split(",")]


def grid_rows(grid_source, weights_method, p_values=None, lambda_values=None):
    config = bands_grid_config if grid_source == "bands" else median_grid_config
    p_list = p_values or config["p_list"]
    lambda_list = lambda_values or config["lambda_list"]
    rows = []

    if weights_method in ["all", "kernel"]:
        rows.extend(
            ("kernel", p, lambda_)
            for p, lambda_ in itertools.product(p_list, lambda_list)
        )
    if weights_method in ["all", "mae"]:
        rows.append(("mae", "nan", "nan"))

    return rows


def model_configs(args):
    if args.model_setting is None:
        return MODEL_CONFIGS

    column = args.underlying_model_column
    if column is None:
        column = (
            "benchmark_prediction"
            if args.model_setting == "_____None____"
            else "MULTI_prediction"
        )
    return [(args.model_setting, column)]


def main():
    args = parse_args()
    rows = grid_rows(
        args.grid_source,
        args.weights_method,
        p_values=_parse_float_list(args.p_values),
        lambda_values=_parse_float_list(args.lambda_values),
    )
    if args.max_grid_rows is not None:
        rows = rows[: args.max_grid_rows]

    completed_path = os.path.join(MAE_CRPS_RESULTS_DIR, "calibration_dynamic_reweighting_forecast.csv")
    with open(completed_path) as file:
        completed = file.read()
    append_output = os.path.exists(os.path.join(MAE_CRPS_RESULTS_DIR, args.output_file))
    for model_setting, column_name in model_configs(args):
        for weights_method, p, lambda_ in rows:
            params = ("", "") if weights_method == "mae" else (float(p), float(lambda_))
            config = f"{args.run_type},{params[0]},{params[1]},{weights_method},{model_setting},{column_name},"
            if f"\n{config}" in completed:
                print(f"Skipping existing: {model_setting}, {weights_method}, {p}, {lambda_}", flush=True)
                continue

            cmd = [
                sys.executable,
                "-m",
                "Forecasting_results_analysis.dynamic_reweighting_forecast_diagnostics",
                "--model_setting",
                model_setting,
                "--underlying_model_column",
                column_name,
                "--run_type",
                args.run_type,
                "--weights_method",
                weights_method,
                "--distribution_param",
                str(p),
                "--lambda_parameter",
                str(lambda_),
                "--aggregate_over_t0",
                "--output_file",
                args.output_file,
                "--processes",
                str(args.processes),
            ]
            if append_output:
                cmd.append("--append_output")
            if args.delivery_index is not None:
                cmd.extend(["--delivery_index", str(args.delivery_index)])
            if args.daily_file is not None:
                cmd.extend(["--daily_file", args.daily_file])

            print(" ".join(cmd), flush=True)
            subprocess.run(cmd, check=True)
            append_output = True

    print(
        "Saved calibration diagnostics to "
        f"{os.path.join('MAE_CRPS_RESULTS', args.output_file)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
