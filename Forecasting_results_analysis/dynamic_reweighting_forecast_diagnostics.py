"""Forecast diagnostics for dynamic scenario reweighting.

The diagnostic compares raw ensemble, inverse-MAE, and kernel scenario weights
at each information time t0, evaluating only the future path t0+1..30.
"""

import argparse
import os
from multiprocessing import Pool
from tqdm import tqdm

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config.test_calibration_validation import (
    calibration_window_length,
    validation_window_length,
)
from config.forecasting_simulation_config import deliveries_no

from config.paths import (
    BENCHMARK_RESULTS_DIR,
    CALIBRATION_STRATEGIES_MEASURES_DIR,
    MAE_CRPS_RESULTS_DIR,
    MODEL_RESULTS_DIR,
    PAPER_FIGURES_DIR,
)
from Forecasting_results_analysis.forecasting_results_utils import (
    analysis_pinball_loss,
)
from Trading_strategies.strategies_utils import (
    compute_weights,
    get_trust_threshold,
    batch_weighted_quantiles,
)

EXPECTED_N_DELIVERIES = deliveries_no
EXPECTED_DELIVERY_INDICES = list(range(deliveries_no))  # 0..95
EXPECTED_DAYS_PER_DELIVERY = {
    "calibration": calibration_window_length,
    "test": validation_window_length,
}
TAUS = np.linspace(0.01, 0.99, 99)
METHOD_NAMES = {
    "raw": "Raw ensemble",
    "mae": "MAE weighting",
    "kernel": "Kernel weighting",
}
ALL_Q = np.union1d(TAUS, [0.5])          # sorted union; includes median
_MEDIAN_POS = int(np.searchsorted(ALL_Q, 0.5))
_TAU_POS = np.searchsorted(ALL_Q, TAUS)

def _validate_completeness(tasks, delivery_index, daily_file, run_type):
    """Raise if the task set doesn't cover the full expected grid of
    96 deliveries (indices 0-95) x the expected daily files per run type.

    Skipped when a manual subsetting flag is active, since those
    deliberately produce a partial task set.
    """
    if delivery_index is not None or daily_file is not None:
        return

    if len(tasks) != EXPECTED_N_DELIVERIES:
        found_indices = sorted(int(dir_name.split("_")[3]) for _, dir_name, *_ in tasks)
        missing = sorted(set(EXPECTED_DELIVERY_INDICES) - set(found_indices))
        extra = sorted(set(found_indices) - set(EXPECTED_DELIVERY_INDICES))
        raise ValueError(
            f"Expected {EXPECTED_N_DELIVERIES} deliveries (indices 0-95), "
            f"found {len(tasks)}. Missing: {missing}. Unexpected: {extra}."
        )

    found_indices = sorted(int(dir_name.split("_")[3]) for _, dir_name, *_ in tasks)
    if found_indices != EXPECTED_DELIVERY_INDICES:
        missing = sorted(set(EXPECTED_DELIVERY_INDICES) - set(found_indices))
        duplicates = sorted({i for i in found_indices if found_indices.count(i) > 1})
        raise ValueError(
            f"Delivery indices don't match 0-95 exactly. "
            f"Missing: {missing}. Duplicated: {duplicates}."
        )

    expected_days = EXPECTED_DAYS_PER_DELIVERY.get(run_type)
    if expected_days is None:
        raise ValueError(f"Unsupported run_type for completeness check: {run_type}")

    bad = [
        (dir_name, len(daily_files))
        for _, dir_name, daily_files, *_ in tasks
        if len(daily_files) != expected_days
    ]
    if bad:
        raise ValueError(
            f"{len(bad)} deliveries have wrong file count "
            f"(expected {expected_days} each): {bad}"
        )


def _default_column(model_setting):
    return "benchmark_prediction" if model_setting == "_____None____" else "MULTI_prediction"


def _calibration_filename(one_sided, model, band_type):
    strategy_direction = -1 if one_sided else 0
    return (
        f"calibration_trading_strategy_measures_"
        f"{one_sided}_{strategy_direction}_{model}_{band_type}.csv"
    )


def _safe_name(value):
    return (
        str(value)
        .replace(os.sep, "_")
        .replace("/", "_")
        .replace(" ", "_")
        .replace(";", "_")
        .replace(":", "_")
    )


def _as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _load_best_calibrated_params(calibration_path, model_setting, column_name):
    df = pd.read_csv(calibration_path)
    best_params = {}

    for weights_method in ["mae", "kernel"]:
        sub = df[
            (df["model_setting"] == model_setting)
            & (df["model"] == column_name)
            & (df["weights"] == weights_method)
        ]
        if sub.empty:
            raise ValueError(
                "No calibration rows found for "
                f"model_setting={model_setting}, model={column_name}, "
                f"weights={weights_method} in {calibration_path}"
            )

        best = sub.loc[sub["Sortino_ratio"].idxmax()]
        best_params[weights_method] = {
            "p": _as_float(best["param2"]),
            "lambda_": _as_float(best["param3"]),
            "trust_threshold_method": best["threshold"],
            "sortino": best["Sortino_ratio"],
        }

    return best_params


def _results_dir(column_name):
    return (
        BENCHMARK_RESULTS_DIR
        if column_name == "benchmark_prediction"
        else MODEL_RESULTS_DIR
    )


def _build_delivery_tasks(
    model_setting,
    column_name,
    calibration_params,
    run_type,
    methods=None,
    delivery_index=None,
    daily_file=None,
):
    """Build one independent worker task per delivery directory."""
    methods = tuple(methods or METHOD_NAMES.keys())
    results_dir = _results_dir(column_name)
    delivery_directories = sorted(
        (d for d in os.listdir(results_dir) if d.endswith(model_setting)),
        key=lambda d: int(d.split("_")[3]),
    )

    selected = []

    for dir_name in delivery_directories:
        current_delivery = int(dir_name.split("_")[3])
        if delivery_index is not None and current_delivery != delivery_index:
            continue

        daily_files = []
        for filename in sorted(
            f for f in os.listdir(os.path.join(results_dir, dir_name))
            if f.startswith(f"{run_type}_")
        ):
            if daily_file is not None and filename != daily_file:
                continue

            daily_files.append(filename)

        if daily_files:
            selected.append(
                (
                    results_dir,
                    dir_name,
                    daily_files,
                    column_name,
                    calibration_params,
                    methods,
                )
            )

        if delivery_index is not None and current_delivery == delivery_index:
            break

    return selected


def _dynamic_weights(observed_actual, observed_forecast, params, weights_method):
    residuals = np.median(observed_forecast, axis=1) - observed_actual
    _, nonzero_mae = get_trust_threshold(
        residuals,
        params.get("trust_threshold_method", "mae"),
    )
    return compute_weights(
        observed_forecast,
        observed_actual,
        nonzero_mae,
        params["p"],
        params["lambda_"],
        weights_method,
    )


def _empty_loss_accumulators(methods=None):
    methods = tuple(methods or METHOD_NAMES.keys())
    return {
        t0: {
            method: {"mae_sum": 0.0, "mae_count": 0, "crps_sum": 0.0, "crps_count": 0}
            for method in methods
        }
        for t0 in range(30)
    }


def _check_weight(weights, n_scenarios):
    assert len(weights) == n_scenarios
    assert np.isclose(weights.sum(), 1)
    assert np.all(np.isfinite(weights))


def _evaluate_delivery(task):
    """Evaluate all selected days of one delivery and return loss sums/counts."""
    results_dir, dir_name, daily_files, column_name, calibration_params, methods = task
    losses = _empty_loss_accumulators(methods)
    evaluated_cases = []

    for daily_file_name in daily_files:
        df = pd.read_csv(
            os.path.join(results_dir, dir_name, daily_file_name), index_col=0
        )
        y_actual = df["actual"].to_numpy()
        y_forecast = df[
            [
                c for c in df.columns
                if c.startswith(column_name) and "base_path" not in c
            ]
        ].to_numpy()

        assert y_actual.shape[0] == 31
        assert y_forecast.shape[0] == 31
        n_scenarios = y_forecast.shape[1]
        w_raw = np.ones(n_scenarios) / n_scenarios

        for t0 in range(30):
            observed_actual = y_actual[: t0 + 1]
            observed_forecast = y_forecast[: t0 + 1, :]
            future_actual = y_actual[t0 + 1 :]
            future_forecast = y_forecast[t0 + 1 :, :]
            assert len(future_actual) == 30 - t0

            weights_by_method = {"raw": w_raw}
            for method in methods:
                if method == "raw":
                    continue
                weights_by_method[method] = _dynamic_weights(
                    observed_actual,
                    observed_forecast,
                    calibration_params[method],
                    method,
                )

            for weights in weights_by_method.values():
                _check_weight(weights, n_scenarios)

            for actual, forecasts in zip(future_actual, future_forecast):
                for method in methods:
                    weights = weights_by_method[method]
                    acc = losses[t0][method]

                    q_hats = batch_weighted_quantiles(forecasts, weights, ALL_Q)

                    median = q_hats[_MEDIAN_POS]
                    acc["mae_sum"] += abs(actual - median)
                    acc["mae_count"] += 1

                    for tau, q_hat in zip(TAUS, q_hats[_TAU_POS]):
                        acc["crps_sum"] += analysis_pinball_loss(actual, q_hat, tau)
                        acc["crps_count"] += 1

        evaluated_cases.append((dir_name, daily_file_name))

    return losses, evaluated_cases


def _merge_losses(results):
    methods = results[0][0][0].keys()
    total = _empty_loss_accumulators(methods)
    evaluated_cases = []
    for losses, cases in results:
        evaluated_cases.extend(cases)
        for t0 in range(30):
            for method in methods:
                for key in ("mae_sum", "mae_count", "crps_sum", "crps_count"):
                    total[t0][method][key] += losses[t0][method][key]
    return total, evaluated_cases


def calculate_diagnostics(
    model_setting,
    column_name,
    calibration_params,
    run_type,
    processes=1,
    methods=None,
    delivery_index=None,
    daily_file=None,
):
    """Evaluate diagnostics in parallel over delivery directories."""
    methods = tuple(methods or METHOD_NAMES.keys())
    if "raw" not in methods:
        methods = ("raw",) + methods

    tasks = _build_delivery_tasks(
        model_setting,
        column_name,
        calibration_params,
        run_type,
        methods=methods,
        delivery_index=delivery_index,
        daily_file=daily_file,
    )
    if not tasks:
        raise ValueError(
            f"No {run_type} forecast cases found for "
            f"model_setting={model_setting}, model={column_name}, "
            f"delivery_index={delivery_index}, daily_file={daily_file}"
        )

    _validate_completeness(tasks, delivery_index, daily_file, run_type)

    if processes == 1:
        results = [
            _evaluate_delivery(task)
            for task in tqdm(tasks, desc="Deliveries", unit="delivery")
        ]
    else:
        with Pool(processes=processes) as pool:
            results = list(
                tqdm(
                    pool.imap(_evaluate_delivery, tasks),
                    total=len(tasks),
                    desc="Deliveries",
                    unit="delivery",
                )
            )

    losses, evaluated_cases = _merge_losses(results)
    rows = []
    for t0 in range(30):
        row = {"t0": t0, "n_future_steps": 30 - t0}
        for method in methods:
            acc = losses[t0][method]
            row[f"mae_{method}"] = acc["mae_sum"] / acc["mae_count"]
            row[f"crps_{method}"] = acc["crps_sum"] / acc["crps_count"]

        for method in methods:
            if method == "raw":
                continue
            row[f"mae_{method}_improvement_pct"] = 100 * (
                row["mae_raw"] - row[f"mae_{method}"]
            ) / row["mae_raw"]
            row[f"crps_{method}_improvement_pct"] = 100 * (
                row["crps_raw"] - row[f"crps_{method}"]
            ) / row["crps_raw"]
        rows.append(row)

    df = pd.DataFrame(rows)
    df.attrs["evaluated_cases"] = evaluated_cases
    return df


def aggregate_over_t0(df, weights_method):
    """Collapse a per-t0 diagnostic table to one compact calibration row."""
    cols = [
        "mae_raw",
        f"mae_{weights_method}",
        "crps_raw",
        f"crps_{weights_method}",
        f"mae_{weights_method}_improvement_pct",
        f"crps_{weights_method}_improvement_pct",
    ]
    return {col: df[col].mean() for col in cols}


def _output_path(filename):
    if os.path.isabs(filename):
        return filename
    return os.path.join(MAE_CRPS_RESULTS_DIR, filename)


def _aggregate_row(df, args, weights_method):
    metrics = aggregate_over_t0(df, weights_method)
    return {
        "run_type": args.run_type,
        "param2": args.distribution_param,
        "param3": args.lambda_parameter,
        "weights": weights_method,
        "model_setting": args.model_setting,
        "model": args.underlying_model_column or _default_column(args.model_setting),
        "mae_raw": metrics["mae_raw"],
        "mae_weighted": metrics[f"mae_{weights_method}"],
        "crps_raw": metrics["crps_raw"],
        "crps_weighted": metrics[f"crps_{weights_method}"],
        "mae_improvement_pct": metrics[f"mae_{weights_method}_improvement_pct"],
        "crps_improvement_pct": metrics[f"crps_{weights_method}_improvement_pct"],
    }


def write_figure(df, figure_path, title, methods=None):
    methods = tuple(
        methods
        or [
            method
            for method in METHOD_NAMES
            if f"mae_{method}" in df.columns and f"crps_{method}" in df.columns
        ]
    )
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=("MAE", "CRPS"),
    )

    for method in methods:
        label = METHOD_NAMES[method]
        fig.add_trace(
            go.Scatter(
                x=df["t0"],
                y=df[f"mae_{method}"],
                mode="lines+markers",
                name=label,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df["t0"],
                y=df[f"crps_{method}"],
                mode="lines+markers",
                name=label,
                showlegend=False,
            ),
            row=2,
            col=1,
        )

    fig.update_layout(
        title=title,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=-0.22, xanchor="center", x=0.5),
        font=dict(size=11),
    )
    fig.update_xaxes(
        title_text="Last observed trajectory step / information time (t0)",
        row=2,
        col=1,
    )
    fig.update_yaxes(title_text="MAE", row=1, col=1)
    fig.update_yaxes(title_text="CRPS", row=2, col=1)
    fig.write_html(figure_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate causal dynamic scenario reweighting diagnostics."
    )
    parser.add_argument("--model_setting", default="_hist_insample_None_True_dual_coeff")
    parser.add_argument("--underlying_model_column", default=None)
    parser.add_argument("--strategy_model", default="median", choices=["median", "bands"])
    parser.add_argument("--one_sided", default=False, action="store_true")
    parser.add_argument("--band_type", default="risk_seeking")
    parser.add_argument("--run_type", default="test")
    parser.add_argument("--calibration_file", default=None)
    parser.add_argument("--weights_method", default=None, choices=["mae", "kernel"])
    parser.add_argument("--distribution_param", default=np.nan, type=float)
    parser.add_argument("--lambda_parameter", default=np.nan, type=float)
    parser.add_argument("--aggregate_over_t0", default=False, action="store_true")
    parser.add_argument("--output_file", default=None)
    parser.add_argument("--append_output", default=False, action="store_true")
    parser.add_argument("--skip_figure", default=False, action="store_true")
    parser.add_argument(
        "--delivery_index",
        default=None,
        type=int,
        help="Optional delivery index from the forecast-result directory name.",
    )
    parser.add_argument(
        "--daily_file",
        default=None,
        help="Optional exact daily forecast CSV filename to evaluate.",
    )
    parser.add_argument(
        "--processes",
        default=32,
        type=int,
        help="Number of parallel delivery workers; use 1 for debugging.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    column_name = args.underlying_model_column or _default_column(args.model_setting)

    if args.weights_method is None:
        calibration_file = args.calibration_file or _calibration_filename(
            args.one_sided,
            args.strategy_model,
            args.band_type,
        )
        calibration_path = os.path.join(
            CALIBRATION_STRATEGIES_MEASURES_DIR,
            calibration_file,
        )
        calibration_params = _load_best_calibrated_params(
            calibration_path,
            args.model_setting,
            column_name,
        )
        methods = tuple(METHOD_NAMES.keys())
    else:
        calibration_params = {
            args.weights_method: {
                "p": args.distribution_param,
                "lambda_": args.lambda_parameter,
            }
        }
        methods = ("raw", args.weights_method)

    df = calculate_diagnostics(
        args.model_setting,
        column_name,
        calibration_params,
        args.run_type,
        processes=args.processes,
        methods=methods,
        delivery_index=args.delivery_index,
        daily_file=args.daily_file,
    )

    if args.aggregate_over_t0:
        if args.weights_method is None:
            raise ValueError("--aggregate_over_t0 requires --weights_method")

        output_file = args.output_file or (
            "calibration_dynamic_reweighting_forecast_diagnostics.csv"
        )
        output_path = _output_path(output_file)
        row = _aggregate_row(df, args, args.weights_method)
        append = args.append_output and os.path.exists(output_path)
        pd.DataFrame([row]).to_csv(
            output_path,
            mode="a" if append else "w",
            header=not append,
            index=False,
        )
        print(f"Saved aggregated diagnostics row: {output_path}")
        return

    identity = (
        f"{args.run_type}_{_safe_name(args.model_setting)}_{_safe_name(column_name)}_"
        f"{args.one_sided}_{args.strategy_model}_{_safe_name(args.band_type)}"
    )
    if args.delivery_index is not None:
        identity += f"_delivery_{args.delivery_index}"
    if args.daily_file is not None:
        identity += f"_{_safe_name(args.daily_file)}"
    csv_path = os.path.join(
        MAE_CRPS_RESULTS_DIR,
        f"dynamic_reweighting_forecast_diagnostics_{identity}.csv",
    )
    figure_path = os.path.join(
        PAPER_FIGURES_DIR,
        f"dynamic_reweighting_forecast_diagnostics_{identity}.html",
    )

    df.to_csv(csv_path, index=False)
    if not args.skip_figure:
        write_figure(
            df,
            figure_path,
            (
                "Dynamic Scenario Reweighting Forecast Diagnostics: "
                f"{args.strategy_model}, {args.model_setting}, {column_name}"
            ),
            methods=methods,
        )

    print(f"Saved diagnostics CSV: {csv_path}")
    if args.skip_figure:
        print("Skipped diagnostics figure")
    else:
        print(f"Saved diagnostics figure: {figure_path}")

    evaluated_cases = df.attrs["evaluated_cases"]
    if (
        args.delivery_index is None
        and args.daily_file is None
    ):
        print(f"Evaluated forecast cases: {len(evaluated_cases)}")
    else:
        print("Evaluated forecast cases:")
        for delivery_dir, daily_file_name in evaluated_cases:
            print(f"  {delivery_dir} / {daily_file_name}")
    print("Weighting parameters:")
    for method in methods:
        if method != "raw":
            print(f"  {method}: {calibration_params[method]}")


if __name__ == "__main__":
    main()
