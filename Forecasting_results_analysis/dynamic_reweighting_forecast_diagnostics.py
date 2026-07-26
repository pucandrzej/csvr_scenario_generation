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

from config.test_calibration_validation import validation_window_length
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
EXPECTED_DAYS_PER_DELIVERY = validation_window_length
TAUS = np.linspace(0.01, 0.99, 99)
METHOD_NAMES = {
    "raw": "Raw ensemble",
    "mae": "MAE weighting",
    "kernel": "Kernel weighting",
}
ALL_Q = np.union1d(TAUS, [0.5])          # sorted union; includes median
_MEDIAN_POS = int(np.searchsorted(ALL_Q, 0.5))
_TAU_POS = np.searchsorted(ALL_Q, TAUS)

def _validate_completeness(tasks, delivery_index, daily_file):
    """Raise if the task set doesn't cover the full expected grid of
    96 deliveries (indices 0-95) x 366 daily files each.

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

    bad = [
        (dir_name, len(daily_files))
        for _, dir_name, daily_files, _, _ in tasks
        if len(daily_files) != EXPECTED_DAYS_PER_DELIVERY
    ]
    if bad:
        raise ValueError(
            f"{len(bad)} deliveries have wrong file count "
            f"(expected {EXPECTED_DAYS_PER_DELIVERY} each): {bad}"
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
    delivery_index=None,
    daily_file=None,
):
    """Build one independent worker task per delivery directory."""
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
                (results_dir, dir_name, daily_files, column_name, calibration_params)
            )

        if delivery_index is not None and current_delivery == delivery_index:
            break

    return selected


def _dynamic_weights(observed_actual, observed_forecast, params, weights_method):
    residuals = np.median(observed_forecast, axis=1) - observed_actual
    _, nonzero_mae = get_trust_threshold(
        residuals,
        params["trust_threshold_method"],
    )
    return compute_weights(
        observed_forecast,
        observed_actual,
        nonzero_mae,
        params["p"],
        params["lambda_"],
        weights_method,
    )


def _empty_loss_accumulators():
    return {
        t0: {
            method: {"mae_sum": 0.0, "mae_count": 0, "crps_sum": 0.0, "crps_count": 0}
            for method in METHOD_NAMES
        }
        for t0 in range(30)
    }


def _check_weights(w_raw, w_mae, w_kernel, n_scenarios):
    assert len(w_raw) == n_scenarios
    assert len(w_mae) == n_scenarios
    assert len(w_kernel) == n_scenarios
    assert np.isclose(w_raw.sum(), 1)
    assert np.isclose(w_mae.sum(), 1)
    assert np.isclose(w_kernel.sum(), 1)
    assert np.all(np.isfinite(w_mae))
    assert np.all(np.isfinite(w_kernel))


def _evaluate_delivery(task):
    """Evaluate all selected days of one delivery and return loss sums/counts."""
    results_dir, dir_name, daily_files, column_name, calibration_params = task
    losses = _empty_loss_accumulators()
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

            w_mae = _dynamic_weights(
                observed_actual, observed_forecast, calibration_params["mae"], "mae"
            )
            w_kernel = _dynamic_weights(
                observed_actual, observed_forecast, calibration_params["kernel"], "kernel"
            )
            _check_weights(w_raw, w_mae, w_kernel, n_scenarios)

            weights_by_method = {
                "raw": w_raw,
                "mae": w_mae,
                "kernel": w_kernel,
            }

            for actual, forecasts in zip(future_actual, future_forecast):
                for method, weights in weights_by_method.items():
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
    total = _empty_loss_accumulators()
    evaluated_cases = []
    for losses, cases in results:
        evaluated_cases.extend(cases)
        for t0 in range(30):
            for method in METHOD_NAMES:
                for key in ("mae_sum", "mae_count", "crps_sum", "crps_count"):
                    total[t0][method][key] += losses[t0][method][key]
    return total, evaluated_cases


def calculate_diagnostics(
    model_setting,
    column_name,
    calibration_params,
    run_type,
    processes=1,
    delivery_index=None,
    daily_file=None,
):
    """Evaluate diagnostics in parallel over delivery directories."""
    tasks = _build_delivery_tasks(
        model_setting,
        column_name,
        calibration_params,
        run_type,
        delivery_index=delivery_index,
        daily_file=daily_file,
    )
    if not tasks:
        raise ValueError(
            f"No {run_type} forecast cases found for "
            f"model_setting={model_setting}, model={column_name}, "
            f"delivery_index={delivery_index}, daily_file={daily_file}"
        )

    _validate_completeness(tasks, delivery_index, daily_file)

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
        for method in METHOD_NAMES:
            acc = losses[t0][method]
            row[f"mae_{method}"] = acc["mae_sum"] / acc["mae_count"]
            row[f"crps_{method}"] = acc["crps_sum"] / acc["crps_count"]

        row["mae_mae_improvement_pct"] = 100 * (
            row["mae_raw"] - row["mae_mae"]
        ) / row["mae_raw"]
        row["mae_kernel_improvement_pct"] = 100 * (
            row["mae_raw"] - row["mae_kernel"]
        ) / row["mae_raw"]
        row["crps_mae_improvement_pct"] = 100 * (
            row["crps_raw"] - row["crps_mae"]
        ) / row["crps_raw"]
        row["crps_kernel_improvement_pct"] = 100 * (
            row["crps_raw"] - row["crps_kernel"]
        ) / row["crps_raw"]
        rows.append(row)

    df = pd.DataFrame(rows)
    df.attrs["evaluated_cases"] = evaluated_cases
    return df


def write_figure(df, figure_path, title):
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=("MAE", "CRPS"),
    )

    for method, label in METHOD_NAMES.items():
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
    df = calculate_diagnostics(
        args.model_setting,
        column_name,
        calibration_params,
        args.run_type,
        processes=args.processes,
        delivery_index=args.delivery_index,
        daily_file=args.daily_file,
    )

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
    write_figure(
        df,
        figure_path,
        (
            "Dynamic Scenario Reweighting Forecast Diagnostics: "
            f"{args.strategy_model}, {args.model_setting}, {column_name}"
        ),
    )

    print(f"Saved diagnostics CSV: {csv_path}")
    print(f"Saved diagnostics figure: {figure_path}")
    print("Evaluated forecast cases:")
    for delivery_dir, daily_file_name in df.attrs["evaluated_cases"]:
        print(f"  {delivery_dir} / {daily_file_name}")
    print("Best calibrated parameters:")
    for method, params in calibration_params.items():
        print(f"  {method}: {params}")


if __name__ == "__main__":
    main()