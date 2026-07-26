"""Forecast diagnostics for dynamic scenario reweighting.

The diagnostic compares raw ensemble, inverse-MAE, and kernel scenario weights
at each information time t0, evaluating only the future path t0+1..30.
"""

import argparse
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
    weighted_median,
    weighted_quantile,
)


TAUS = np.linspace(0.01, 0.99, 99)
METHOD_NAMES = {
    "raw": "Raw ensemble",
    "mae": "MAE weighting",
    "kernel": "Kernel weighting",
}


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


def _iter_forecast_cases(
    model_setting,
    column_name,
    run_type,
    delivery_index=None,
    daily_file=None,
    case_index=None,
    max_cases=None,
):
    results_dir = (
        BENCHMARK_RESULTS_DIR
        if column_name == "benchmark_prediction"
        else MODEL_RESULTS_DIR
    )

    delivery_directories = sorted(
        (d for d in os.listdir(results_dir) if d.endswith(model_setting)),
        key=lambda d: int(d.split("_")[3]),
    )

    selected_cases = 0
    ordered_case_index = 0

    for dir_name in delivery_directories:
        current_delivery_index = int(dir_name.split("_")[3])
        if delivery_index is not None and current_delivery_index != delivery_index:
            continue

        for current_daily_file in sorted(
            f_name
            for f_name in os.listdir(os.path.join(results_dir, dir_name))
            if f_name.startswith(f"{run_type}_")
        ):
            if case_index is not None and ordered_case_index != case_index:
                ordered_case_index += 1
                continue

            ordered_case_index += 1

            if daily_file is not None and current_daily_file != daily_file:
                continue

            df = pd.read_csv(
                os.path.join(results_dir, dir_name, current_daily_file),
                index_col=0,
            )
            actual = df["actual"].values
            forecast = df[
                [
                    c
                    for c in df.columns
                    if c.startswith(column_name) and "base_path" not in c
                ]
            ].values

            yield dir_name, current_daily_file, actual, forecast

            selected_cases += 1
            if max_cases is not None and selected_cases >= max_cases:
                return


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
            "mae": {method: [] for method in METHOD_NAMES},
            "crps": {method: [] for method in METHOD_NAMES},
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


def calculate_diagnostics(
    model_setting,
    column_name,
    calibration_params,
    run_type,
    delivery_index=None,
    daily_file=None,
    case_index=None,
    max_cases=None,
):
    losses = _empty_loss_accumulators()
    cases = 0
    evaluated_cases = []

    for delivery_dir, daily_file_name, y_actual, y_forecast in _iter_forecast_cases(
        model_setting,
        column_name,
        run_type,
        delivery_index=delivery_index,
        daily_file=daily_file,
        case_index=case_index,
        max_cases=max_cases,
    ):
        evaluated_cases.append((delivery_dir, daily_file_name))
        y_actual = np.asarray(y_actual)
        y_forecast = np.asarray(y_forecast)
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
                observed_actual,
                observed_forecast,
                calibration_params["mae"],
                "mae",
            )
            w_kernel = _dynamic_weights(
                observed_actual,
                observed_forecast,
                calibration_params["kernel"],
                "kernel",
            )
            _check_weights(w_raw, w_mae, w_kernel, n_scenarios)

            weights_by_method = {
                "raw": w_raw,
                "mae": w_mae,
                "kernel": w_kernel,
            }

            for actual, forecasts in zip(future_actual, future_forecast):
                for method, weights in weights_by_method.items():
                    median = weighted_median(forecasts, weights)
                    losses[t0]["mae"][method].append(abs(actual - median))

                    for tau in TAUS:
                        q_hat = weighted_quantile(forecasts, weights, tau)
                        losses[t0]["crps"][method].append(
                            analysis_pinball_loss(actual, q_hat, tau)
                        )

        cases += 1

    if cases == 0:
        raise ValueError(
            f"No {run_type} forecast cases found for "
            f"model_setting={model_setting}, model={column_name}, "
            f"delivery_index={delivery_index}, daily_file={daily_file}, "
            f"case_index={case_index}"
        )

    rows = []
    for t0 in range(30):
        row = {
            "t0": t0,
            "n_future_steps": 30 - t0,
            "mae_raw": np.mean(losses[t0]["mae"]["raw"]),
            "mae_mae": np.mean(losses[t0]["mae"]["mae"]),
            "mae_kernel": np.mean(losses[t0]["mae"]["kernel"]),
            "crps_raw": np.mean(losses[t0]["crps"]["raw"]),
            "crps_mae": np.mean(losses[t0]["crps"]["mae"]),
            "crps_kernel": np.mean(losses[t0]["crps"]["kernel"]),
        }
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
        "--case_index",
        default=None,
        type=int,
        help="Optional zero-based index in the ordered forecast-case stream.",
    )
    parser.add_argument(
        "--max_cases",
        default=None,
        type=int,
        help="Optional cap on selected forecast cases, useful for smoke tests.",
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
        delivery_index=args.delivery_index,
        daily_file=args.daily_file,
        case_index=args.case_index,
        max_cases=args.max_cases,
    )

    identity = (
        f"{args.run_type}_{_safe_name(args.model_setting)}_{_safe_name(column_name)}_"
        f"{args.one_sided}_{args.strategy_model}_{_safe_name(args.band_type)}"
    )
    if args.case_index is not None:
        identity += f"_case_{args.case_index}"
    if args.delivery_index is not None:
        identity += f"_delivery_{args.delivery_index}"
    if args.daily_file is not None:
        identity += f"_{_safe_name(args.daily_file)}"
    if args.max_cases is not None:
        identity += f"_n_{args.max_cases}"
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
