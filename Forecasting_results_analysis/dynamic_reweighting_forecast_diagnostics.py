"""Fast MAE/CRPS evaluation for dynamic scenario weighting."""

import os
from multiprocessing import Pool

import numpy as np
import pandas as pd
from tqdm import tqdm

from config.forecasting_simulation_config import deliveries_no
from config.paths import BENCHMARK_RESULTS_DIR, MODEL_RESULTS_DIR
from config.test_calibration_validation import (
    calibration_window_length,
    validation_window_length,
)


TAUS = np.linspace(0.01, 0.99, 99)
ALL_QUANTILES = np.union1d(TAUS, [0.5])
MEDIAN_INDEX = int(np.searchsorted(ALL_QUANTILES, 0.5))
TAU_INDICES = np.searchsorted(ALL_QUANTILES, TAUS)
EXPECTED_DAYS = {
    "calibration": calibration_window_length,
    "test": validation_window_length,
}


def _read_forecast(path, column_name):
    """Read actuals and the variable number of matching scenario columns."""
    def wanted(name):
        """Return whether a CSV column is required for evaluation."""
        return name == "actual" or (
            name.startswith(column_name) and "base_path" not in name
        )

    frame = pd.read_csv(path, usecols=wanted)
    scenarios = [name for name in frame if wanted(name) and name != "actual"]
    if "actual" not in frame or not scenarios:
        raise ValueError(f"Missing actual or {column_name!r} scenarios in {path}")
    return frame["actual"].to_numpy(), frame[scenarios].to_numpy()


def _delivery_tasks(
    model_setting,
    column_name,
    parameters,
    run_type,
    delivery_index=None,
    daily_file=None,
):
    """Build one worker task per delivery directory.

    Optional delivery and filename filters are used only for focused checks;
    normal calibration and validation cover the complete dataset.
    """
    results_dir = (
        BENCHMARK_RESULTS_DIR
        if column_name == "benchmark_prediction"
        else MODEL_RESULTS_DIR
    )
    directories = sorted(
        (name for name in os.listdir(results_dir) if name.endswith(model_setting)),
        key=lambda name: int(name.split("_")[3]),
    )

    tasks = []
    for directory in directories:
        if delivery_index is not None and int(directory.split("_")[3]) != delivery_index:
            continue
        path = os.path.join(results_dir, directory)
        files = sorted(
            name
            for name in os.listdir(path)
            if name.startswith(f"{run_type}_")
            and (daily_file is None or name == daily_file)
        )
        if files:
            tasks.append(
                (results_dir, directory, files, column_name, tuple(parameters))
            )
    return tasks


def _validate_tasks(tasks, run_type, delivery_index, daily_file):
    """Check that an unfiltered run covers every delivery and expected day."""
    if not tasks:
        raise ValueError(f"No {run_type} forecast files found")
    if delivery_index is not None or daily_file is not None:
        return

    indices = sorted(int(task[1].split("_")[3]) for task in tasks)
    if indices != list(range(deliveries_no)):
        raise ValueError(f"Expected delivery indices 0..{deliveries_no - 1}, got {indices}")

    expected_days = EXPECTED_DAYS[run_type]
    invalid = [(task[1], len(task[2])) for task in tasks if len(task[2]) != expected_days]
    if invalid:
        raise ValueError(f"Expected {expected_days} files per delivery, got {invalid}")


def _pinball_sum(actual, quantiles):
    """Return the summed pinball loss across horizons and quantile levels."""
    errors = np.asarray(actual)[..., None] - quantiles
    return np.where(
        errors >= 0,
        TAUS * errors,
        (1 - TAUS) * (-errors),
    ).sum()


def _weighted_quantiles(sorted_values, sort_indices, weights):
    """Quantiles for several horizons using their previously sorted scenarios."""
    rows = []
    for values, indices in zip(sorted_values, sort_indices):
        cumulative = np.cumsum(weights[indices])
        right = np.minimum(
            np.searchsorted(cumulative, ALL_QUANTILES),
            len(values) - 1,
        )
        boundary = (cumulative[right] == ALL_QUANTILES) | (right == 0)
        left = np.maximum(right - 1, 0)

        quantiles = values[right].copy()
        interpolate = ~boundary
        quantiles[interpolate] = (
            values[left[interpolate]]
            * (cumulative[right[interpolate]] - ALL_QUANTILES[interpolate])
            + values[right[interpolate]]
            * (ALL_QUANTILES[interpolate] - cumulative[left[interpolate]])
        ) / (
            cumulative[right[interpolate]] - cumulative[left[interpolate]]
        )
        rows.append(quantiles)
    return np.vstack(rows)


def _parameter_weights(actual, forecasts, t0, parameters):
    """Compute all weight vectors while sharing work between equal lambdas."""
    actual = actual[: t0 + 1]
    forecasts = forecasts[: t0 + 1]
    differences = forecasts - actual[:, None]
    width = max(np.mean(np.abs(np.median(forecasts, axis=1) - actual)), 0.01)
    kernel_errors = {}
    result = []

    for _, p, lambda_ in parameters:
        if lambda_ not in kernel_errors:
            ages = t0 - np.arange(t0 + 1)
            time_weights = np.exp(-lambda_ * ages)
            time_weights /= time_weights.sum()
            kernel_errors[lambda_] = time_weights @ (differences ** 2)
        unscaled = np.exp(-width * kernel_errors[lambda_] ** (p / 2))

        total = unscaled.sum()
        result.append(
            unscaled / total
            if total and np.isfinite(total)
            else np.ones(forecasts.shape[1]) / forecasts.shape[1]
        )
    return result


def _evaluate_delivery(task):
    """Evaluate every parameter setting for all daily files of one delivery."""
    results_dir, directory, files, column_name, parameters = task
    raw_sums = np.zeros((2, 30))
    weighted_sums = np.zeros((len(parameters), 2, 30))
    evaluated_cases = []

    for filename in files:
        actual, forecasts = _read_forecast(
            os.path.join(results_dir, directory, filename),
            column_name,
        )
        if actual.shape[0] != 31 or forecasts.shape[0] != 31:
            raise ValueError(f"Expected 31 trajectory rows in {directory}/{filename}")

        order = np.argsort(forecasts, axis=1)
        sorted_forecasts = np.take_along_axis(forecasts, order, axis=1)
        raw_medians = np.nanmedian(forecasts, axis=1)
        raw_quantiles = np.nanquantile(forecasts, TAUS, axis=1).T

        for t0 in range(30):
            future_actual = actual[t0 + 1 :]
            raw_sums[0, t0] += np.abs(
                future_actual - raw_medians[t0 + 1 :]
            ).sum()
            raw_sums[1, t0] += _pinball_sum(
                future_actual,
                raw_quantiles[t0 + 1 :],
            )

            weights = _parameter_weights(actual, forecasts, t0, parameters)
            for index, parameter_weights in enumerate(weights):
                quantiles = _weighted_quantiles(
                    sorted_forecasts[t0 + 1 :],
                    order[t0 + 1 :],
                    parameter_weights,
                )
                weighted_sums[index, 0, t0] += np.abs(
                    future_actual - quantiles[:, MEDIAN_INDEX]
                ).sum()
                weighted_sums[index, 1, t0] += _pinball_sum(
                    future_actual,
                    quantiles[:, TAU_INDICES],
                )

        evaluated_cases.append((directory, filename))

    return raw_sums, weighted_sums, evaluated_cases


def evaluate_parameter_grid(
    model_setting,
    column_name,
    parameters,
    run_type,
    processes=1,
    delivery_index=None,
    daily_file=None,
    return_per_t0=False,
):
    """Evaluate a weighting grid in one pass over calibration or test CSVs.

    Parameters are ``(method, p, lambda)`` tuples. The returned DataFrame has
    one row per tuple, aggregated over the 30 information times. When
    ``return_per_t0`` is true, a second DataFrame containing one row per tuple
    and information time is returned alongside it.
    """
    tasks = _delivery_tasks(
        model_setting,
        column_name,
        parameters,
        run_type,
        delivery_index,
        daily_file,
    )
    _validate_tasks(tasks, run_type, delivery_index, daily_file)

    if processes == 1:
        results = map(_evaluate_delivery, tasks)
        results = list(tqdm(results, total=len(tasks), desc="Deliveries"))
    else:
        with Pool(processes) as pool:
            results = list(
                tqdm(
                    pool.imap(_evaluate_delivery, tasks),
                    total=len(tasks),
                    desc="Deliveries",
                )
            )

    raw_sums = sum(result[0] for result in results)
    weighted_sums = sum(result[1] for result in results)
    cases = [case for result in results for case in result[2]]
    counts = len(cases) * (30 - np.arange(30))
    raw_mae = raw_sums[0] / counts
    raw_crps = raw_sums[1] / (counts * len(TAUS))

    rows = []
    per_t0_rows = []
    for index, (method, p, lambda_) in enumerate(parameters):
        weighted_mae = weighted_sums[index, 0] / counts
        weighted_crps = weighted_sums[index, 1] / (counts * len(TAUS))
        rows.append(
            {
                "run_type": run_type,
                "param2": p,
                "param3": lambda_,
                "weights": method,
                "model_setting": model_setting,
                "model": column_name,
                "mae_raw": raw_mae.mean(),
                "mae_weighted": weighted_mae.mean(),
                "crps_raw": raw_crps.mean(),
                "crps_weighted": weighted_crps.mean(),
                "mae_improvement_pct": np.mean(100 * (raw_mae - weighted_mae) / raw_mae),
                "crps_improvement_pct": np.mean(100 * (raw_crps - weighted_crps) / raw_crps),
            }
        )
        if return_per_t0:
            for t0 in range(30):
                per_t0_rows.append(
                    {
                        "run_type": run_type,
                        "t0": t0,
                        "n_future_steps": 30 - t0,
                        "param2": p,
                        "param3": lambda_,
                        "weights": method,
                        "model_setting": model_setting,
                        "model": column_name,
                        "mae_raw": raw_mae[t0],
                        "mae_weighted": weighted_mae[t0],
                        "crps_raw": raw_crps[t0],
                        "crps_weighted": weighted_crps[t0],
                        "mae_improvement_pct": 100
                        * (raw_mae[t0] - weighted_mae[t0])
                        / raw_mae[t0],
                        "crps_improvement_pct": 100
                        * (raw_crps[t0] - weighted_crps[t0])
                        / raw_crps[t0],
                    }
                )

    aggregated = pd.DataFrame(rows)
    if return_per_t0:
        return aggregated, pd.DataFrame(per_t0_rows)
    return aggregated
