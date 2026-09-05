"""Pairwise multivariate Diebold-Mariano tests for MAE and CRPS."""

import argparse
import hashlib
from dataclasses import dataclass
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

from Forecasting_results_analysis.diebold_mariano_plotting import (
    paper_label,
    plot_dm,
)
from config.forecasting_simulation_config import (
    deliveries_no,
    forecasting_horizon,
    last_trade_time_in_path_delta,
)
from config.paths import (
    BENCHMARK_RESULTS_DIR,
    DM_MAE_CRPS_RESULTS_DIR,
    MODEL_RESULTS_DIR,
    PAPER_FIGURES_DIR,
    RAW_BENCHMARK_RESULTS_DIR,
    RAW_DM_MAE_CRPS_RESULTS_DIR,
    RAW_MODEL_RESULTS_DIR,
)
from config.test_calibration_validation import (
    validation_window_end,
    validation_window_length,
    validation_window_start,
)


TAUS = np.linspace(0.01, 0.99, 99)
N_DAYS = validation_window_length
N_DELIVERIES = deliveries_no
DM_RESULTS_DIR = Path(DM_MAE_CRPS_RESULTS_DIR)
LOSS_CACHE_DIR = DM_RESULTS_DIR / "LOSSES"


@dataclass(frozen=True)
class DMDirectories:
    """Input and numerical-output directories for one DM run."""

    model_results: Path
    benchmark_results: Path
    dm_results: Path

    @property
    def loss_cache(self):
        """Return the cache directory associated with this run."""
        return self.dm_results / "LOSSES"


@dataclass(frozen=True)
class ForecastConfig:
    approach: str
    column: str
    required_scenarios: object
    wasserstein: object = False
    sampling: object = None

    @property
    def key(self):
        """Return the unique, file-safe configuration identifier."""
        model = "benchmark" if self.approach == "benchmark" else self.column
        return (
            f"{self.approach}_{model}_{self.wasserstein}_"
            f"{self.sampling}_{self.required_scenarios}"
        )

    @property
    def label(self):
        """Return the paper-style configuration label used in plots."""
        return paper_label(self)


def forecast_configs():
    """Return the 27 configurations used in the MAE/CRPS study tables."""
    configs = [
        ForecastConfig("benchmark", "benchmark_prediction", window)
        for window in (28, 182, "None")
    ]
    for approach in ("hist_insample", "weather_scenarios"):
        for column in ("MULTI_prediction", "CHAIN_prediction"):
            for wasserstein, sampling in ((True, "dual_coeff"), (False, None)):
                configs.extend(
                    ForecastConfig(
                        approach,
                        column,
                        window,
                        wasserstein,
                        sampling,
                    )
                    for window in (28, 182, "None")
                )
    return configs


def _resolve_directories(special_results_directory=None):
    """Resolve all non-figure directories from an optional alternate root."""
    if special_results_directory is None:
        return DMDirectories(
            model_results=Path(MODEL_RESULTS_DIR),
            benchmark_results=Path(BENCHMARK_RESULTS_DIR),
            dm_results=DM_RESULTS_DIR,
        )

    root = Path(special_results_directory)
    return DMDirectories(
        model_results=root / RAW_MODEL_RESULTS_DIR,
        benchmark_results=root / RAW_BENCHMARK_RESULTS_DIR,
        dm_results=root / RAW_DM_MAE_CRPS_RESULTS_DIR,
    )


def _result_directory(config, delivery, directories=None):
    """Return the raw-results directory for one configuration and delivery."""
    directories = directories or _resolve_directories()
    trade_time = delivery * 3 + last_trade_time_in_path_delta
    common = (
        f"{validation_window_start}_{validation_window_end}_364_{delivery}_"
        f"{forecasting_horizon}_{trade_time}"
    )
    if config.approach == "benchmark":
        name = f"{common}_____{config.required_scenarios}____"
        return directories.benchmark_results / name
    name = (
        f"{common}_{config.approach}_{config.required_scenarios}_"
        f"{config.wasserstein}_{config.sampling}"
    )
    return directories.model_results / name


def _daily_losses(path, column):
    """Calculate trajectory-level MAE and CRPS from one daily result file."""
    df = pd.read_csv(path, index_col=0)
    actual = df["actual"].to_numpy(float)
    forecast_columns = [
        name for name in df if name.startswith(column) and "base_path" not in name
    ]
    forecasts = df[forecast_columns].to_numpy(float)
    if (
        actual.shape != (forecasting_horizon,)
        or forecasts.shape[0] != forecasting_horizon
    ):
        raise ValueError(f"Unexpected trajectory shape in {path}")

    median = np.median(forecasts, axis=1)
    quantiles = np.quantile(forecasts, TAUS, axis=1)
    error = actual[None, :] - quantiles
    pinball = np.where(
        error >= 0,
        TAUS[:, None] * error,
        (1 - TAUS[:, None]) * -error,
    )
    return np.mean(np.abs(actual - median)), np.mean(pinball), actual


def _delivery_losses(delivery, config, directories=None):
    """Calculate daily MAE and CRPS series for one delivery and configuration."""
    directory = _result_directory(config, delivery, directories)
    files = sorted(directory.glob("test_*.csv"))
    if len(files) != N_DAYS:
        raise ValueError(
            f"Expected {N_DAYS} test files in {directory}, found {len(files)}"
        )

    mae = np.empty(N_DAYS)
    crps = np.empty(N_DAYS)
    actual_hash = hashlib.sha256()
    for day, path in enumerate(files):
        mae[day], crps[day], actual = _daily_losses(path, config.column)
        actual_hash.update(actual.astype("<f8", copy=False).tobytes())
    if not np.all(np.isfinite([mae, crps])):
        raise ValueError(f"Non-finite loss in {directory}")
    return delivery, mae, crps, actual_hash.hexdigest()


def _config_losses(config, pool, recompute=False, directories=None):
    """Load or calculate day-by-delivery loss matrices for one configuration."""
    directories = directories or _resolve_directories()
    directories.loss_cache.mkdir(parents=True, exist_ok=True)
    cache = directories.loss_cache / f"{config.key}.npz"
    if cache.exists() and not recompute:
        with np.load(cache) as data:
            mae, crps, hashes = data["mae"], data["crps"], data["actual_hashes"]
        if mae.shape != (N_DAYS, N_DELIVERIES) or crps.shape != mae.shape:
            raise ValueError(f"Invalid cache shape in {cache}; rerun with --recompute")
        return mae, crps, hashes

    worker = partial(_delivery_losses, config=config, directories=directories)
    rows = list(
        tqdm(
            pool.imap(worker, range(N_DELIVERIES)),
            total=N_DELIVERIES,
            desc=config.label,
            unit="delivery",
        )
    )
    mae = np.column_stack([row[1] for row in rows])
    crps = np.column_stack([row[2] for row in rows])
    hashes = np.array([row[3] for row in rows])
    np.savez_compressed(cache, mae=mae, crps=crps, actual_hashes=hashes)
    return mae, crps, hashes


def dm_p_value(loss_1, loss_2):
    """One-sided epftoolbox-style DM p-value; small values favor loss_2."""
    if loss_1.shape != loss_2.shape or loss_1.ndim != 2:
        raise ValueError("Loss arrays must have equal shape (days, deliveries)")
    differential = loss_1.mean(axis=1) - loss_2.mean(
        axis=1
    )  # does not matter to the final result if this is normalized L1 or not, here it is
    mean = differential.mean()
    variance = differential.var(ddof=0)
    if variance == 0:
        raise ValueError(
            "Variance of loss measures differences is 0. We do not expect this to happen in this project - please review the inputs."
        )
    statistic = mean / np.sqrt(variance / differential.size)
    return float(1 - stats.norm.cdf(statistic))


def pairwise_dm(losses):
    """Return the directed pairwise p-value matrix for model loss arrays."""
    model_count = losses.shape[0]
    p_values = np.ones((model_count, model_count))
    for row in range(model_count):
        for column in range(model_count):
            if row != column:
                p_values[row, column] = dm_p_value(losses[row], losses[column])
    return p_values


def parse_args():
    """Parse command-line options for paths, workers, and cache replacement."""
    parser = argparse.ArgumentParser(
        description="Run 27-case multivariate DM tests for MAE and CRPS."
    )
    parser.add_argument("--processes", type=int, default=32)
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument(
        "--special_results_directory",
        type=Path,
        help=(
            "Alternate root containing the forecast-result directories and "
            "receiving DM_MAE_CRPS_RESULTS"
        ),
    )
    return parser.parse_args()


def main():
    """Calculate, save, and plot the MAE and CRPS DM comparisons."""
    args = parse_args()
    directories = _resolve_directories(args.special_results_directory)
    configs = forecast_configs()
    if len(configs) != 27:
        raise RuntimeError(f"Expected 27 configurations, found {len(configs)}")

    mae_losses, crps_losses = [], []
    reference_hashes = None
    with Pool(args.processes) as pool:
        for config in configs:
            mae, crps, hashes = _config_losses(
                config,
                pool,
                recompute=args.recompute,
                directories=directories,
            )
            if reference_hashes is not None and not np.array_equal(
                hashes, reference_hashes
            ):
                raise ValueError(f"Actual trajectories differ for {config.key}")
            reference_hashes = hashes
            mae_losses.append(mae)
            crps_losses.append(crps)

    labels = [config.label for config in configs]
    keys = [config.key for config in configs]
    output_dir = Path(PAPER_FIGURES_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    directories.dm_results.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"configuration": keys, "plot_label": labels}).to_csv(
        directories.dm_results / "DM_model_labels.csv", index=False
    )

    for metric, losses in (
        ("MAE", np.stack(mae_losses)),
        ("CRPS", np.stack(crps_losses)),
    ):
        print(f"Calculating {metric} DM matrix", flush=True)
        p_values = pairwise_dm(losses)
        pd.DataFrame(p_values, index=keys, columns=keys).to_csv(
            directories.dm_results / f"DM_{metric}_p_values.csv"
        )
        plot_dm(p_values, configs, metric, output_dir / f"DM_{metric}")


if __name__ == "__main__":
    main()
