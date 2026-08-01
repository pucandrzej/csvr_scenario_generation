"""Pairwise multivariate Diebold-Mariano tests for MAE and CRPS."""

import argparse
import hashlib
from dataclasses import dataclass
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

from config.forecasting_simulation_config import (
    deliveries_no,
    forecasting_horizon,
    last_trade_time_in_path_delta,
)
from config.paths import (
    BENCHMARK_RESULTS_DIR,
    MAE_CRPS_RESULTS_DIR,
    MODEL_RESULTS_DIR,
    PAPER_FIGURES_DIR,
    ROOT,
)
from config.test_calibration_validation import (
    validation_window_end,
    validation_window_length,
    validation_window_start,
)


TAUS = np.linspace(0.01, 0.99, 99)
N_DAYS = validation_window_length
N_DELIVERIES = deliveries_no
LOSS_CACHE_DIR = Path(MAE_CRPS_RESULTS_DIR) / "DM_LOSSES"


@dataclass(frozen=True)
class ForecastConfig:
    approach: str
    column: str
    required_scenarios: object
    wasserstein: object = False
    sampling: object = None

    @property
    def key(self):
        model = "benchmark" if self.approach == "benchmark" else self.column
        return (
            f"{self.approach}_{model}_{self.wasserstein}_"
            f"{self.sampling}_{self.required_scenarios}"
        )

    @property
    def label(self):
        window = "all" if self.required_scenarios == "None" else self.required_scenarios
        if self.approach == "benchmark":
            return f"BENCH/{window}"
        source = "HIST" if self.approach == "hist_insample" else "WEATHER"
        model = self.column.removesuffix("_prediction")
        method = "dual" if self.wasserstein else "plain"
        return f"{source}/{model}/{method}/{window}"


def forecast_configs():
    """Return the 27 configurations used in the MAE/CRPS study tables."""
    configs = [
        ForecastConfig("benchmark", "benchmark_prediction", window)
        for window in ("None", 28, 182)
    ]
    for approach in ("hist_insample", "weather_scenarios"):
        for column in ("MULTI_prediction", "CHAIN_prediction"):
            for window in ("None", 28, 182):
                configs.extend(
                    [
                        ForecastConfig(approach, column, window, True, "dual_coeff"),
                        ForecastConfig(approach, column, window),
                    ]
                )
    return configs


def _result_directory(config, delivery):
    trade_time = delivery * 3 + last_trade_time_in_path_delta
    common = (
        f"{validation_window_start}_{validation_window_end}_364_{delivery}_"
        f"{forecasting_horizon}_{trade_time}"
    )
    if config.approach == "benchmark":
        name = f"{common}_____{config.required_scenarios}____"
        return Path(BENCHMARK_RESULTS_DIR) / name
    name = (
        f"{common}_{config.approach}_{config.required_scenarios}_"
        f"{config.wasserstein}_{config.sampling}"
    )
    return Path(MODEL_RESULTS_DIR) / name


def _daily_losses(path, column):
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

    median = np.nanmedian(forecasts, axis=1)
    quantiles = np.nanquantile(forecasts, TAUS, axis=1)
    error = actual[None, :] - quantiles
    pinball = np.where(
        error >= 0,
        TAUS[:, None] * error,
        (1 - TAUS[:, None]) * -error,
    )
    return np.mean(np.abs(actual - median)), np.mean(pinball), actual


def _delivery_losses(delivery, config):
    directory = _result_directory(config, delivery)
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


def _config_losses(config, pool, recompute=False):
    LOSS_CACHE_DIR.mkdir(exist_ok=True)
    cache = LOSS_CACHE_DIR / f"{config.key}.npz"
    if cache.exists() and not recompute:
        with np.load(cache) as data:
            mae, crps, hashes = data["mae"], data["crps"], data["actual_hashes"]
        if mae.shape != (N_DAYS, N_DELIVERIES) or crps.shape != mae.shape:
            raise ValueError(f"Invalid cache shape in {cache}; rerun with --recompute")
        return mae, crps, hashes

    worker = partial(_delivery_losses, config=config)
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
    differential = loss_1.mean(axis=1) - loss_2.mean(axis=1)
    mean = differential.mean()
    variance = differential.var(ddof=0)
    if variance == 0:
        return 1.0 if mean == 0 else float(mean < 0)
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


def _dm_colormap():
    red = np.r_[np.linspace(0, 1, 50), np.linspace(1, 0.5, 50)[1:], 0]
    green = np.r_[np.linspace(0.5, 1, 50), np.zeros(50)]
    cmap = mpl.colors.ListedColormap(np.c_[red, green, np.zeros(100)])
    cmap.set_bad("0.65")
    return cmap


def plot_dm(p_values, labels, metric, output_stem):
    """Write the complete Weron-style p-value heatmap as PNG and PDF."""
    shown = p_values.copy()
    np.fill_diagonal(shown, np.nan)
    with plt.style.context(Path(ROOT) / "paper_style.mplstyle"):
        mpl.rcParams["text.usetex"] = False
        mpl.rcParams["font.serif"] = ["DejaVu Serif"]
        fig, ax = plt.subplots(figsize=(12, 10))
        image = ax.imshow(shown, cmap=_dm_colormap(), vmin=0, vmax=0.1)
        ticks = np.arange(len(labels))
        ax.set_xticks(ticks, labels, rotation=90)
        ax.set_yticks(ticks, labels)
        ax.plot(ticks, ticks, "x", color="0.2", markersize=5, markeredgewidth=0.7)
        ax.set(xlabel="Model 1", ylabel="Model 2", title=f"DM test - {metric}")
        ax.tick_params(axis="both", length=0, labelsize=5.5)
        colorbar = fig.colorbar(image, ax=ax, fraction=0.04, pad=0.03)
        colorbar.set_label("p-value")
        colorbar.set_ticks([0, 0.025, 0.05, 0.075, 0.1])
        fig.tight_layout()
        fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
        fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run 27-case multivariate DM tests for MAE and CRPS."
    )
    parser.add_argument("--processes", type=int, default=32)
    parser.add_argument("--recompute", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    configs = forecast_configs()
    if len(configs) != 27:
        raise RuntimeError(f"Expected 27 configurations, found {len(configs)}")

    mae_losses, crps_losses = [], []
    reference_hashes = None
    with Pool(args.processes) as pool:
        for config in configs:
            mae, crps, hashes = _config_losses(config, pool, args.recompute)
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
    output_dir.mkdir(exist_ok=True)
    pd.DataFrame({"configuration": keys, "plot_label": labels}).to_csv(
        Path(MAE_CRPS_RESULTS_DIR) / "DM_model_labels.csv", index=False
    )

    for metric, losses in (
        ("MAE", np.stack(mae_losses)),
        ("CRPS", np.stack(crps_losses)),
    ):
        print(f"Calculating {metric} DM matrix", flush=True)
        p_values = pairwise_dm(losses)
        pd.DataFrame(p_values, index=keys, columns=keys).to_csv(
            Path(MAE_CRPS_RESULTS_DIR) / f"DM_{metric}_p_values.csv"
        )
        plot_dm(p_values, labels, metric, output_dir / f"DM_{metric}")


if __name__ == "__main__":
    main()
