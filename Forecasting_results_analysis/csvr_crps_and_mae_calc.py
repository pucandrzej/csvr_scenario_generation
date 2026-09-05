"""Aggregating the forecasting simulation results
to obtain the values of MAE and CRPS measures"""

import argparse
import os
import pandas as pd
import numpy as np
from multiprocessing import Pool

from config.forecasting_simulation_config import limited_scenarios_number
from .forecasting_results_utils import timing

from config.test_calibration_validation import (
    validation_window_length,
    validation_window_start,
    validation_window_end,
)
from config.paths import (
    BENCHMARK_RESULTS_DIR,
    MODEL_RESULTS_DIR,
    MAE_CRPS_RESULTS_DIR,
    RAW_BENCHMARK_RESULTS_DIR,
    RAW_MODEL_RESULTS_DIR,
    RAW_MAE_CRPS_RESULTS_DIR,
)
from config.forecasting_simulation_config import (
    last_trade_time_in_path_delta,
    deliveries_no,
)

probab_approaches = ["weather_scenarios", "hist_insample", "benchmark"]

wasserstein_stopping_crits = [True, False]

senarios_sampling_methods = ["dual_coeff", None]

repeated_required_scenarios_list = [
    val
    for val in limited_scenarios_number
    for _ in range(len(wasserstein_stopping_crits))
]


@timing
def get_pinball_from_csvr(inp):
    model_name = inp[0]
    results_dir = inp[1]
    delivery = inp[2]
    wasserstein_stopping_crit = inp[3]
    scenarios_sampling_method = inp[4]
    required_scenarios = inp[5]
    probab_approach = inp[6]
    output_dir = inp[7]
    trade_time = delivery * 3 + last_trade_time_in_path_delta

    pinball_path = os.path.join(
        output_dir,
        f"CRPS_{probab_approach}_{model_name}_{delivery}_{wasserstein_stopping_crit}_{scenarios_sampling_method}_{required_scenarios}.csv",
    )
    mae_path = os.path.join(
        output_dir,
        f"MAE_{probab_approach}_{model_name}_{delivery}_{wasserstein_stopping_crit}_{scenarios_sampling_method}_{required_scenarios}.csv",
    )

    # Skip configurations that have already been computed
    if os.path.exists(pinball_path) and os.path.exists(mae_path):
        print(
            f"Skipping existing results: {probab_approach}, {model_name}, "
            f"delivery={delivery}, wasserstein={wasserstein_stopping_crit}, "
            f"sampling={scenarios_sampling_method}, scenarios={required_scenarios}"
        )
        return None

    if model_name == "benchmark":
        results_subdir = f"{validation_window_start}_{validation_window_end}_364_{delivery}_31_{trade_time}_____{required_scenarios}____"
    else:
        results_subdir = f"{validation_window_start}_{validation_window_end}_364_{delivery}_31_{trade_time}_{probab_approach}_{required_scenarios}_{wasserstein_stopping_crit}_{scenarios_sampling_method}"

    all_scenarios = []
    actuals = []
    for fil in os.listdir(os.path.join(results_dir, results_subdir)):
        if fil.startswith(
            "test_"
        ):  # we only want to extract validation (test) window results here
            df = pd.read_csv(
                os.path.join(results_dir, results_subdir, fil),
                usecols=lambda c: c == "actual"
                or (c.startswith(model_name) and "base_path" not in c),
            )
            model_cols = [
                c
                for c in df.columns
                if c.startswith(model_name) and "base_path" not in c
            ]
            actuals.append(df["actual"].values)
            all_scenarios.append(df[model_cols].values.T)

    if len(all_scenarios) != validation_window_length:
        raise ValueError(
            f"We require all of the results to cover exactly {validation_window_length} days of test period. Len {all_scenarios} for {results_subdir} detected."
        )

    step = 1
    limited_path = [i for i in range(0, 31, step)]

    # Find the max length
    max_len = max(len(s) for s in all_scenarios)

    # Create a full array of NaNs
    arr = np.full((len(all_scenarios), max_len, len(limited_path)), np.nan)

    # Fill each row with the existing values
    for i, row in enumerate(all_scenarios):
        arr[i, : len(row), :] = np.array(row)[:, limited_path]

    all_scenarios = arr

    actuals = np.array(actuals)
    actuals = actuals[:, limited_path]

    taus = np.linspace(0.01, 0.99, 99)

    quantiles = np.nanquantile(
        all_scenarios, taus, axis=1
    )  # we need to use nan-robust quantile here as we use nan padding
    errors = actuals[None, :, :] - quantiles
    tau_grid = taus[:, None, None]
    losses = np.where(errors >= 0, tau_grid * errors, (1 - tau_grid) * -errors)

    full_path = range(all_scenarios.shape[2])
    path_idxs = np.repeat(full_path, len(taus))
    used_taus = np.tile(taus, len(full_path))
    pinball = losses.mean(axis=1).T.ravel()
    mae = np.abs(np.nanmedian(all_scenarios, axis=1) - actuals).mean(
        axis=0
    )  # we need to use nan-robust median here as we use nan padding

    qra_pinball_score_df = pd.DataFrame()
    qra_pinball_score_df["path_idx"] = path_idxs
    qra_pinball_score_df["tau"] = used_taus
    qra_pinball_score_df["pinball"] = pinball
    qra_pinball_score_df.to_csv(pinball_path)

    qra_mae_df = pd.DataFrame()
    qra_mae_df["path_idx"] = full_path
    qra_mae_df["mae"] = mae
    qra_mae_df.to_csv(mae_path)

    # return the results
    return qra_pinball_score_df, qra_mae_df


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processes", default=32, type=int)
    parser.add_argument("--special_results_directory")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model_results_dir = MODEL_RESULTS_DIR
    benchmark_results_dir = BENCHMARK_RESULTS_DIR
    output_dir = MAE_CRPS_RESULTS_DIR
    if args.special_results_directory:
        model_results_dir = os.path.join(
            args.special_results_directory, RAW_MODEL_RESULTS_DIR
        )
        benchmark_results_dir = os.path.join(
            args.special_results_directory, RAW_BENCHMARK_RESULTS_DIR
        )
        output_dir = os.path.join(
            args.special_results_directory, RAW_MAE_CRPS_RESULTS_DIR
        )
    os.makedirs(output_dir, exist_ok=True)

    model_names = ["CHAIN_prediction", "MULTI_prediction"]
    deliveries = range(deliveries_no)

    inputlist = []

    for probab_approach in probab_approaches:
        for delivery in deliveries:
            if probab_approach != "benchmark":
                results_dir = model_results_dir
                for model_name in model_names:
                    for (
                        wasserstein_stopping_crit,
                        scenarios_sampling_method,
                        required_scenarios,
                    ) in zip(
                        wasserstein_stopping_crits * len(limited_scenarios_number),
                        senarios_sampling_methods * len(limited_scenarios_number),
                        repeated_required_scenarios_list,
                    ):
                        inputlist.append(
                            [
                                model_name,
                                results_dir,
                                delivery,
                                wasserstein_stopping_crit,
                                scenarios_sampling_method,
                                required_scenarios,
                                probab_approach,
                                output_dir,
                            ]
                        )

            else:
                result_dir = benchmark_results_dir
                for required_scenarios in limited_scenarios_number:
                    inputlist.append(
                        [
                            "benchmark",
                            result_dir,
                            delivery,
                            False,
                            None,
                            required_scenarios,
                            probab_approach,
                            output_dir,
                        ]
                    )

    print(f"Running {len(inputlist)} tasks")

    with Pool(processes=args.processes) as p:
        _ = p.map(get_pinball_from_csvr, inputlist)
