import os
import pandas as pd
import numpy as np
from multiprocessing import Pool

from config.forecasting_simulation_config import limited_scenarios_number
from .forecasting_results_utils import my_mae, timing, analysis_pinball_loss

from config.test_calibration_validation import (
    validation_window_length,
    validation_window_start,
    validation_window_end,
)
from config.paths import BENCHMARK_RESULTS_DIR, MODEL_RESULTS_DIR, MAE_CRPS_RESULTS_DIR
from config.forecasting_simulation_config import last_trade_time_in_path_delta

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
    trade_time = delivery * 3 + last_trade_time_in_path_delta

    pinball_path = os.path.join(
        MAE_CRPS_RESULTS_DIR,
        f"CRPS_{probab_approach}_{model_name}_{delivery}_{wasserstein_stopping_crit}_{scenarios_sampling_method}_{required_scenarios}.csv",
    )
    mae_path = os.path.join(
        MAE_CRPS_RESULTS_DIR,
        f"MAE_{probab_approach}_{model_name}_{delivery}_{wasserstein_stopping_crit}_{scenarios_sampling_method}_{required_scenarios}.csv",
    )

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
                os.path.join(results_dir, results_subdir, fil), index_col=0
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

    all_forecasts = np.ones(
        (len(taus), np.shape(all_scenarios)[2], validation_window_length)
    )
    all_actuals = np.ones((np.shape(all_scenarios)[2], validation_window_length))

    pinball_sum = 0
    pinball = []
    path_idxs = []
    used_taus = []
    mae = []

    full_path = range(np.shape(all_scenarios)[2])

    for path_component in full_path:
        y = actuals[:, path_component]
        X = all_scenarios[:, :, path_component].T

        for tau_idx, tau in enumerate(taus):
            forecasts = np.nanquantile(X, q=tau, axis=0)

            all_forecasts[tau_idx, path_component, :] = forecasts

            pinball.append(analysis_pinball_loss(y, forecasts, tau))
            pinball_sum += analysis_pinball_loss(y, forecasts, tau)
            path_idxs.append(path_component)
            used_taus.append(tau)

        mae.append(my_mae(np.nanmedian(X, axis=0), y))

        all_actuals[path_component, :] = y

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


if __name__ == "__main__":
    model_names = ["CHAIN_prediction", "MULTI_prediction"]
    deliveries = range(96)

    inputlist = []

    for probab_approach in probab_approaches:
        for delivery in deliveries:
            if probab_approach != "benchmark":
                results_dir = MODEL_RESULTS_DIR
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
                            ]
                        )

            else:
                result_dir = BENCHMARK_RESULTS_DIR
                for required_scenarios in repeated_required_scenarios_list:
                    inputlist.append(
                        [
                            "benchmark",
                            result_dir,
                            delivery,
                            False,
                            None,
                            required_scenarios,
                            probab_approach,
                        ]
                    )

    print(f"Running {len(inputlist)} tasks")

    with Pool(processes=32) as p:
        _ = p.map(get_pinball_from_csvr, inputlist)
