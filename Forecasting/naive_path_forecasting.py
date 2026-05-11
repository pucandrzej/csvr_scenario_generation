"""
File contains the code for GAMLSS forecast of continuous market prices on DE intraday market.
"""

import pandas as pd
import numpy as np
import os

from datetime import timedelta
import argparse

import time
from multiprocessing import Pool, RawArray

import sqlite3

from config.paths import MARKET_DATA_DIR, TIMING_RESULTS_DIR, BENCHMARK_RESULTS_DIR
from config.test_calibration_validation import (
    validation_window_start,
    validation_window_end,
)
from config.forecasting_simulation_config import (
    last_trade_time_in_path_delta,
    calib_window_days_no,
    forecasting_horizon,
    first_trading_start_of_simulation,
    needed_columns_of_continuous_preprocessed_data,
    first_day_index_of_simulation,
    total_no_of_cont_market_columns,
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--daterange_start", default=validation_window_start, help="Start of eval data"
)
parser.add_argument(
    "--daterange_end", default=validation_window_end, help="End of eval data"
)
parser.add_argument("--lookback", default=364, help="Min. training window length")
parser.add_argument(
    "--delivery_time",
    default=32,
    help="Index from 0 to 95 of the delivery quarter of the day",
)
parser.add_argument(
    "--trade_time",
    default=32 * 3 + last_trade_time_in_path_delta,
    help="Last 5min interval before the delivery that we want to trade in.",
)
parser.add_argument(
    "--calibration_window_len",
    default=calib_window_days_no,
    help="For every date consider a historical results from a calibration window.",
)
parser.add_argument("--processes", default=1, help="No of processes")
parser.add_argument(
    "--required_scenarios",
    default=None,
    help='Number of the most recent scenarios to be used for SVR path scenarios recalculation if scenarios_sampling_method is "latest". Cannot exceed the sample length.',
)

PATHS_NO = 1000  # number of naive paths scenarios to simulate

args = parser.parse_args()
start = args.daterange_start
end = args.daterange_end
lookback = int(args.lookback)
trade_time = int(args.trade_time)
delivery_time = int(args.delivery_time)
calibration_window_len = int(args.calibration_window_len)
if args.required_scenarios is not None:
    required_scenarios = int(args.required_scenarios)
else:
    required_scenarios = args.required_scenarios

os.makedirs(
    os.path.join(
        BENCHMARK_RESULTS_DIR,
        f"{start}_{end}_{lookback}_{delivery_time}_{forecasting_horizon}_{trade_time}_____{args.required_scenarios}____",
    ),
    exist_ok=True,
)

dates = pd.date_range(start, end)
dates_calibration = pd.date_range(
    pd.to_datetime(start) - timedelta(days=calibration_window_len),
    pd.to_datetime(start) - timedelta(days=1),
)

# a global dictionary storing the variables passed from the initializer.
var_dict = {}


def init_worker(Model_data, Model_data_shape):
    # Using a dictionary is not strictly necessary. You can also
    # use global variables.
    var_dict["Model_data"] = Model_data
    var_dict["Model_data_shape"] = Model_data_shape


def run_one_day(inp):
    idx = inp[0]
    date_fore = inp[1]
    forecasting_horizon = inp[2]
    calibration_flag = inp[3]

    """
    GATHERING THE REQUIRED DATA
    """
    # to compare ourselves with results from previous papers, we want to 185-30 min before delivery periods, 31 5min intervals

    daily_data_window = np.swapaxes(
        np.frombuffer(var_dict["Model_data"]).reshape(var_dict["Model_data_shape"]),
        1,
        2,
    )[:idx, :, :]  # swapaxes needed after migration from np array to database

    Y = (
        daily_data_window[:, delivery_time, -forecasting_horizon:]
        - daily_data_window[:, delivery_time, -forecasting_horizon - 1 : -1]
    )
    trajectory_start_point = daily_data_window[
        -1, delivery_time, -forecasting_horizon - 1
    ]
    """
    SAMPLE TRAJECTORIES AND SAVE THEM
    """
    Y_historical = Y[:-1, :]  # select only the historical trajectories to sample from
    if required_scenarios is not None:
        Y_historical = Y_historical[-required_scenarios:, :]
    sampled_indices = np.random.choice(
        Y_historical.shape[0], size=PATHS_NO, replace=True
    )
    sampled_Y = Y_historical[sampled_indices, :]
    sampled_cumsum = np.cumsum(sampled_Y, axis=1)
    simulation_results = (sampled_cumsum + trajectory_start_point).T

    results = pd.DataFrame(
        simulation_results,
        columns=[
            f"benchmark_prediction_{i}" for i in range(simulation_results.shape[1])
        ],
    )

    result_file_name = os.path.join(
        BENCHMARK_RESULTS_DIR,
        f"{start}_{end}_{lookback}_{delivery_time}_{forecasting_horizon}_{trade_time}_____{args.required_scenarios}____",
        f"{calibration_flag}_{str((pd.to_datetime(date_fore) - timedelta(days=1)).replace(hour=16) + timedelta(minutes=5 * (trade_time - 1))).replace(':', ';')}_{forecasting_horizon}___weights___window__.csv",
    )

    results["actual"] = daily_data_window[-1, delivery_time, -forecasting_horizon:]
    results["naive"] = [trajectory_start_point] * forecasting_horizon

    try:
        results.to_csv(result_file_name)
    except Exception as err:
        os.remove(result_file_name)
        raise KeyboardInterrupt(
            f"Interrupted on saving: last file removed to avoid empty files. Exception: {err}"
        )


if __name__ == "__main__":
    con = sqlite3.connect(MARKET_DATA_DIR)
    sql_str = f"SELECT * FROM with_dummies WHERE Index_daily <= {trade_time} AND Time >= '{first_trading_start_of_simulation}' AND Day >= {first_day_index_of_simulation};"  # load only the data required for simu, so up to last trade time in the trajectory
    daily_data = pd.read_sql(sql_str, con)[
        needed_columns_of_continuous_preprocessed_data
    ].to_numpy()
    daily_data = np.reshape(
        daily_data,
        (
            np.shape(daily_data)[0] // trade_time,
            trade_time,
            total_no_of_cont_market_columns,
        ),
    )  # making it a shape of [days, total steps in trajectory, variables no.]

    raw_arr = RawArray(
        "d", np.shape(daily_data)[0] * np.shape(daily_data)[1] * np.shape(daily_data)[2]
    )

    # Wrap X as an numpy array so we can easily manipulates its data in multiprocessing scheme
    daily_data_np = np.frombuffer(raw_arr).reshape(np.shape(daily_data))
    # Copy data to our shared array.
    np.copyto(daily_data_np, daily_data)
    data_shape = np.shape(daily_data)

    # free memory
    daily_data = None
    del daily_data

    simu_start = time.time()
    if calibration_window_len > 0:  # perform the calibration forecast
        inputlist_calibration = [
            [
                lookback - calibration_window_len + idx + 1,
                date,
                forecasting_horizon,
                "calibration",
            ]
            for idx, date in enumerate(dates_calibration)
        ]
        with Pool(
            processes=int(args.processes),
            initializer=init_worker,
            initargs=(raw_arr, data_shape),
        ) as p:
            _ = p.map(run_one_day, inputlist_calibration)

        inputlist = [
            [lookback + idx + 1, date, forecasting_horizon, "test"]
            for idx, date in enumerate(dates)
        ]
        with Pool(
            processes=int(args.processes),
            initializer=init_worker,
            initargs=(raw_arr, data_shape),
        ) as p:
            _ = p.map(run_one_day, inputlist)

    else:
        inputlist = [
            [lookback + idx + 1, date, forecasting_horizon, "test"]
            for idx, date in enumerate(dates)
        ]
        with Pool(
            processes=int(args.processes),
            initializer=init_worker,
            initargs=(raw_arr, data_shape),
        ) as p:
            _ = p.map(run_one_day, inputlist)
    simu_end = time.time()
    print(simu_end - simu_start, "Total time of simulation:")
    with open(
        os.path.join(
            TIMING_RESULTS_DIR,
            f"timing_results_benchmark_d_{delivery_time}_t_{trade_time}.txt",
        ),
        "w",
    ) as file:
        file.write(f"Execution time: {simu_end - simu_start} seconds\n")
