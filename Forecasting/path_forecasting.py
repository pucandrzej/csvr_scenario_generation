"""File contains the code for parallel (over days of evaluation window) cSVR path forecast of continuous market prices on DE intraday market."""

import pandas as pd
import numpy as np
import os

import argparse

import time
from multiprocessing import Pool, RawArray
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

import sqlite3
from sklearn.multioutput import MultiOutputRegressor
from sklearn.multioutput import RegressorChain

from config.paths import (
    DATA_DIR,
    MODEL_RESULTS_DIR,
    RAW_MODEL_RESULTS_DIR,
    TIMING_RESULTS_DIR,
)
from .forecasting_utils import (
    my_mae,
    filter_scenarios,
    load_exogenous_to_cache,
    add_exogenous_from_cache_to_variables,
    add_last_known_exogenous_from_cache,
    multifore_corrected_laplace_kernel,
    corrected_laplace_kernel,
    build_weather_scenarios_and_similarity,
    check_wasserstein_stopping,
)
from config.forecasting_simulation_config import (
    last_trade_time_in_path_delta,
    calib_window_days_no,
    forecasting_horizon,
    first_trading_start_of_simulation,
    needed_columns_of_continuous_preprocessed_data,
    total_no_of_cont_market_columns,
)
from config.test_calibration_validation import (
    validation_window_start,
    validation_window_end,
    required_start_pd_timestamp,
)
from config.csvr_model_config import (
    q_kernel,
    q_kernel_naive,
    q_data,
    q_data_naive,
    svr_epsilon,
    C,
)
from config.model_scenario_generation_config import (
    quantile_diff_tolerance,
    min_scenarios,
    wasserstein_moving_avg_window,
    weather_scenarios_split_direction,
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--daterange_start",
    default=validation_window_start,
    help="Start of evaluation window.",
)
parser.add_argument(
    "--daterange_end", default=validation_window_end, help="End of evaluation window."
)
parser.add_argument("--lookback", default=364, help="Training window length.")
parser.add_argument(
    "--delivery_time",
    default=32,
    help="Int from 0 to 95 of the delivery quarter of the day.",
)
parser.add_argument(
    "--trade_time",
    default=32 * 3 + last_trade_time_in_path_delta,
    help="index of the last 5min interval before the delivery that we want to trade in.",
)
parser.add_argument(
    "--calibration_window_len",
    default=calib_window_days_no,
    help="For every date consider a historical results from a calibration window.",
)
parser.add_argument(
    "--wasserstein_stopping_crit",
    default=False,
    action="store_true",
    help="Stopping criterion for adding SVR scenarios that allows us to stop when adding a new scenario is not helping much.",
)
parser.add_argument(
    "--probab_approach",
    default="weather_scenarios",
    help="Approach to probabilistic forecasting. Choose 'hist_insample' or 'weather_scenarios'",
)
parser.add_argument(
    "--scenarios_sampling_method", default=None, help="Method of sorting the scenarios"
)
parser.add_argument(
    "--required_scenarios",
    default=None,
    help='Number of the most recent scenarios to be used for SVR path scenarios recalculation if scenarios_sampling_method is "latest". Cannot exceed the sample length.',
)
parser.add_argument(
    "--ffs_measure",
    default="MAE",
    help="Measure used for smaller representing sample selection in FFS.",
)
parser.add_argument(
    "--special_results_directory",
    default=None,
    help="Running on WCSS Wroclaw University of Science and Technology supercomputers requires us to save the results in dedicated path.",
)
parser.add_argument("--processes", default=1, help="No of processes")

first_training_date = required_start_pd_timestamp

args = parser.parse_args()
if args.required_scenarios is not None:
    required_scenarios = int(args.required_scenarios)
else:
    required_scenarios = args.required_scenarios

start = args.daterange_start
end = args.daterange_end
lookback = int(args.lookback)
trade_time = int(args.trade_time)
delivery_time = int(args.delivery_time)
calibration_window_len = int(args.calibration_window_len)

specific_results_folname = f"{start}_{end}_{lookback}_{delivery_time}_{forecasting_horizon}_{trade_time}_{args.probab_approach}_{args.required_scenarios}_{args.wasserstein_stopping_crit}_{args.scenarios_sampling_method}"

if args.special_results_directory is not None:
    MODEL_RESULTS_DIR = os.path.join(
        args.special_results_directory, RAW_MODEL_RESULTS_DIR
    )

os.makedirs(
    os.path.join(
        MODEL_RESULTS_DIR,
        specific_results_folname,
    ),
    exist_ok=True,
)

dates = pd.date_range(start, end)
dates_calibration = pd.date_range(
    pd.to_datetime(start) - pd.Timedelta(days=calibration_window_len),
    pd.to_datetime(start) - pd.Timedelta(days=1),
)

# a global dictionary storing the data and data shape passed from the initializer - allows to share one copy of intraday market prices & volumes between parallel workers
var_dict = {}


def init_worker(Model_data, Model_data_shape):
    """Function used to manage the data sharing between parallel forecasters."""
    var_dict["Model_data"] = Model_data
    var_dict["Model_data_shape"] = Model_data_shape


class CorrectedSVRKernelWrapper:
    """Wrapper for cSVR kernel, allowing for applying kernel correction in path forecasting setting"""

    def __init__(self):
        self.X_train = None

    def __call__(self, X, Y):
        # Detect train vs test mode
        if self.X_train is None:
            # Training phase: X == Y
            self.X_train = X
            if (
                args.probab_approach == "weather_scenarios"
            ):  # run ram inefficient but faster version for weather scenarios
                return multifore_corrected_laplace_kernel(
                    X, Y, q_kernel, q_data, q_data_naive, q_kernel_naive, is_train=True
                )
            elif (
                args.probab_approach == "hist_insample"
            ):  # run ram efficient and faster version for hist scenarios
                return corrected_laplace_kernel(
                    X, Y, q_kernel, q_data, q_data_naive, q_kernel_naive, is_train=True
                )

        else:
            # Prediction phase: Y is the training data
            if (
                args.probab_approach == "weather_scenarios"
            ):  # run ram inefficient but faster version for weather scenarios
                return multifore_corrected_laplace_kernel(
                    X, Y, q_kernel, q_data, q_data_naive, q_kernel_naive, is_train=False
                )
            elif (
                args.probab_approach == "hist_insample"
            ):  # run ram efficient and faster version for hist scenarios
                return corrected_laplace_kernel(
                    X, Y, q_kernel, q_data, q_data_naive, q_kernel_naive, is_train=False
                )


def run_one_day(inp):
    """Function used to forecast for one day, one delivery."""
    idx = inp[0]
    date_fore = inp[1]
    forecasting_horizon = inp[2]
    calibration_flag = inp[3]

    """
    GATHERING THE REQUIRED DATA
    """
    # to compare ourselves with Ziel GAMLSS papers results we want to 185-30 min before delivery periods, 31 5min intervals

    information_shift = (
        forecasting_horizon + 1
    )  # the shift from the end of a vector allowing to define the known prices and volumes; we assume that we know the price in real time, i.e. after one period

    daily_data_window = np.swapaxes(
        np.frombuffer(var_dict["Model_data"]).reshape(var_dict["Model_data_shape"]),
        1,
        2,
    )[:idx, :, :]  # swapaxes needed after migration from np array to database

    if np.shape(daily_data_window)[-1] <= forecasting_horizon:
        raise ValueError(
            f"The length of time series passed to the model is {np.shape(daily_data_window)[-1]} while the forecasting horizon is {forecasting_horizon}. Please provide a longer time series."
        )

    last_known_price = daily_data_window[
        -1, delivery_time, -information_shift
    ]  # last known price before forecasting time

    # data available at the time of forecasting
    daily_data_window_known = daily_data_window[
        :, :-1, : -information_shift + 1
    ]  # cutting the -1 from 2d dimension as these are dummies, also limiting to path elements known at forecasting

    total_training_days_available = np.shape(daily_data_window_known)[0]

    # collect the X differenced with every step taken
    X_agg_volume = []
    X_naive = []

    # add the volume and naive price differenced *with all the horizons in path* - showed to give an improvement in initial experiments than taking just last known price
    for i in range(1, forecasting_horizon + 1):
        X_tmp = np.expand_dims(
            np.sum(
                (
                    daily_data_window_known[:, 96:, -1]
                    - daily_data_window_known[:, 96:, -i - 1]
                ),
                1,
            ),
            1,
        )
        X_agg_volume.append(X_tmp)

        X_tmp = np.expand_dims(
            daily_data_window_known[:, delivery_time, -1]
            - daily_data_window_known[:, delivery_time, -i - 1],
            1,
        )
        X_naive.append(X_tmp)
    # add the raw (not differenced) last known naive
    X_naive.append(np.expand_dims(daily_data_window_known[:, delivery_time, -1], 1))

    X_exog = np.concatenate(X_agg_volume, axis=-1)
    X_naive = np.concatenate(X_naive, axis=-1)

    Y_list = []
    for i in range(forecasting_horizon):
        Y_list.append(
            daily_data_window[:, delivery_time, -forecasting_horizon + i]
            - daily_data_window[:, delivery_time, -information_shift]
        )
    Y = np.array(Y_list).T

    # extract the volume trajectory for forecasted delivery
    volume_unaggr_undiff = daily_data_window[
        :, 96 + delivery_time, : -information_shift + 1
    ]

    """
    ADDING THE EXOGENOUS VARIABLES
    """

    # load the exogenous variables to cache
    cache = {}
    cache = load_exogenous_to_cache(
        cache, first_training_date, delivery_time, data_root=DATA_DIR
    )

    X_exog_fundamental_plus_price = []

    # stack X vector and naive forecast (last known price without differencing)
    X_exog_fundamental_plus_price.append(X_naive)

    # add the last known aggregated volume information
    X_exog_fundamental_plus_price.append(X_exog)

    X_exog_fundamental_plus_price = add_exogenous_from_cache_to_variables(  # add variables that are known for the delivery we forecast for
        cache,
        X_exog_fundamental_plus_price.copy(),
        delivery_time,
        date_fore,
        trade_time,
        forecasting_horizon,
        total_training_days_available,
    )

    X_exog_fundamental_plus_price = add_last_known_exogenous_from_cache(  # add variables that are known only for period before forecasting time
        cache,
        X_exog_fundamental_plus_price.copy(),
        date_fore,
        trade_time,
        forecasting_horizon,
        total_training_days_available,
    )

    # add the sum of volume traded for forecasted delivery in the last 60 minutes (12 5min intervals)
    X_exog_fundamental_plus_price.append(
        np.expand_dims(
            np.sum(volume_unaggr_undiff[-total_training_days_available:, -12:], axis=1),
            1,
        )
    )
    # add the number of minutes with trades in them in the last 60 minutes (12 5min intervals)
    X_exog_fundamental_plus_price.append(
        np.expand_dims(
            np.sum(
                volume_unaggr_undiff[-total_training_days_available:, -12:] > 0, axis=1
            ),
            1,
        )
    )
    # add dummies
    dummies_col = daily_data_window[:, -1, -1]
    X_exog_fundamental_plus_price.append(
        dummies_col[-total_training_days_available:, np.newaxis]
    )
    # prepare a set of exog for kernel regression
    X_exog_fundamental_plus_price = np.concatenate(
        X_exog_fundamental_plus_price, axis=1
    )

    """
    ADDING THE SCENARIO VARIABLES
    """
    if args.probab_approach == "weather_scenarios":
        (
            X_exog_fundamental_plus_price,
            historical_scenarios_training_norm,
            historical_scenarios_similarity_measures_norm_training,
        ) = build_weather_scenarios_and_similarity(
            cache,
            date_fore,
            trade_time,
            forecasting_horizon,
            total_training_days_available,
            weather_scenarios_split_direction,
            X_exog_fundamental_plus_price.copy(),
        )
    elif args.probab_approach == "hist_insample":
        (
            _,
            _,
            historical_scenarios_similarity_measures_norm_training,
        ) = build_weather_scenarios_and_similarity(
            cache,
            date_fore,
            trade_time,
            forecasting_horizon,
            total_training_days_available,
            weather_scenarios_split_direction,
            X_exog_fundamental_plus_price.copy(),
            only_similarity=True,
        )

    """
    APPLY SCALER TO EXPLANATORY DATA AND TARGET
    """

    # transform the target
    scaler = StandardScaler()

    if args.probab_approach == "hist_insample":
        # we do not have future in the last training sample, thus we can use it too to compute mean and std
        X_exog_fundamental_plus_price = scaler.fit_transform(
            X_exog_fundamental_plus_price
        )

    else:
        # we have future weather data in the last sample - we need to drop it for scaler fitting step
        X_exog_fundamental_plus_price[:-1, :] = scaler.fit_transform(
            X_exog_fundamental_plus_price[:-1, :]
        )
        X_exog_fundamental_plus_price[-1, :] = scaler.transform(
            X_exog_fundamental_plus_price[-1, :].reshape(1, -1)
        )

    # define the training windows for each variable set
    training_window_fundamental_plus_price = X_exog_fundamental_plus_price[:-1, :]

    # define the naive vector
    naive_vec = daily_data_window[:, delivery_time, -information_shift]
    naive_vec_standardized = (naive_vec - np.mean(naive_vec)) / np.std(naive_vec)

    """
    TRAIN THE MODELS AND FORECAST THE PRICE
    """
    results = pd.DataFrame()
    step = 1  # tells the alg how many steps to skip, e.g. 3 will forecast each 15min 5min interval - useful for development purposes as it speeds up the simulation significantly
    limited_path = [i for i in range(0, 31, step)]
    scaler_target = StandardScaler()  # define transformer for the target
    Y_limited_path_len = scaler_target.fit_transform(Y[:-1, limited_path])
    Y_limited_path_len = Y_limited_path_len[
        -total_training_days_available + 1 :, :
    ]  # ensure the fit to the length of X

    # for most of the approaches to scenarios sampling we do not need to do it separately for all estimators - so we calc it above the loop to save some time
    if (
        args.scenarios_sampling_method != "dual_coeff"
        and args.probab_approach == "weather_scenarios"
    ):
        required_scenarios_indices, latest_scenarios = filter_scenarios(
            historical_scenarios_training_norm,
            historical_scenarios_similarity_measures_norm_training,
            None,
            required_scenarios,
            args.scenarios_sampling_method,
            args.ffs_measure,
        )

    kernel_function = (
        CorrectedSVRKernelWrapper()
    )  # custom kernel functions wrapped in a kernel class

    training_data = training_window_fundamental_plus_price
    test_data = X_exog_fundamental_plus_price[np.newaxis, -1, :]

    training_data_aggregated = np.hstack(  # adding naive vector to the variables to allow for the correction of kernel (cSVR)
        (naive_vec_standardized[:-1, np.newaxis], training_data)
    )
    test_data_aggregated = np.hstack(
        (naive_vec_standardized[-1:, np.newaxis], test_data)
    )

    for path_svr_estimator in ["CHAIN", "MULTI"]:
        estimator = SVR(kernel=kernel_function, epsilon=svr_epsilon, C=C)

        if path_svr_estimator == "CHAIN":
            estimator = RegressorChain(estimator)
        elif path_svr_estimator == "MULTI":
            estimator = MultiOutputRegressor(estimator)

        estimator.fit(training_data_aggregated, Y_limited_path_len)  # FIT THE SVR MODEL
        insample_forecast = estimator.predict(training_data_aggregated)
        results[f"{path_svr_estimator}_insample_MAE"] = [
            my_mae(insample_forecast[:, i], Y_limited_path_len[:, i])
            for i in range(np.shape(Y_limited_path_len)[1])
        ]

        all_forecasts = []
        deltas = []  # Wasserstein deltas
        wasserstein = np.inf  # initialize the Wasserstein measure

        if args.probab_approach == "weather_scenarios":
            # get the filtered scenarios
            if args.scenarios_sampling_method == "dual_coeff":
                required_scenarios_indices, latest_scenarios = filter_scenarios(
                    historical_scenarios_training_norm,
                    historical_scenarios_similarity_measures_norm_training,
                    estimator,
                    required_scenarios,
                    args.scenarios_sampling_method,
                    args.ffs_measure,
                )

            all_scenarios = []
            for scenario_idx, scenario in enumerate(
                required_scenarios_indices
            ):  # prepare weather scenarios by overwriting the actual delta of deltas with historical scenarios
                test_data_aggregated_scenario = test_data_aggregated.copy()
                test_data_aggregated_scenario[:, -np.shape(latest_scenarios)[1] :] = (
                    latest_scenarios[scenario, :]
                )
                all_scenarios.append(test_data_aggregated_scenario)

            preds = estimator.predict(
                np.array(all_scenarios)[:, 0, :]
            )  # predict all the weather scenarios at once - on average faster than iteratively even if we limit the scenarios no.
            preds_inverse_transformed = scaler_target.inverse_transform(preds)

            for scenario_idx, scenario in enumerate(required_scenarios_indices):
                pred = preds[scenario_idx, :]
                results = results.copy()
                forecast = preds_inverse_transformed[scenario_idx, :] + last_known_price
                results[f"{path_svr_estimator}_prediction_{scenario}"] = forecast
                all_forecasts.append(pred)

                # check the stopping crit for scenarios addition
                if args.wasserstein_stopping_crit:
                    wasserstein_stop, wasserstein, deltas = check_wasserstein_stopping(
                        scenario_idx,
                        all_forecasts,
                        wasserstein,
                        deltas,
                        quantile_diff_tolerance,
                        min_scenarios,
                        wasserstein_moving_avg_window,
                    )

                    if wasserstein_stop:
                        break

        elif args.probab_approach == "hist_insample":
            base_predictions = estimator.predict(test_data_aggregated)
            historical_residues = Y_limited_path_len - insample_forecast
            required_scenarios_indices, latest_scenarios = filter_scenarios(
                historical_residues,
                historical_scenarios_similarity_measures_norm_training,
                estimator,
                required_scenarios,
                args.scenarios_sampling_method,
                args.ffs_measure,
            )
            scenario_forecast = base_predictions + latest_scenarios

            all_forecasts = []

            # save the base forecast
            results[f"{path_svr_estimator}_prediction_base"] = (
                scaler_target.inverse_transform(base_predictions.reshape(1, -1))[0, :]
                + last_known_price
            )

            for scenario_idx, scenario in enumerate(required_scenarios_indices):
                pred = scenario_forecast[scenario]
                results = results.copy()
                forecast = (
                    scaler_target.inverse_transform(pred.reshape(1, -1))[0, :]
                    + last_known_price
                )
                results[f"{path_svr_estimator}_prediction_{scenario}"] = forecast
                all_forecasts.append(pred)

                # check the stopping crit for scenarios addition
                if args.wasserstein_stopping_crit:
                    wasserstein_stop, wasserstein, deltas = check_wasserstein_stopping(
                        scenario_idx,
                        all_forecasts,
                        wasserstein,
                        deltas,
                        quantile_diff_tolerance,
                        min_scenarios,
                        wasserstein_moving_avg_window,
                    )

                    if wasserstein_stop:
                        break

    results["actual"] = daily_data_window[-1, delivery_time, -forecasting_horizon:][
        limited_path
    ]
    results["naive"] = [last_known_price] * len(limited_path)

    results_filename = f"{calibration_flag}_{str((pd.to_datetime(date_fore) - pd.Timedelta(days=1)).replace(hour=16) + pd.Timedelta(minutes=5 * (trade_time - 1))).replace(':', ';')}_{forecasting_horizon}.csv"

    result_file_name = os.path.join(
        MODEL_RESULTS_DIR,
        specific_results_folname,
        results_filename,
    )

    try:
        results.to_csv(result_file_name)
    except Exception as err:
        os.remove(result_file_name)
        raise KeyboardInterrupt(
            f"Interrupted on saving: last file removed to avoid empty files. Exception: {err}"
        )


if __name__ == "__main__":
    con = sqlite3.connect(
        os.path.join(DATA_DIR, "preprocessed_continuous_intraday_prices_and_volume.db")
    )
    sql_str = f"SELECT * FROM with_dummies WHERE Index_daily <= {trade_time} AND Time >= '{first_trading_start_of_simulation}';"  # load only the data required for simu, so up to last trade time in the trajectory
    daily_data = pd.read_sql(sql_str, con)[
        needed_columns_of_continuous_preprocessed_data
    ].to_numpy()  # column 288 contains weekday no. indicators
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

    # make sure to free the memory from unneccessary large variables
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
            [
                lookback + idx + 1,
                date,
                forecasting_horizon,
                "test",
            ]
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
            [
                lookback + idx + 1,
                date,
                forecasting_horizon,
                "test",
            ]
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
            f"timing_results_model_d_{delivery_time}_t_{trade_time}.txt",
        ),
        "w",
    ) as file:
        file.write(f"Execution time: {simu_end - simu_start} seconds\n")
