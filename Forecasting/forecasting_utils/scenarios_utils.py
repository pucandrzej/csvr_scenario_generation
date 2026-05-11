"""A collection of helpers for scenarios generation: SVS, FFS and weather scenarios implementations."""

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.stats import wasserstein_distance_nd
from datetime import timedelta


def filter_scenarios(
    historical_scenarios_training_norm,
    historical_scenarios_similarity_measures_norm_training,
    estimator,
    required_scenarios,
    scenarios_sampling_method,
    ffs_measure,
):
    """
    Filter and select historical scenarios based on the chosen sampling method.

    Parameters
    ----------
    historical_scenarios_training_norm : ndarray
        Array of normalized historical scenarios used for model training.
        Shape: (n_samples, n_features).
    historical_scenarios_similarity_measures_norm_training : ndarray
        Array containing similarity features for clustering-based sampling.
        Shape: (n_samples, n_similarity_features).
    estimator : object
        A trained ensemble estimator (SVR models for each path element).
        Must expose ``estimators_`` (collection of SVR models), each having
        ``support_`` and ``dual_coef_`` attributes when using ``dual_coeff`` mode.

    Returns
    -------
    required_scenarios_indices : list of int
        Indices of selected scenarios based on the sampling method.
    latest_scenarios : ndarray
        The filtered (latest) scenarios used for sampling.
        Returned so downstream code keeps consistent indices.
    """
    # limit the scenarios to the latest batch
    if required_scenarios is not None:
        latest_scenarios = historical_scenarios_training_norm[-required_scenarios:, :]
        latest_scenarios_similarity = (
            historical_scenarios_similarity_measures_norm_training[
                -required_scenarios:, :
            ]
        )
    else:
        latest_scenarios = historical_scenarios_training_norm.copy()
        latest_scenarios_similarity = (
            historical_scenarios_similarity_measures_norm_training.copy()
        )
    # sort the scenarios based on the dual coeff from SVR model
    if scenarios_sampling_method == "dual_coeff":
        # 1. zero-initialize
        weights = np.zeros(len(latest_scenarios), dtype=float)

        # 2. accumulate absolute dual coefficients
        for fitted_svr in estimator.estimators_:
            for idx, alpha in zip(fitted_svr.support_, fitted_svr.dual_coef_[0]):
                if required_scenarios is not None:
                    if (
                        idx
                        >= len(historical_scenarios_training_norm) - required_scenarios
                    ):  # if we filtered out only the recent scenarios we need to also filter them here as the model was still trained on the whole sample
                        weights[
                            idx
                            - (
                                len(historical_scenarios_training_norm)
                                - required_scenarios
                            )
                        ] += abs(alpha)
                else:
                    weights[idx] += abs(alpha)

        # 3. sort the scenarios based on the dual coefficient
        required_scenarios_indices = np.argsort(
            weights
        )[
            ::-1
        ]  # argsort is sorting in increasing order, we want it in decreasing thus [::-1]

    # use the FFS approach
    elif scenarios_sampling_method == "FFS":
        required_scenarios_indices, _ = ffs(
            latest_scenarios, m=len(latest_scenarios), ffs_measure=ffs_measure
        )

    # select a cluster of similar days scenarios based on known forecasts/actuals from current day
    elif scenarios_sampling_method == "clustering":
        clustering = HDBSCAN(
            min_cluster_size=10,
            store_centers="centroid",
            allow_single_cluster=True,  # useful for the small samples (e.g. the last 28d)
        )
        labels = clustering.fit_predict(latest_scenarios_similarity)

        unique_labels = np.unique(labels)
        centroids = clustering.centroids_

        if (
            len(unique_labels) == 1 and unique_labels[0] == -1
        ):  # in case all is noise (no close dense scenarios group) we use k means to split the data into the closer and a more distant cluster
            clustering = KMeans(n_clusters=2, n_init="auto")
            labels = clustering.fit_predict(latest_scenarios_similarity)
            unique_labels = np.unique(labels)
            centroids = clustering.cluster_centers_

        # eliminate the -1 as we do not have centroid for it
        unique_labels = unique_labels[unique_labels != -1]
        closest_cluster_label = unique_labels[np.argmin(np.mean(centroids, axis=1))]
        required_scenarios_indices = [
            i for i, lbl in enumerate(labels) if lbl == closest_cluster_label
        ]

    # take the scenarios "as is" i.e. sorted in time t-N, ..., t
    elif scenarios_sampling_method is None:
        required_scenarios_indices = range(len(latest_scenarios))

    else:
        raise ValueError(
            f"Unknown scenarios sampling method: {scenarios_sampling_method}"
        )

    return required_scenarios_indices, latest_scenarios


def ffs(scenarios, m, ffs_measure, tol=None, verbose=False):
    """
    Forward feature selection on `scenarios` using MAE, no weighting.
    This approach tends to favour the addition of new samples
    that are close to the poorly represented regions in current sample.

    See paper:
    Scenario Reduction Algorithms in Stochastic Programming 2003

    Parameters
    ----------
    scenarios : array-like, shape (N, L)
        N scenario‐vectors of length L.
    m : int
        Maximum size of the reduced support.
    tol : float or None
        If set, stop when the drop in average MAE < tol.
    verbose : bool
        If True, print each addition’s info.

    Returns
    -------
    selected : list of int
        Indices of chosen scenarios in selection order.
    costs : list of float
        Mean‐MAE to full set after each addition.
    """
    N, L = scenarios.shape

    # Precompute pairwise MAE distances
    # D[i,j] = mean(|scenarios[i] - scenarios[j]|)
    if ffs_measure == "RMSE":
        D = np.sqrt(((scenarios[:, None, :] - scenarios[None, :, :]) ** 2).mean(axis=2))
    elif ffs_measure == "MAE":
        D = np.abs(scenarios[:, None, :] - scenarios[None, :, :]).mean(axis=2)

    # seed: pick element minimizing sum_k D[k,l]
    seed = np.argmin(D.sum(axis=0))
    selected = [seed]

    # current nearest‐neighbor distances for each scenario
    nearest = D[:, seed]
    cost = nearest.mean()
    costs = [cost]

    # greedy add up to m
    for i in range(m - 1):
        best_cost = np.inf
        best_u = None

        for u in range(N):
            if u in selected:
                continue
            # if we add u, new nearest = min(old, D[:,u]) - fast way of doing it as instead of recalculating min over old set + new element we just add the new element and min of it and old min
            cand = np.minimum(nearest, D[:, u])
            cand_cost = cand.mean()
            if cand_cost < best_cost:
                best_cost, best_u = cand_cost, u

        # optional early stop
        if tol is not None and (cost - best_cost) < tol:
            break

        selected.append(best_u)
        nearest = np.minimum(nearest, D[:, best_u])
        cost = best_cost
        costs.append(cost)

    return selected, costs


def _daily_mae(df, column):
    """
    Compute the mean absolute error (MAE) between each historical day and the
    most recent day in a time series. Used in weather scenarios preparation logic.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with a datetime index and at least one numeric column.
    column : str
        Column name containing the values to compare.

    Returns
    -------
    pandas.Series
        A Series indexed by date, where each value is the MAE between that day
        and the last day in the dataset. The last day itself is excluded.
    """
    df = df.copy()
    df.index = pd.to_datetime(df.index)

    # pivot into a (n_times × n_days) array: rows=quarter-hours, cols=dates
    pivot = (
        df[[column]]
        .assign(
            date=lambda d: d.index.date, time=lambda d: d.index.time
        )  # add the separate date and time columns
        .reset_index(drop=False)  # reset index to keep it in the columns
        .pivot(
            index="time", columns="date", values=column
        )  # pivot to periods x days matrix
        .sort_index(axis=1)  # make sure the dates are sorted
    )

    ref = pivot.iloc[:, -1]  # reference date: current one
    mae = pivot.sub(ref, axis=0).abs().mean()  # MAE with all the other dates

    return mae


def _update_scenarios_matrix(
    deltas_delta,
    historical_scenarios,
    scenario_variable_counter,
    required_scenario_days,
    weather_scenarios_split_direction,
):
    """Add weather scenarios to the explanatory variables matrix.
    Each scenario is added as two variables: nonnegative and non positive.
    Used in weather scenarios preparation logic.
    """
    if weather_scenarios_split_direction:
        deltas_delta = np.column_stack(  # prepare the positive and negative variables
            (np.maximum(deltas_delta, 0), np.minimum(deltas_delta, 0))
        )[-required_scenario_days:, :]
        historical_scenarios[:, scenario_variable_counter] = deltas_delta[:, 0]
        scenario_variable_counter += 1
        historical_scenarios[:, scenario_variable_counter] = deltas_delta[:, 1]
        scenario_variable_counter += 1
    else:
        deltas_delta = np.expand_dims(deltas_delta[-required_scenario_days:], 1)
        historical_scenarios[:, scenario_variable_counter] = deltas_delta[:, 0]
        scenario_variable_counter += 1

    return deltas_delta, historical_scenarios, scenario_variable_counter


def build_weather_scenarios_and_similarity(
    cache,
    date_fore,
    trade_time,
    forecasting_horizon,
    total_training_days_available,
    weather_scenarios_split_direction,
    X_exog_fundamental_plus_price,
    only_similarity=False,
):
    """
    Build historical weather scenarios (deltas) and similarity measures
    for weather-scenario‐based probabilistic forecasting.

    Parameters
    ----------
    cache : dict
        Dictionary containing cached exogenous datasets:
        Load, Generation, DE_physex, DE_commex.
    date_fore : datetime-like
        Forecasting date.
    trade_time : int
        Trading interval index.
    forecasting_horizon : int
        Forecasting horizon in minutes.
    total_training_days_available : int
        Number of training days.
    weather_scenarios_split_direction : bool
        Whether to double the number of scenario variables.
    X_exog_fundamental_plus_price : np.ndarray
        Design matrix to which scenario deltas will be appended.

    Returns
    -------
    historical_scenarios_training_norm : np.ndarray
    historical_scenarios_similarity_measures_norm_training : np.ndarray
    scaler : StandardScaler
    hist_scenarios_similarity_scaler : MinMaxScaler
    historical_scenarios : np.ndarray
    """

    exog_avail_mins = 5 * (trade_time - forecasting_horizon)
    Load = cache["Load"]
    gen_copy = cache["Generation"]

    # ---- scenario matrix size ----
    if weather_scenarios_split_direction:
        hist_scenarios_no_for_trajectory = int(np.ceil(forecasting_horizon / 3)) * 6
    else:
        hist_scenarios_no_for_trajectory = int(np.ceil(forecasting_horizon / 3)) * 3

    historical_scenarios = (
        np.ones((total_training_days_available, hist_scenarios_no_for_trajectory))
        * np.nan
    )  # declare the historical scenarios matrix

    historical_scenarios_similarity_measures = []
    scenario_variable_counter = 0

    # =====================================================================
    # scenario computation
    # =====================================================================
    for exog_idx, exog_df in enumerate([Load, gen_copy]):
        shift = 61 + 15  # data availability shift for load and gens

        datetime_avail = (pd.to_datetime(date_fore) - timedelta(days=1)).replace(
            hour=16
        ) + timedelta(
            minutes=exog_avail_mins - shift
        )  # the timestamp before which we expect the ENTSOe data to be available

        last_trade_time = (pd.to_datetime(date_fore) - timedelta(days=1)).replace(
            hour=16
        ) + timedelta(
            minutes=5 * (trade_time - 1)
        )  # the end of the path we want to forecast

        exog_df = exog_df[
            exog_df.index <= last_trade_time
        ]  # trim the data up to the end of the path

        exog_last_info = exog_df[
            (exog_df.index.hour == datetime_avail.hour)
            & (exog_df.index.minute == (datetime_avail.minute // 15) * 15)
        ]

        exog_known_at_forecasting = exog_df[
            (exog_df.index <= datetime_avail)
            & (
                (exog_df.index.hour < datetime_avail.hour)
                | (
                    (exog_df.index.hour == datetime_avail.hour)
                    & (exog_df.index.minute <= datetime_avail.minute)
                )
            )
        ]

        # ---- collect the similarity measures between todays actual - forecast delta and training sample deltas for every variable ----
        if exog_idx == 0:
            historical_scenarios_similarity_measures.append(
                pd.concat(
                    [
                        _daily_mae(exog_known_at_forecasting, "Actual"),
                        _daily_mae(exog_known_at_forecasting, "Forecast"),
                    ],
                    axis=1,
                )[-total_training_days_available:]
            )

        elif exog_idx == 1:
            historical_scenarios_similarity_measures.append(
                pd.concat(
                    [
                        _daily_mae(exog_known_at_forecasting, "SPV"),
                        _daily_mae(exog_known_at_forecasting, "SPV DA"),
                    ],
                    axis=1,
                )[-total_training_days_available:]
            )
            historical_scenarios_similarity_measures.append(
                pd.concat(
                    [
                        _daily_mae(exog_known_at_forecasting, "W"),
                        _daily_mae(exog_known_at_forecasting, "W DA"),
                    ],
                    axis=1,
                )[-total_training_days_available:]
            )

        if only_similarity:  # we do not want to collect the weather scenarios in case of historical simulation
            continue

        # ---- build weather scenarios for future path intervals ----
        for future_interval in range(int(np.ceil(forecasting_horizon / 3))):
            datetime_max_required_fore = (  # weather scenario path 15min interval
                pd.to_datetime(date_fore) - timedelta(days=1)
            ).replace(hour=16) + timedelta(
                minutes=5 * (trade_time - 1) - future_interval * 15
            )

            exog_fore = (
                exog_df[  # values of the weather scenarios for chosen path interval
                    (exog_df.index.hour == datetime_max_required_fore.hour)
                    & (
                        exog_df.index.minute
                        == (datetime_max_required_fore.minute // 15) * 15
                    )
                ]
            )

            if (
                datetime_max_required_fore.date() > datetime_avail.date()
            ):  # if the path is crossing the midnight we need to adjust the days lookback
                exog_fore = exog_fore.iloc[1:]

            # =============================
            # Load scenario deltas
            # =============================
            if exog_idx == 0:
                delta_known = (
                    exog_last_info["Actual"] - exog_last_info["Forecast"]
                )  # delta known at forecasting
                delta_fore = (
                    exog_fore["Actual"] - exog_fore["Forecast"]
                )  # delta in the future interval
                deltas_delta = (
                    delta_known.values - delta_fore.values
                )  # operation on numpy vectors to avoid the need for adjusting the indices

                deltas_delta, historical_scenarios, scenario_variable_counter = (
                    _update_scenarios_matrix(
                        deltas_delta,
                        historical_scenarios,
                        scenario_variable_counter,
                        total_training_days_available,
                        weather_scenarios_split_direction,
                    )
                )
                X_exog_fundamental_plus_price = (
                    np.hstack(  # add the weather scenarios to the explanatory variables
                        (X_exog_fundamental_plus_price, deltas_delta)
                    )
                )

            # =============================
            # Generation scenario deltas
            # =============================
            elif exog_idx == 1:
                # SPV
                delta_known = exog_last_info["SPV"] - exog_last_info["SPV DA"]
                delta_fore = exog_fore["SPV"] - exog_fore["SPV DA"]
                deltas_delta = delta_known.values - delta_fore.values

                deltas_delta, historical_scenarios, scenario_variable_counter = (
                    _update_scenarios_matrix(
                        deltas_delta,
                        historical_scenarios,
                        scenario_variable_counter,
                        total_training_days_available,
                        weather_scenarios_split_direction,
                    )
                )
                X_exog_fundamental_plus_price = (
                    np.hstack(  # add the weather scenarios to the explanatory variables
                        (X_exog_fundamental_plus_price, deltas_delta)
                    )
                )

                # Wind
                delta_known = exog_last_info["W"] - exog_last_info["W DA"]
                delta_fore = exog_fore["W"] - exog_fore["W DA"]
                deltas_delta = delta_known.values - delta_fore.values

                deltas_delta, historical_scenarios, scenario_variable_counter = (
                    _update_scenarios_matrix(
                        deltas_delta,
                        historical_scenarios,
                        scenario_variable_counter,
                        total_training_days_available,
                        weather_scenarios_split_direction,
                    )
                )
                X_exog_fundamental_plus_price = (
                    np.hstack(  # add the weather scenarios to the explanatory variables
                        (X_exog_fundamental_plus_price, deltas_delta)
                    )
                )

    # assemble similarity matrix based on trajectory of actual and forecast before the forecasting time
    historical_scenarios_similarity_measures_concatenated = pd.concat(
        historical_scenarios_similarity_measures, axis=1
    ).to_numpy()

    # normalize the similarity (min-max)
    hist_scenarios_similarity_scaler = MinMaxScaler()
    historical_scenarios_similarity_measures_norm_training = (
        hist_scenarios_similarity_scaler.fit_transform(
            historical_scenarios_similarity_measures_concatenated[
                :-1, :
            ]  # we cut the last day as it is not "training" day
        )
    )

    if only_similarity:
        return None, None, historical_scenarios_similarity_measures_norm_training

    # normalize the weather scenarios (standard scaler)
    scaler = StandardScaler()
    historical_scenarios_training_norm = scaler.fit_transform(
        historical_scenarios[:-1, :]
    )

    return (
        X_exog_fundamental_plus_price,
        historical_scenarios_training_norm,
        historical_scenarios_similarity_measures_norm_training,
    )


def check_wasserstein_stopping(
    scenario_idx: int,
    all_forecasts: list,
    wasserstein: float,
    deltas: list,
    quantile_diff_tolerance: float,
    min_scenarios: int = 10,
    wasserstein_moving_avg_window: int = 10,
):
    """
    Evaluate whether to stop adding scenarios based on Wasserstein distance convergence.
    The convergence is defined based on the average of last 10 Wasserstein distance step-to-step deltas.

    Parameters
    ----------
    scenario_idx : int
        Current scenario index in the loop.
    all_forecasts : list
        List of forecast vectors (usually standardized predictions).
    wasserstein : float
        Last recorded Wasserstein distance.
    deltas : list
        List of historical Wasserstein differences W_{n-1} - W_n.
    quantile_diff_tolerance : float
        Tolerance threshold for the moving average of |W_n - W_{n-1}|.
    min_scenarios : int
        Minimum number of scenarios required before stopping is allowed.

    Returns
    -------
    stop : bool
        True if stopping condition is met.
    wasserstein : float
        Updated Wasserstein distance.
    deltas : list
        Updated deltas list.
    """

    # --- Scenario 0: initialize with no breaking, Wasserstein distance set as inf and empty deltas
    if scenario_idx == 0:
        return False, np.inf, deltas

    # --- Compute new Wasserstein distance
    wasserstein_new = wasserstein_distance_nd(
        np.array(all_forecasts), np.array(all_forecasts[:-1])
    )

    # --- Scenario 1: record first value
    if scenario_idx == 1:
        return False, wasserstein_new, deltas

    # --- Scenario >= 2: compute deltas and stopping condition
    deltas.append(wasserstein - wasserstein_new)
    deltas_arr = np.array(deltas)

    wasserstein_deltas_ma = np.mean(
        np.abs(deltas_arr)[-wasserstein_moving_avg_window:]
    )  # moving average over step-to-step deltas

    if (wasserstein_deltas_ma < quantile_diff_tolerance) and (
        scenario_idx >= min_scenarios
    ):
        return True, wasserstein_new, deltas

    return False, wasserstein_new, deltas
