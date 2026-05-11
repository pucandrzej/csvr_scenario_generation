"""A collection of helpers allowing to load the exogenous variables into the forecasting simulation."""

import os
import pandas as pd
import numpy as np


def load_csv(data_root, cutoff, *path_parts, **kwargs):
    """
    Load a CSV from a path constructed via os.path.join and
    filter rows to dates >= cutoff.

    Parameters
    ----------
    data_root : str
        Base directory for all files.
    cutoff : datetime
        Minimum date to include.
    path_parts : tuple
        Directory + filename parts to join.
    kwargs : dict
        Forwarded to pandas.read_csv().

    Returns
    -------
    DataFrame
    """
    full_path = os.path.join(data_root, *path_parts)
    df = pd.read_csv(full_path, index_col=0, **kwargs)
    df.index = pd.to_datetime(df.index)
    return df[df.index >= cutoff]


def load_exogenous_to_cache(cache, first_training_date, delivery_time, data_root=None):
    """
    Load all required exogenous datasets into cache using robust os.path.join.

    The helper load_csv() is now defined above, NOT inside this function.
    """

    cutoff = (
        first_training_date - pd.Timedelta(days=1)
    )  # always loading one day more as we are touching/crossing the midnight with some paths

    # ---------- Load datasets ----------

    cache["DE_DA_qtrly"] = load_csv(
        data_root,
        cutoff,
        "Day-Ahead-Quarterly-Data",
        "DE_quarterhourly_day_ahead_prices.csv",
    )

    cache["DA_hourly"] = load_csv(
        data_root,
        cutoff,
        "Day-Ahead-Quarterly-Data",
        "DE_and_all_DE_borders_hourly_day_ahead_prices.csv",
    )

    cache["ID_qtrly"] = load_csv(
        data_root,
        cutoff,
        "ID_auction_preprocessed",
        "ID_auction_price_2018-2020_preproc.csv",
        parse_dates=True,
    )

    cache["Load"] = load_csv(
        data_root,
        cutoff,
        "Load",
        "Load_2018-2020.csv",
        parse_dates=["Time from"],
    )

    cache["Generation"] = load_csv(
        data_root,
        cutoff,
        "Generation",
        "Generation_2018-2020.csv",
    )

    cache["DE_physex"] = load_csv(data_root, cutoff, "Crossborder", "DE_physexc.csv")
    cache["DE_commex"] = load_csv(data_root, cutoff, "Crossborder", "DE_commex.csv")

    elasticity_filename = f"{delivery_time}_elasticities.csv"
    cache["Elasticities"] = load_csv(
        data_root, cutoff, "Elasticities", elasticity_filename
    )

    return cache


def add_exogenous_from_cache_to_variables(
    cache,
    variables,
    delivery_time,
    date_fore,
    trade_time,
    forecasting_horizon,
    total_training_days_available,
):
    """
    Populate `variables` list with all exogenous features using values stored in `cache`.

    Parameters
    ----------
    cache : dict
        Contains all loaded exogenous datasets.
    variables : list
        List that will be appended with new feature blocks.
    delivery_time : int
        Delivery period index (0–95 for quarter-hourly).
    date_fore : datetime.date or datetime-like
        Forecast date.
    trade_time : int
        Trading time index in minutes.
    forecasting_horizon : int
        Forecast horizon in minutes.
    X_naive, X_exog
        Precomputed fundamental vectors to include.
    """

    # ----- Extract cached datasets -----
    DE_DA_qtrly = cache["DE_DA_qtrly"]
    DA_hourly = cache["DA_hourly"]
    ID_qtrly = cache["ID_qtrly"]
    Load = cache["Load"]
    Gen = cache["Generation"]
    DE_commex = cache["DE_commex"]
    DE_intraday_auction_elasticity = cache["Elasticities"]

    # ----- Compute fundamental timing -----
    index_hour = int(delivery_time * 0.25)
    index_minute = int((delivery_time * 0.25 - index_hour) * 4 * 15)
    index_date = date_fore + pd.Timedelta(days=1)

    # Forecasting time
    exog_avail_mins = 5 * (trade_time - forecasting_horizon)
    forecasting_time = (pd.to_datetime(date_fore) - pd.Timedelta(days=1)).replace(
        hour=16
    ) + pd.Timedelta(minutes=exog_avail_mins)

    # ----- Adjust DA forecasts based on 08:00 update -----
    Gen_intraday_adjusted = Gen.copy(deep=True)
    if forecasting_time.hour >= 8:
        Gen_intraday_adjusted["W DA"] = Gen_intraday_adjusted["W ID"]
        Gen_intraday_adjusted["SPV DA"] = Gen_intraday_adjusted["SPV ID"]

    # ==================================================================
    # Start appending components to `variables`
    # ==================================================================

    # 1. load forecast
    Load_fore = Load[
        (Load.index.hour == index_hour)
        & (Load.index.minute == index_minute)
        & (Load.index < index_date)
    ]["Forecast"]

    variables.append(np.expand_dims(Load_fore[-total_training_days_available:], 1))

    # 2. renewable generation forecasts: solar and wind
    spv_da_fore = Gen_intraday_adjusted[
        (Gen_intraday_adjusted.index.hour == index_hour)
        & (Gen_intraday_adjusted.index.minute == index_minute)
        & (Gen_intraday_adjusted.index < index_date)
    ]["SPV DA"]

    wnd_da_fore = Gen_intraday_adjusted[
        (Gen_intraday_adjusted.index.hour == index_hour)
        & (Gen_intraday_adjusted.index.minute == index_minute)
        & (Gen_intraday_adjusted.index < index_date)
    ]["W DA"]

    variables.append(np.expand_dims(spv_da_fore[-total_training_days_available:], 1))
    variables.append(np.expand_dims(wnd_da_fore[-total_training_days_available:], 1))

    # 3. DA price
    DE_DA_price = DE_DA_qtrly[
        (DE_DA_qtrly.index.hour == index_hour)
        & (DE_DA_qtrly.index.minute == index_minute)
        & (DE_DA_qtrly.index < index_date)
    ]["DE"]

    variables.append(np.expand_dims(DE_DA_price[-total_training_days_available:], 1))

    # 4. border hourly prices
    DA_border_prices = DA_hourly[
        (DA_hourly.index.hour == index_hour) & (DA_hourly.index < index_date)
    ]
    variables.append(DA_border_prices[-total_training_days_available:])

    # 5. commercial exchange
    DE_commex_window = DE_commex[
        (DE_commex.index.hour == index_hour) & (DE_commex.index < index_date)
    ]
    variables.append(DE_commex_window[-total_training_days_available:])

    # 6. ID auction price
    ID_price = ID_qtrly[
        (ID_qtrly.index.hour == index_hour)
        & (ID_qtrly.index.minute == index_minute)
        & (ID_qtrly.index < index_date)
    ]["price"]

    variables.append(np.expand_dims(ID_price[-total_training_days_available:], 1))

    # 7. elasticity
    elasticity = DE_intraday_auction_elasticity.loc[
        DE_intraday_auction_elasticity.index < forecasting_time,
        ["0", "1", "2"],
    ]

    variables.append(elasticity[-total_training_days_available:])

    return variables


def add_neg_and_pos_variables(variables, variable):
    """Add negative and positive side of the variable as two separate variables to the variables set"""
    variables.append(
        np.expand_dims(
            np.minimum(variable, 0),
            1,
        )
    )
    variables.append(
        np.expand_dims(
            np.maximum(variable, 0),
            1,
        )
    )
    return variables


def add_last_known_exogenous_from_cache(
    cache,
    variables,
    date_fore,
    trade_time,
    forecasting_horizon,
    total_training_days_available,
):
    """
    Add last known load, generation, and physical exchange information (including errors)
    to the `variables` list using only cached data.

    Parameters
    ----------
    cache : dict
        Dictionary containing cached exogenous datasets:
        Load, Generation, DE_physex, DE_commex.
    variables : list
        List to append feature arrays into.
    date_fore : datetime-like
        Forecasting date.
    trade_time : int
        Trading time index in minutes.
    forecasting_horizon : int
        Forecast horizon in minutes.
    """

    Load = cache["Load"]
    Gen = cache["Generation"]
    DE_physex = cache["DE_physex"]
    DE_commex = cache["DE_commex"]

    # Loop inputs
    exog_sets = [Load, Gen, DE_physex]

    for exog_idx, exog_df in enumerate(exog_sets):
        # Load & generation use shift 61; physical exchange uses 121
        shift = (
            61 + 15 if exog_idx != 2 else 121 + 60
        )  # +15 to account for the fact that we need to take the previous (already closed) 15min interval, analogously 60 is for hourly granularity

        # Compute datetime at which info becomes available
        exog_avail_mins = 5 * (trade_time - forecasting_horizon)
        datetime_avail = (pd.to_datetime(date_fore) - pd.Timedelta(days=1)).replace(
            hour=16
        ) + pd.Timedelta(minutes=exog_avail_mins - shift)

        # ---------------------------------------------------------
        # Select only available rows from dataframes (date shift)
        # ---------------------------------------------------------
        if datetime_avail.date() < date_fore.date():
            # we are shifted by one day
            limit_date = pd.to_datetime(date_fore.date())
        else:
            limit_date = pd.to_datetime(date_fore.date() + pd.Timedelta(days=1))

        exog_df_limited = exog_df[exog_df.index <= limit_date]
        DE_commex_sel = DE_commex[DE_commex.index <= limit_date]

        # ---------------------------------------------------------
        # Hour / minute resolution differences for exog types
        # ---------------------------------------------------------
        if exog_idx != 2:
            # quarter-hour aligned
            exog_last_info = exog_df_limited[
                (exog_df_limited.index.hour == datetime_avail.hour)
                & (exog_df_limited.index.minute == (datetime_avail.minute // 15) * 15)
            ][-total_training_days_available:]

        else:
            # hourly physical exchange
            exog_last_info = exog_df_limited[
                exog_df_limited.index.hour == datetime_avail.hour
            ][-total_training_days_available:]

            DE_commex_last_info = DE_commex_sel[
                DE_commex_sel.index.hour == datetime_avail.hour
            ][-total_training_days_available:]

        # =========================================================
        # ADD FEATURES
        # =========================================================

        # ---------- 1) Load ----------
        if exog_idx == 0:
            # actual load
            variables.append(np.expand_dims(exog_last_info["Actual"], 1))

            # load forecast error
            load_error = exog_last_info["Actual"] - exog_last_info["Forecast"]
            variables = add_neg_and_pos_variables(variables, load_error)

        # ---------- 2) Generation (SPV, W) ----------
        elif exog_idx == 1:
            # actual SPV & wind
            variables.append(np.expand_dims(exog_last_info["SPV"], 1))
            variables.append(np.expand_dims(exog_last_info["W"], 1))

            # SPV error
            spv_err = exog_last_info["SPV"] - exog_last_info["SPV DA"]
            variables = add_neg_and_pos_variables(variables, spv_err)

            # wind error
            wnd_err = exog_last_info["W"] - exog_last_info["W DA"]
            variables = add_neg_and_pos_variables(variables, wnd_err)

        # ---------- 3) Physical Exchange ----------
        elif exog_idx == 2:
            # last known physical exchange
            variables.append(exog_last_info)

            # deviation from commercial schedule
            phys_minus_da = (
                exog_last_info[DE_commex_last_info.columns] - DE_commex_last_info
            )

            for col in phys_minus_da.columns:
                variables = add_neg_and_pos_variables(variables, phys_minus_da[col])

    return variables
