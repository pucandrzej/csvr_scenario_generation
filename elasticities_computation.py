"""Script computes the elasticities based on the intraday auction market supply/demand curves."""

import os
import sqlite3
import numpy as np
from tqdm import tqdm
import pandas as pd
from datetime import timedelta

from config.paths import DATA_DIR, MARKET_DATA_DIR
from config.test_calibration_validation import (
    required_start,
    required_end,
)
from config.forecasting_simulation_config import (
    deliveries_no,
    forecasting_horizon,
    trading_start_hour,
    first_trading_start_of_simulation,
    first_day_index_of_simulation,
    needed_columns_of_continuous_preprocessed_data,
    total_no_of_cont_market_columns,
    last_trade_time_in_path_delta,
)

from utils import (
    fill_march_dst_daily,
    check_for_missing_data,
    devel_elasticities_plot,
    price2vol_sup,
    price2vol_dem,
    sup_trans_inv,
    S_trans,
)

DEVEL_PLOTS = False  # if True the function will generate a plotly go interactive plot of elasticities, use ONLY with debugger - you will generate too many figures otherwise!
VOLUME_DELTAS = [
    500,
    1000,
    2000,
]
LOWER_TECHNICAL_MINIMUM = -3000  # technical min. price on intraday auction market


def compute_transformed_supply_elasticity(
    df_curve,
    P_clearing,
    P_auction,
    volume_delta,
    forecasting_date,
    P_min=-3000,
    side_supply="Sell",
    side_demand="Purchase",
):
    """
    - df_curve: DataFrame with columns ["Price","Volume","Sale/Purchase"].
    - P_clearing: observed clearing price P_{DA} (intraday price).
    - volume_delta: volume offset for finite-difference.
    - P_min: floor price (e.g. -3000).
    Returns: elasticity and shows 3-panel plots.
    """

    # Split curves
    df_sup = df_curve[df_curve["Sale/Purchase"] == side_supply].copy()
    df_dem = df_curve[df_curve["Sale/Purchase"] == side_demand].copy()

    if len(df_sup) == 0 or len(df_dem) == 0:
        return np.nan  # no offers - time change

    # Sort by Price for price→volume mappings
    df_sup_price = df_sup.sort_values("Price").reset_index(drop=True)
    df_dem_price = df_dem.sort_values("Price").reset_index(drop=True)

    # Compute key volumes
    q0 = price2vol_sup(
        P_auction, df_sup_price.copy()
    )  # original clearing volume from supply side
    DEM_inelastic = price2vol_dem(
        P_min, df_dem_price.copy()
    )  # inelastic demand volume at floor price

    q_implied = S_trans(
        P_clearing, df_sup_price.copy(), DEM_inelastic, df_dem_price.copy()
    )  # implied volume corresponding to the last known continuous market price (naive)

    # Build inversion arrays from breakpoints
    # Collect unique price levels from supply and demand
    P_vals = np.unique(
        np.concatenate([df_sup_price["Price"].values, df_dem_price["Price"].values])
    )

    # Compute transformed volumes at each price
    V_vals = np.array(
        [
            price2vol_sup(P, df_sup_price.copy())
            + DEM_inelastic
            - price2vol_dem(P, df_dem_price.copy())
            for P in P_vals
        ]
    )

    # Sort by volume to ensure monotonicity
    idx = np.argsort(V_vals)
    V_sorted = V_vals[idx]
    P_sorted = P_vals[idx]

    # Now invert: given z, find P via interpolation on (V_sorted, P_sorted)
    # Slope segment around q0
    P_plus = sup_trans_inv(q_implied + volume_delta, V_sorted.copy(), P_sorted.copy())
    P_minus = sup_trans_inv(q_implied - volume_delta, V_sorted.copy(), P_sorted.copy())

    # Compute elasticity: dP/dz ≈ (P_plus - P_minus)/(2*volume_delta)
    elasticity = (P_plus - P_minus) / (2 * volume_delta)

    if DEVEL_PLOTS:
        devel_elasticities_plot(
            df_sup_price=df_sup_price.copy(deep=True),
            df_dem_price=df_dem_price.copy(deep=True),
            P_auction=P_auction,
            P_minus=P_minus,
            P_clearing=P_clearing,
            P_plus=P_plus,
            q0=q0,
            volume_delta=volume_delta,
            V_sorted=V_sorted,
            P_sorted=P_sorted,
            q_implied=q_implied,
            DEM_inelastic=DEM_inelastic,
        )

    return elasticity


first_training_date = required_start
last_forecasting_date = required_end - timedelta(days=1)

for delivery_time in tqdm(range(deliveries_no)):
    last_trade_time = (
        delivery_time * 3 + last_trade_time_in_path_delta
    )  # 8*12 is 8 hours each containing 12 5min periods, -6 as we are trading up to 30min before the delivery

    information_shift = forecasting_horizon + 1

    first_trade_time = (
        last_trade_time - information_shift
    )  # absolute index of the first step in the path

    con = sqlite3.connect(MARKET_DATA_DIR)
    sql_str = f"SELECT * FROM with_dummies WHERE Index_daily <= {last_trade_time} AND Time >= '{first_trading_start_of_simulation}';"  # load only the data required for simu, so up to trade time
    daily_data = pd.read_sql(sql_str, con)[
        needed_columns_of_continuous_preprocessed_data
    ].to_numpy()  # column 288 contains weekday no. indicators
    daily_data = np.reshape(
        daily_data,
        (
            np.shape(daily_data)[0] // last_trade_time,
            last_trade_time,
            total_no_of_cont_market_columns,
        ),
    )

    last_known_id_prices = daily_data[:, -information_shift, delivery_time]

    # for each day required in the simulation extract outages known before 12:00 d-1 and outages reported between 12:00 d-1 and time of forecasting
    elasticities_per_date = []
    last_known_prices = []
    actual_datetimes = []

    for forecasting_date_idx, forecasting_date in enumerate(
        pd.date_range(first_training_date, last_forecasting_date)
    ):
        naive_datetime = (forecasting_date - timedelta(days=1)).replace(
            hour=trading_start_hour
        ) + timedelta(minutes=5 * first_trade_time)
        actual_datetimes.append(naive_datetime)

        last_known_id_price = last_known_id_prices[forecasting_date_idx]
        last_known_prices.append(last_known_id_price)

        df = pd.read_csv(
            os.path.join(
                DATA_DIR,
                "Intraday_Auction",
                "Aggregated curves",
                f"{forecasting_date.year}",
                f"intraday_auction_aggregated_curves_15-call_germany_{forecasting_date.strftime('%Y%m%d')}.csv",
            ),
            skiprows=1,
            dtype={"Hour": str},
        )

        # in 2019 the volume was not scaled to 15min, but rather reported as 1h value - we correct that, so that it is coherent with 2020 values of volume
        if forecasting_date.year == 2019:
            df["Volume"] = df["Volume"] / 4

        # handle the duplicated hours by dropping the 2nd one
        if "3A" in df["Hour"].values:
            df.loc[df.Hour == "3A", "Hour"] = "3"
            df = df[df.Hour != "3B"]

        # change the dtype of Hour column
        df["Hour"] = df["Hour"].astype(int)

        # extract demanded delivery
        df = df[
            (df["Hour"] == delivery_time // 4 + 1)
            & (df["Quarter hour"] == delivery_time % 4 + 1)
        ]

        # load the intraday auction clearing price
        intraday_auction_prices = pd.read_csv(
            os.path.join(
                DATA_DIR,
                "ID_auction_preprocessed",
                "ID_auction_price_2018-2020_preproc.csv",
            ),
            index_col=0,
        )
        intraday_auction_prices.index = pd.to_datetime(intraday_auction_prices.index)
        index_hour = int(delivery_time * 0.25)
        index_minute = int((delivery_time * 0.25 - index_hour) * 4 * 15)
        ID_price = intraday_auction_prices[
            (intraday_auction_prices.index.hour == index_hour)
            & (intraday_auction_prices.index.minute == index_minute)
            & (intraday_auction_prices.index.date == forecasting_date.date())
        ]["price"].values[0]

        all_volume_shifts = []
        for (
            volume_delta
        ) in VOLUME_DELTAS:  # volume deltas to calculate the slope between
            all_volume_shifts.append(
                compute_transformed_supply_elasticity(
                    df,
                    last_known_id_price,
                    ID_price,
                    volume_delta,
                    forecasting_date,
                    P_min=LOWER_TECHNICAL_MINIMUM,
                )
            )

        elasticities_per_date.append(all_volume_shifts)

    df_elasticities_per_date = pd.DataFrame(elasticities_per_date)
    df_elasticities_per_date.index = actual_datetimes
    df_elasticities_per_date["naive_used"] = last_known_prices
    df_elasticities_per_date = fill_march_dst_daily(
        df_elasticities_per_date, 0
    )  # handle the missing data from March dst change
    print(f"Calculated the elasticities for {delivery_time}.")
    check_for_missing_data(
        df_elasticities_per_date,
        (min(df_elasticities_per_date.index) + timedelta(days=1)).normalize(),
        (max(df_elasticities_per_date.index) + timedelta(days=1)).normalize(),
        freq="1d",  # we use max of index here as we sometimes have elasticities for first deliveries of the day where naive is taken from prev. day
    )
    df_elasticities_per_date.to_csv(
        os.path.join(DATA_DIR, "Elasticities", f"{delivery_time}_elasticities.csv")
    )
