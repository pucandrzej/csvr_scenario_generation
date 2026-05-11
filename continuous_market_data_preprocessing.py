"""
Script that preprocesses the continuous market prices and volumes and adds the calendar day-of-week indicators.
"""

import glob
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from tqdm import tqdm

import sqlite3
import warnings
from config.paths import (
    DATA_DIR,
    MARKET_DATA_DIR,
    INTERMEDIATE_MARKET_DATA,
    CONCATENATED_RAW_MARKET_DATA,
    INITIALLY_PREPROCESSED_MARKET_DATA,
)
from config.test_calibration_validation import (
    required_start,
    required_end,
    summer_time_2020_dst,
    winter_time_2020_dst,
    data_reporting_standardization_start,
    data_reporting_standardization_end,
)
from config.forecasting_simulation_config import deliveries_no, trading_start_hour

warnings.filterwarnings("ignore")

SECONDS_TO_5MIN_SCALER = 300


def initial_preprocessing():
    """Loading the yearly continuous intraday market data.
    Initial handling of dst.
    Adjusting the 2020 datetimes which were stored under different assumptions.
    """
    if not os.path.exists(INTERMEDIATE_MARKET_DATA):
        # load the complete dataset
        if not os.path.exists(CONCATENATED_RAW_MARKET_DATA):
            df = pd.concat(
                [
                    pd.read_csv(
                        f,
                        skiprows=2,
                        names=[
                            "Date",
                            "Area Buy",
                            "Market Area Sell",
                            "Hour from",
                            "Hour to",
                            "Volume (MW)",
                            "Price (EUR)",
                            "Time Stamp",
                            "Trade ID",
                        ],
                    )
                    for f in glob.glob(
                        os.path.join(DATA_DIR, "Transactions", "*", "*.csv")
                    )
                ]
            )
            df[
                [
                    "Hour from",
                    "Hour to",
                    "Time Stamp",
                    "Price (EUR)",
                    "Volume (MW)",
                    "Date",
                ]
            ].to_csv(CONCATENATED_RAW_MARKET_DATA)
        else:
            df = pd.read_csv(
                CONCATENATED_RAW_MARKET_DATA,
                usecols=[
                    "Hour from",
                    "Hour to",
                    "Time Stamp",
                    "Price (EUR)",
                    "Volume (MW)",
                    "Date",
                ],
            )

        # select only the quarter-hourly deliveries from the dataset & the columns that we need
        df_copy = df[
            (df["Hour from"].str.contains("qh"))
            & (
                ~df["Hour from"].str.contains("B")
            )  # we do not consider the B hour in our analysis as averaging would include abnormally more points into the trajectory
            & (
                df["Hour from"] == df["Hour to"]
            )  # there is no such case in our dataset where hour from is different than hour to - this is just a safeguard
        ].reset_index()

        # create the datetime from and datetime offer time columns
        mod_trans_from = []
        offer_time = []
        for i, val in tqdm(enumerate(df_copy["Hour from"])):
            if "A" not in val:
                minutes = str((int(val.split("qh")[1]) - 1) * 15)
                hours = val.split("qh")[0]

                hours = str(int(hours) - 1)
                mod_trans_from.append(
                    datetime.strptime(
                        df_copy["Date"][i]
                        + " "
                        + hours.zfill(2)
                        + ":"
                        + minutes
                        + ":00",
                        "%d/%m/%Y %H:%M:%S",
                    )
                )
            elif "A" in val:
                minutes = str((int(val.split("Aqh")[1]) - 1) * 15)
                hours = str(int(val.split("Aqh")[0]) - 1)

                if datetime.strptime(
                    df_copy["Date"][i] + " " + hours.zfill(2) + ":" + minutes + ":00",
                    "%d/%m/%Y %H:%M:%S",
                ) < pd.to_datetime(df_copy.loc[i, "Time Stamp"]):
                    raise
                mod_trans_from.append(
                    datetime.strptime(
                        df_copy["Date"][i]
                        + " "
                        + hours.zfill(2)
                        + ":"
                        + minutes
                        + ":00",
                        "%d/%m/%Y %H:%M:%S",
                    )
                )

            offer_time.append(
                pd.Timestamp(
                    datetime.strptime(
                        df_copy["Time Stamp"][i][:-2] + "00", "%d/%m/%Y %H:%M:%S"
                    )
                ).floor("5min")
            )
        df_copy["Datetime from"] = mod_trans_from
        df_copy["Datetime offer time"] = offer_time
        df_copy.to_csv(INTERMEDIATE_MARKET_DATA)
    print(
        "Preliminary preprocessed already performed. Performing additional preprocessing..."
    )
    # read the preprocessed dataset
    df_copy = pd.read_csv(INTERMEDIATE_MARKET_DATA)

    # transform columns to datetime from string
    df_copy["Datetime offer time"] = pd.to_datetime(df_copy["Datetime offer time"])
    df_copy["Datetime from"] = pd.to_datetime(df_copy["Datetime from"])

    # shift trades from 14:00 to 15:00 on 24.10.2020 by 1h - patching the reporting inconsistency of provider
    df_copy.loc[
        (df_copy["Datetime offer time"] >= data_reporting_standardization_start)
        & (df_copy["Datetime offer time"] < data_reporting_standardization_end)
        & (df_copy["Datetime offer time"].dt.date != df_copy["Datetime from"].dt.date),
        "Datetime offer time",
    ] = df_copy.loc[
        (df_copy["Datetime offer time"] >= data_reporting_standardization_start)
        & (df_copy["Datetime offer time"] < data_reporting_standardization_end)
        & (df_copy["Datetime offer time"].dt.date != df_copy["Datetime from"].dt.date),
        "Datetime offer time",
    ] + timedelta(hours=1)

    # shift winter of 2020 by 1h
    df_copy.loc[
        (df_copy["Datetime from"].dt.year == 2020)
        & (
            (df_copy["Datetime from"] < summer_time_2020_dst)
            | (df_copy["Datetime from"] >= winter_time_2020_dst)
        ),
        "Datetime offer time",
    ] = df_copy.loc[
        (df_copy["Datetime from"].dt.year == 2020)
        & (
            (df_copy["Datetime from"] < summer_time_2020_dst)
            | (df_copy["Datetime from"] >= winter_time_2020_dst)
        ),
        "Datetime offer time",
    ] + timedelta(hours=1)

    # shift summer in 2020 by 2h
    df_copy.loc[
        (df_copy["Datetime from"] >= summer_time_2020_dst)
        & (df_copy["Datetime from"] < winter_time_2020_dst),
        "Datetime offer time",
    ] = df_copy.loc[
        (df_copy["Datetime from"] >= summer_time_2020_dst)
        | (df_copy["Datetime from"] < winter_time_2020_dst),
        "Datetime offer time",
    ] + timedelta(hours=2)

    # define TTD column as a difference between delivery and offer times
    df_copy["Time to delivery"] = (
        pd.to_datetime(df_copy["Datetime from"])
        - pd.to_datetime(df_copy["Datetime offer time"])
    ).dt.total_seconds() / SECONDS_TO_5MIN_SCALER  # 5 min aggregation

    # drop the columns unnecessary to analysis
    df_copy = df_copy.drop("Date", axis=1)
    df_copy = df_copy.drop("Time Stamp", axis=1)
    df_copy = df_copy.drop("Unnamed: 0", axis=1)
    df_copy = df_copy.drop("index", axis=1)
    df_copy = df_copy.drop("Hour from", axis=1)
    df_copy = df_copy.drop("Hour to", axis=1)

    # save the resulting table of unevenly spaced trades with volume, prices and day of the week
    df_copy.to_csv(
        INITIALLY_PREPROCESSED_MARKET_DATA,
        date_format="%s",
    )


def preprocess_data(start, end, ID_qtrly, add_dummies):
    """Preprocess the continuous market prices and volumes by aggregating prices into 5min bins using VWAP
    and summing volume in the same bins.
    """
    demanded_len = (
        32 * 12
    )  # daily data len (all 5 min intervals between trading start and day end)
    print("Cached data unavailable, preparing & saving the data.")
    if os.path.exists(INITIALLY_PREPROCESSED_MARKET_DATA):
        df = pd.read_csv(INITIALLY_PREPROCESSED_MARKET_DATA)
    else:
        print("Preparing the initially preprocessed dataset...")
        initial_preprocessing()
        print("Done.")
        df = pd.read_csv(INITIALLY_PREPROCESSED_MARKET_DATA)
    df["Datetime from"] = pd.to_datetime(df["Datetime from"])
    df["Datetime offer time"] = pd.to_datetime(df["Datetime offer time"])
    df = df[(df["Datetime from"] >= start) & ((df["Datetime from"]) < end)]

    print("Done reading the initially preprocessed dataset.")

    print("Preprocessing the data...")
    index = np.arange(1, demanded_len + 1)
    index_daily = np.arange(1, demanded_len + 1)
    for d, date in enumerate(np.unique(df["Datetime from"].dt.date)):
        preprocessed_data = pd.DataFrame(index=index)  # daily df to sqlite update
        index = index + d * demanded_len  # update the index

        df_day = df[
            df["Datetime from"].dt.date == date
        ]  # frame with all trades for day d
        unique_datetime_from = np.unique(df_day["Datetime from"])

        demanded_delta = 15
        if len(unique_datetime_from) != deliveries_no:
            print(
                f"{date} WARNING: trajectories for {deliveries_no - len(unique_datetime_from)} deliveries will be averaged using steps back and forward corresponding to no. of missing deliveries."
            )
        shift = 0  # shift the index in case a hole (distance greater than 15min) is detected in unique deliveries
        stable_shift = 0
        first_deliveries_unavail = False
        missing_windows = {}
        last_delivery = pd.to_datetime(
            np.sort(unique_datetime_from)[0]
        ).replace(  # initialize the delivery time
            minute=0
        )

        for delivery_idx, delivery in enumerate(np.sort(unique_datetime_from)):
            if (
                pd.to_datetime(np.sort(unique_datetime_from)[0]).replace(minute=0)
                != last_delivery
                and (
                    pd.to_datetime(delivery) - pd.to_datetime(last_delivery)
                ).total_seconds()
                / 60
                != demanded_delta
            ):  # check whether we are not missing any deliveries in between - and if we are -
                shift = (
                    int(
                        (
                            (
                                pd.to_datetime(delivery) - pd.to_datetime(last_delivery)
                            ).total_seconds()
                            / 60
                        )
                        // demanded_delta
                    )
                    - 1  # - 1 because we calculate from the end of the last available period but last_delivery is its start
                )
                missing_windows[delivery_idx] = (
                    shift  # save the no. of missing deliveries
                )
            elif (
                delivery_idx == 0
                and pd.to_datetime(np.sort(unique_datetime_from)[0]).replace(minute=0)
                == last_delivery
                and (
                    pd.to_datetime(delivery) - pd.to_datetime(last_delivery)
                ).total_seconds()
                / 60
                != 0
            ):
                shift = (
                    int(
                        (
                            (
                                pd.to_datetime(delivery) - pd.to_datetime(last_delivery)
                            ).total_seconds()
                            / 60
                        )
                        // demanded_delta
                    )
                    - 1
                )
                missing_windows[delivery_idx] = shift
                first_deliveries_unavail = True

            last_delivery = delivery
            delivery_idx = delivery_idx + shift + stable_shift  # perform the shifting
            stable_shift += shift  # all consecutive indices are shifted by stable shift
            shift = 0  # reset the shift

            df_day_delivery = df_day[df_day["Datetime from"] == delivery]
            current_data_avg = df_day_delivery.groupby(
                "Time to delivery", as_index=False
            )
            # PRICE PREPROCESSING
            price = []
            time_to_delivery = []
            # first known indicator of quarterhourly price is day-ahead auction trade
            for group in current_data_avg:
                price.append(
                    np.sum(
                        group[1]["Price (EUR)"].to_numpy()
                        * group[1]["Volume (MW)"].to_numpy()
                    )
                    / np.sum(group[1]["Volume (MW)"])
                )
                time_to_delivery.append(group[1]["Time to delivery"].to_numpy()[0])
            time_to_delivery = time_to_delivery[::-1]
            price = price[::-1]
            trading_start = (
                (  # anticipated start of trading
                    pd.to_datetime(delivery)
                    - (pd.to_datetime(delivery) - timedelta(days=1))
                    .replace(hour=trading_start_hour)
                    .replace(minute=0)
                ).total_seconds()
                / SECONDS_TO_5MIN_SCALER
            )  # we compute trading start as minutes to delivery
            if (
                trading_start > np.max(time_to_delivery)
            ):  # if trading did not start exactly at opening, we extend our trajectory with the price from ID auction from the left
                price = [
                    float(ID_qtrly[ID_qtrly.index == delivery]["price"].to_numpy()[0])
                ] + price
                time_to_delivery = [trading_start] + time_to_delivery
            end = 0
            ttd = time_to_delivery + [end]
            add_nos = -np.diff(ttd)
            prices = [
                ele for i, ele in enumerate(price) for j in range(int(add_nos[i]))
            ]  # filling the missing minutes with 180min price avg
            if len(prices) < demanded_len:
                prices = np.hstack(
                    (
                        prices,
                        np.ones(demanded_len - len(prices)) * np.mean(prices[-36:]),
                    )
                )  # approximating ID3 - maybe putting ID3 here would be better; but it is not taken into consideration either way in the simulation
            elif len(prices) > demanded_len:
                prices = prices[:demanded_len]

            preprocessed_data[delivery_idx] = prices[:demanded_len]
            # VOLUME PREPROCESSING
            volume = []
            time_to_delivery = []
            for group in current_data_avg:
                volume.append(np.sum(group[1]["Volume (MW)"]))
                time_to_delivery.append(group[1]["Time to delivery"].to_numpy()[0])
            time_to_delivery = time_to_delivery[::-1]
            volume = volume[::-1]
            trading_start = (
                pd.to_datetime(delivery)
                - (pd.to_datetime(delivery) - timedelta(days=1))
                .replace(hour=trading_start_hour)
                .replace(minute=0)
            ).total_seconds() / SECONDS_TO_5MIN_SCALER
            if trading_start > np.max(time_to_delivery):
                volume = [0] + volume
                time_to_delivery = [trading_start] + time_to_delivery
            end = 0
            ttd = time_to_delivery + [end]
            add_nos = -np.diff(ttd)
            volumes = [
                0 if j > 0 else ele
                for i, ele in enumerate(volume)
                for j in range(int(add_nos[i]))
            ]  # filling the missing minutes
            if len(volumes) < demanded_len:
                volumes = np.hstack(
                    (volumes, np.zeros(demanded_len - len(volumes)))
                )  # fill with total traded volume
            elif len(volumes) > demanded_len:
                volumes = volumes[:demanded_len]

            preprocessed_data[delivery_idx + deliveries_no] = volumes[:demanded_len]

            # INDICATORS OF TRADE EXISTENCE IN PERIOD
            trade_indicator = []
            time_to_delivery = []
            for group in current_data_avg:
                trade_indicator.append(1)
                time_to_delivery.append(group[1]["Time to delivery"].to_numpy()[0])
            time_to_delivery = time_to_delivery[::-1]
            trade_indicator = trade_indicator[::-1]
            trading_start = (
                pd.to_datetime(delivery)
                - (pd.to_datetime(delivery) - timedelta(days=1))
                .replace(hour=trading_start_hour)
                .replace(minute=0)
            ).total_seconds() / SECONDS_TO_5MIN_SCALER
            if trading_start > np.max(time_to_delivery):
                trade_indicator = [0] + trade_indicator
                time_to_delivery = [trading_start] + time_to_delivery
            end = 0
            ttd = time_to_delivery + [end]
            add_nos = -np.diff(ttd)
            trade_indicators = [
                0 if j > 0 else ele
                for i, ele in enumerate(trade_indicator)
                for j in range(int(add_nos[i]))
            ]  # filling the missing minutes
            if len(trade_indicators) < demanded_len:
                trade_indicators = np.hstack(
                    (trade_indicators, np.zeros(demanded_len - len(trade_indicators)))
                )  # fill with 0s if no trades recorded
            elif len(trade_indicators) > demanded_len:
                trade_indicators = trade_indicators[:demanded_len]

            preprocessed_data[delivery_idx + 2 * deliveries_no] = trade_indicators[
                :demanded_len
            ]

        preprocessed_data = preprocessed_data.reindex(
            sorted(preprocessed_data.columns), axis=1
        )
        for corr_idx in missing_windows.keys():
            if first_deliveries_unavail and corr_idx == 0:
                for col_n in range(corr_idx, corr_idx + missing_windows[corr_idx]):
                    preprocessed_data[col_n] = preprocessed_data[
                        col_n + missing_windows[corr_idx]
                    ]
                    preprocessed_data[col_n + deliveries_no] = preprocessed_data[
                        col_n + missing_windows[corr_idx] + deliveries_no
                    ]
                    preprocessed_data[col_n + 2 * deliveries_no] = preprocessed_data[
                        col_n + missing_windows[corr_idx] + 2 * deliveries_no
                    ]
            else:
                for col_n in range(corr_idx, corr_idx + missing_windows[corr_idx]):
                    preprocessed_data[col_n] = (
                        preprocessed_data[col_n - missing_windows[corr_idx]]
                        + preprocessed_data[col_n + missing_windows[corr_idx]]
                    ) / 2
                    preprocessed_data[col_n + deliveries_no] = (
                        preprocessed_data[
                            col_n - missing_windows[corr_idx] + deliveries_no
                        ]
                        + preprocessed_data[
                            col_n + missing_windows[corr_idx] + deliveries_no
                        ]
                    ) / 2
                    preprocessed_data[col_n + 2 * deliveries_no] = (
                        preprocessed_data[
                            col_n - missing_windows[corr_idx] + 2 * deliveries_no
                        ]
                        + preprocessed_data[
                            col_n + missing_windows[corr_idx] + 2 * deliveries_no
                        ]
                    ) / 2

        # ADD WEEKDAY DUMMIES
        preprocessed_data[3 * deliveries_no] = np.ones(demanded_len) * date.weekday()

        preprocessed_data["Time"] = pd.date_range(
            (pd.to_datetime(date) - timedelta(days=1))
            .replace(hour=trading_start_hour)
            .replace(minute=0),
            pd.to_datetime(date) + timedelta(days=1),
            freq="5min",
            inclusive="left",
        )
        preprocessed_data["Index_daily"] = index_daily
        preprocessed_data["Day"] = np.ones(np.shape(index_daily)) * d
        preprocessed_data.to_sql("with_dummies", con, if_exists="append")

        print(f"Done {d} of {len(np.unique(df['Datetime from'].dt.date))}")


if __name__ == "__main__":
    if not os.path.exists(MARKET_DATA_DIR):
        con = sqlite3.connect(MARKET_DATA_DIR)
    else:
        os.remove(MARKET_DATA_DIR)
        con = sqlite3.connect(MARKET_DATA_DIR)

    ID_qtrly = pd.read_csv(
        os.path.join(
            DATA_DIR,
            "ID_auction_preprocessed",
            "ID_auction_price_2018-2020_preproc.csv",
        ),
        index_col=0,
        parse_dates=True,
    )

    preprocess_data(
        required_start,
        required_end,
        ID_qtrly,
        True,
    )
    con.close()
