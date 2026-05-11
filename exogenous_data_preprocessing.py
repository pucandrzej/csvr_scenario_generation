"""Verbose, step-by-step exogenous variable preprocessing for the cSVR probabilistic forecasting simulation study.
The script transforms raw ENTSO-E yearly csvs that can be easily downloaded manually into pandas dataframes covering the whole required period.
This way we avoid the usage of ENTSO-E API for this simple backtest.
"""

import os
import pandas as pd
from datetime import timedelta
from config.paths import DATA_DIR

from config.test_calibration_validation import (
    required_start,
    required_end,
    currency_change_date_PL,
)
from utils import fill_march_dst, check_for_missing_data, parse_mtu_index

VERBOSE_TIMESTAMP_FORMAT_EXCEPTIONS = False

BORDER_CTYS = [
    "DE",
    "AT",
    "BE",
    "CZ",
    "CH",
    "DK1",
    "DK2",
    "FR",
    "NL",
    "NO2",
    "PL",
    "SE4",
]  # DE and all the borders of DE

START_YEAR = 2018
END_YEAR = 2021  # 2020+1 for range() in python

###################################################################################
print("Processing the hourly border and DE d-a price data...")

all_ctys_dfs = []

for cty in BORDER_CTYS:
    da_price = []
    for year in range(START_YEAR, END_YEAR):
        df = pd.read_csv(
            os.path.join(DATA_DIR, "Day-Ahead-Quarterly-Data", f"DA_{cty}_{year}.csv"),
            na_values=["n/e"],
            index_col=0,
        )
        da_column = [c for c in df.columns if "Day-ahead" in c][0]
        if df.index.name != "MTU (CET/CEST)":
            raise ValueError(f"Unexpected index name {df.index.name}")
        df.rename(columns={da_column: cty}, inplace=True)
        df = df[[cty]]

        df.index = parse_mtu_index(
            df.index, verbose=VERBOSE_TIMESTAMP_FORMAT_EXCEPTIONS
        )

        da_price.append(df[[cty]])

    da_price_all_years = pd.concat(da_price, axis=0)
    da_price_all_years_dst_corrected = fill_march_dst(
        da_price_all_years[~da_price_all_years.index.duplicated()], col=cty
    )

    # for PL we need to change the currency to EUR for data till 2019.11.19 (inclusively)
    if cty == "PL":
        pln_eur = pd.read_csv(
            os.path.join(DATA_DIR, "Day-Ahead-Quarterly-Data", "PLN_EUR_2018_2019.csv"),
            index_col=0,
        )["Price"]
        pln_eur.index = pd.to_datetime(pln_eur.index, format="%d-%m-%Y")
        pln_eur_resampled = pln_eur.sort_index().resample("1h").ffill()
        da_price_all_years_dst_corrected.loc[
            da_price_all_years_dst_corrected.index < currency_change_date_PL, "PL"
        ] = (
            da_price_all_years_dst_corrected.loc[
                da_price_all_years_dst_corrected.index < currency_change_date_PL, "PL"
            ]
            * pln_eur_resampled.loc[pln_eur_resampled.index < currency_change_date_PL]
        )

    if cty != "DE":
        da_price_all_years_dst_corrected[f"DE-{cty}"] = (
            da_price_all_years_dst_corrected[cty] - all_ctys_dfs[0]["DE"]
        )

    all_ctys_dfs.append(da_price_all_years_dst_corrected)

all_de_border_prices = pd.concat(all_ctys_dfs, axis=1)

all_de_border_prices.to_csv(
    os.path.join(
        DATA_DIR,
        "Day-Ahead-Quarterly-Data",
        "DE_and_all_DE_borders_hourly_day_ahead_prices.csv",
    )
)

check_for_missing_data(all_de_border_prices, required_start, required_end, freq="1h")

if (
    len(all_de_border_prices[all_de_border_prices.index >= required_start])
    != len(pd.date_range(required_start, required_end, inclusive="left", freq="1h"))
    or all_de_border_prices[all_de_border_prices.index >= required_start]
    .isnull()
    .values.any()
):
    raise

###################################################################################
print("Processing the q-hrly DE day-ahead prices...")

cty = "DE"
da_price = []
for year in range(START_YEAR, END_YEAR):
    df = pd.read_csv(
        os.path.join(DATA_DIR, "Day-Ahead-Quarterly-Data", f"DA_{cty}_Q_{year}.csv"),
        na_values=["n/e"],
        index_col=0,
    )
    da_column = [c for c in df.columns if "Day-ahead" in c][0]
    if df.index.name != "MTU (CET/CEST)":
        raise ValueError(f"Unexpected index name {df.index.name}")
    df.rename(columns={da_column: cty}, inplace=True)
    df = df[[cty]]

    df.index = parse_mtu_index(df.index, verbose=VERBOSE_TIMESTAMP_FORMAT_EXCEPTIONS)

    da_price.append(df[[cty]])

da_price_all_years = pd.concat(da_price, axis=0)
da_price_all_years_dst_corrected = fill_march_dst(
    da_price_all_years[~da_price_all_years.index.duplicated()], col=cty
)

da_price_all_years_dst_corrected.to_csv(
    os.path.join(
        DATA_DIR,
        "Day-Ahead-Quarterly-Data",
        f"{cty}_quarterhourly_day_ahead_prices.csv",
    )
)

check_for_missing_data(
    da_price_all_years_dst_corrected, required_start, required_end, freq="15min"
)

###################################################################################
print("Processing the intraday auction data...")

dfs = []
for year in range(START_YEAR, END_YEAR):
    file_path = os.path.join(
        DATA_DIR,
        "ID_auction_preprocessed",
        f"intraday_auction_spot_prices_15-call-DE_{year}.csv",
    )

    df = pd.read_csv(
        file_path,
        skiprows=1,
        index_col=0,
        parse_dates=True,
        dayfirst=True,
    )

    dfs.append(df)
total_df = pd.concat(dfs).iloc[:, :-8]

# drop the B set of DST deliveries
total_df = total_df[[c for c in total_df.columns if "Hour 3B" not in c]]
new_df = []
for idx in total_df.index:
    new_df.append(
        pd.DataFrame(
            columns=["price"],
            index=pd.date_range(idx, idx + timedelta(minutes=95 * 15), freq="15min"),
            data=total_df.loc[idx].to_numpy(),
        )
    )
df = fill_march_dst(pd.concat(new_df).sort_index(), col="price")
df.to_csv(
    os.path.join(
        DATA_DIR, "ID_auction_preprocessed", "ID_auction_price_2018-2020_preproc.csv"
    )
)

check_for_missing_data(df, required_start, required_end, freq="15min")

###################################################################################
print("Processing the hourly physical exchange data...")

all_ctys_exchange = []
for cty in BORDER_CTYS:
    if cty == "DE":
        continue

    exchange = []
    for year in range(START_YEAR, END_YEAR):
        df = pd.read_csv(
            os.path.join(
                DATA_DIR, "Crossborder", f"crossborder_de_{cty.lower()}_{year}.csv"
            ),
            na_values=["n/e"],
            index_col=0,
        )

        if df.index.name != "Time (CET/CEST)":
            raise ValueError(f"Unexpected index name {df.index.name}")
        de_export_column = [c for c in df.columns if "BZN|DE-LU > " in c][0]
        de_import_column = [c for c in df.columns if "> BZN|DE-LU" in c][0]
        df.rename(
            columns={
                de_export_column: f"{cty}_import",
                de_import_column: f"{cty}_export",
            },
            inplace=True,
        )

        df = (df[f"{cty}_import"] - df[f"{cty}_export"]).to_frame(name=cty)

        df.index = parse_mtu_index(
            df.index, verbose=VERBOSE_TIMESTAMP_FORMAT_EXCEPTIONS
        )

        exchange.append(df)

    exchange_all_years = pd.concat(exchange, axis=0)
    exchange_all_years_dst_corrected = fill_march_dst(
        exchange_all_years[~exchange_all_years.index.duplicated()], col=cty
    )

    exchange_all_years_dst_corrected_resampled = (
        exchange_all_years_dst_corrected.resample("1h").mean()
    )

    all_ctys_exchange.append(exchange_all_years_dst_corrected_resampled)

all_de_border_exchanges = pd.concat(all_ctys_exchange, axis=1)

all_de_border_exchanges = all_de_border_exchanges.drop(columns="NO2")

all_de_border_exchanges.to_csv(os.path.join(DATA_DIR, "Crossborder", "DE_physexc.csv"))

check_for_missing_data(all_de_border_exchanges, required_start, required_end, freq="1h")

###################################################################################
print("Processing the hourly commercial scheduled exchange data...")

all_ctys_exchange = []
for cty in BORDER_CTYS:
    if cty in ["DE", "BE", "NO2"]:
        continue

    exchange = []
    for year in range(START_YEAR, END_YEAR):
        df = pd.read_csv(
            os.path.join(
                DATA_DIR, "Crossborder", f"commex_de_{cty.lower()}_{year}.csv"
            ),
            na_values=["n/e"],
            index_col=0,
        )

        if df.index.name != "Time (CET/CEST)":
            raise ValueError(f"Unexpected index name {df.index.name}")
        de_export_column = [c for c in df.columns if "BZN|DE-LU > " in c][0]
        de_import_column = [c for c in df.columns if "> BZN|DE-LU" in c][0]
        df.rename(
            columns={
                de_export_column: f"{cty}_import",
                de_import_column: f"{cty}_export",
            },
            inplace=True,
        )

        df = (df[f"{cty}_import"] - df[f"{cty}_export"]).to_frame(
            name=cty
        )  # sum up the import and export in each delivery to get the net value
        df.index = parse_mtu_index(
            df.index, verbose=VERBOSE_TIMESTAMP_FORMAT_EXCEPTIONS
        )

        exchange.append(df)

    exchange_all_years = pd.concat(exchange, axis=0)
    exchange_all_years_dst_corrected = fill_march_dst(
        exchange_all_years[~exchange_all_years.index.duplicated()], col=cty
    )

    exchange_all_years_dst_corrected_resampled = (
        exchange_all_years_dst_corrected.resample("1h").mean()
    )

    all_ctys_exchange.append(exchange_all_years_dst_corrected_resampled)

all_de_border_exchanges = pd.concat(all_ctys_exchange, axis=1)

all_de_border_exchanges.to_csv(os.path.join(DATA_DIR, "Crossborder", "DE_commex.csv"))

check_for_missing_data(all_de_border_exchanges, required_start, required_end, freq="1h")

###################################################################################
print("Processing the load data...")
load = []
for year in range(START_YEAR, END_YEAR):
    load.append(
        pd.read_csv(
            os.path.join(
                DATA_DIR,
                "Load",
                f"Total Load - Day Ahead _ Actual_{year}01010000-{year + 1}01010000.csv",
            ),
            na_values=["n/e"],
        )
    )

load = pd.concat(load, ignore_index=True)

new_datetimes = []
for dat in load["Time (CET/CEST)"]:
    new_datetimes.append(dat.split(" - ")[0])

load_df = pd.DataFrame()
load_df["Time from"] = pd.to_datetime(
    new_datetimes, format="%d.%m.%Y %H:%M", dayfirst=False, yearfirst=False
)
load_df["Actual"] = load["Actual Total Load [MW] - BZN|DE-LU"]
load_df["Forecast"] = load["Day-ahead Total Load Forecast [MW] - BZN|DE-LU"]
load_df.index = pd.to_datetime(load_df["Time from"])
load_df.drop(columns="Time from", inplace=True)

# remove October dst change impact
load_df = fill_march_dst(load_df[~load_df.index.duplicated()], col="Actual")

load_df.to_csv(os.path.join(DATA_DIR, "Load", "Load_2018-2020.csv"))

check_for_missing_data(load_df, required_start, required_end, freq="15min")

###################################################################################
print("Processing the generation data...")
gen = []
for year in range(START_YEAR, END_YEAR):
    gen.append(
        pd.read_csv(
            os.path.join(DATA_DIR, "Generation", f"generation_{year}.csv"),
            na_values=["n/e"],
        )
    )

gen = pd.concat(gen, ignore_index=True)[
    [
        "MTU",
        "Solar  - Actual Aggregated [MW]",
        "Wind Offshore  - Actual Aggregated [MW]",
        "Wind Onshore  - Actual Aggregated [MW]",
    ]
]

new_datetimes = []
for dat in gen["MTU"]:
    new_datetimes.append(dat.split(" - ")[0])

gen_df = pd.DataFrame()
gen_df["Time from"] = pd.to_datetime(
    new_datetimes, format="%d.%m.%Y %H:%M", dayfirst=False, yearfirst=False
)
gen_df["SPV"] = gen["Solar  - Actual Aggregated [MW]"]
gen_df["W"] = (
    gen["Wind Offshore  - Actual Aggregated [MW]"]
    + gen["Wind Onshore  - Actual Aggregated [MW]"]
)
gen_df.index = pd.to_datetime(gen_df["Time from"])
gen_df.drop(columns="Time from", inplace=True)

gen_fore = []
for year in range(START_YEAR, END_YEAR):
    gen_fore.append(
        pd.read_csv(
            os.path.join(DATA_DIR, "Generation", f"generation_fore_{year}.csv"),
            na_values=["n/e"],
        )
    )

gen_fore = pd.concat(gen_fore, ignore_index=True)[
    [
        "MTU (CET/CEST)",
        "Generation - Solar  [MW] Day Ahead/ BZN|DE-LU",
        "Generation - Wind Offshore  [MW] Day Ahead/ BZN|DE-LU",
        "Generation - Wind Onshore  [MW] Day Ahead/ BZN|DE-LU",
        "Generation - Solar [MW] Intraday / BZN|DE-LU",
        "Generation - Wind Offshore [MW] Intraday / BZN|DE-LU",
        "Generation - Wind Onshore [MW] Intraday / BZN|DE-LU",
    ]
]

new_datetimes = []
for dat in gen_fore["MTU (CET/CEST)"]:
    new_datetimes.append(dat.split(" - ")[0])

gen_fore_df = pd.DataFrame()
gen_fore_df["Time from"] = pd.to_datetime(
    new_datetimes, format="%d.%m.%Y %H:%M", dayfirst=False, yearfirst=False
)
gen_fore_df["SPV DA"] = gen_fore["Generation - Solar  [MW] Day Ahead/ BZN|DE-LU"]
gen_fore_df["W DA"] = (
    gen_fore["Generation - Wind Offshore  [MW] Day Ahead/ BZN|DE-LU"]
    + gen_fore["Generation - Wind Onshore  [MW] Day Ahead/ BZN|DE-LU"]
)
gen_fore_df["SPV ID"] = gen_fore["Generation - Solar [MW] Intraday / BZN|DE-LU"]
gen_fore_df["W ID"] = (
    gen_fore["Generation - Wind Offshore [MW] Intraday / BZN|DE-LU"]
    + gen_fore["Generation - Wind Onshore [MW] Intraday / BZN|DE-LU"]
)
gen_fore_df.index = pd.to_datetime(gen_fore_df["Time from"])
gen_fore_df.drop(columns="Time from", inplace=True)

gen_df = pd.concat([gen_df, gen_fore_df], axis=1)

gen_df = fill_march_dst(gen_df[~gen_df.index.duplicated()], col="SPV")

check_for_missing_data(gen_df, required_start, required_end, freq="15min")

gen_df.to_csv(os.path.join(DATA_DIR, "Generation", "Generation_2018-2020.csv"))
