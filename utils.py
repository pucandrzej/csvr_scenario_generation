import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config.test_calibration_validation import (
    required_start,
)


def parse_mtu_index(index, verbose=False):
    """General processing of ENTSO-E index from string to datetime"""
    datetimes = [dat.split(" - ")[0] for dat in index]

    try:
        return pd.to_datetime(
            datetimes,
            format="%d/%m/%Y %H:%M:%S",
            dayfirst=False,
            yearfirst=False,
        )
    except Exception as err:
        if verbose:
            print(
                f"Exception: {err}. Failed to format the index as datetime using %d/%m/%Y %H:%M:%S;\ntrying %d.%m.%Y %H:%M instead."
            )
        return pd.to_datetime(
            datetimes, format="%d.%m.%Y %H:%M", dayfirst=False, yearfirst=False
        )


def fill_march_dst(df, col):
    """get the dst gaps"""
    dst_gaps = df[(df.index >= required_start) & df[col].isna()].index
    print(f"DST GAPS: {dst_gaps} {col}")

    # Fill missing values using average of Hour 2 and Hour 3 (before and after gap)
    for ts in dst_gaps:
        before = ts - pd.Timedelta(hours=1)
        after = ts + pd.Timedelta(hours=1)
        if before in df.index and after in df.index:
            df = df.copy(deep=True)
            df.loc[ts] = (df.loc[before] + df.loc[after]) / 2
    return df


def fill_march_dst_daily(df, col):
    """get the dst gaps in daily granularity series (corresponding to one specific delivery from each day)"""
    dst_gaps = df[(df.index >= required_start) & df[col].isna()].index
    print(f"DST GAPS: {dst_gaps} {col}")

    # Fill missing values using average of Hour 2 and Hour 3 (before and after gap)
    for ts in dst_gaps:
        before = ts - pd.Timedelta(days=1)
        after = ts + pd.Timedelta(days=1)
        if before in df.index and after in df.index:
            df = df.copy(deep=True)
            df.loc[ts] = (df.loc[before] + df.loc[after]) / 2
    return df


def check_for_missing_data(df, required_start, required_end, freq):
    matching_len = len(df[df.index >= required_start]) == len(
        pd.date_range(required_start, required_end, inclusive="left", freq=freq)
    )
    any_nans = df[df.index >= required_start].isnull().values.any()
    if not matching_len or any_nans:
        raise ValueError("Missing data found!")


def price2vol_sup(P, df_sup_price):
    return np.interp(P, df_sup_price["Price"], df_sup_price["Volume"])


def price2vol_dem(P, df_dem_price):
    return np.interp(P, df_dem_price["Price"], df_dem_price["Volume"])


def S_trans(P, df_sup_price, DEM_inelastic, df_dem_price):
    """Defines transformed supply, returning volume"""
    return (
        price2vol_sup(P, df_sup_price.copy())
        + DEM_inelastic
        - price2vol_dem(P, df_dem_price.copy())
    )


def sup_trans_inv(z, V_sorted, P_sorted):
    return np.interp(z, V_sorted, P_sorted)


def devel_elasticities_plot(
    df_sup_price,
    df_dem_price,
    P_auction,
    P_minus,
    P_clearing,
    P_plus,
    q0,
    volume_delta,
    V_sorted,
    P_sorted,
    q_implied,
    DEM_inelastic,
):
    """Plots the elasticities derivation illustration - usable both for better understanding of the process and its debugging"""
    # Create subplot figure
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=[
            "(a) Original curves",
            "(b) Transformed curves",
            "(c) Slope coefficient",
        ],
    )

    # -----------------------------------------------------------
    # Original inverse curves
    # -----------------------------------------------------------
    fig.add_trace(
        go.Scatter(
            x=df_sup_price["Volume"],
            y=df_sup_price["Price"],
            mode="lines",
            name="SUP_WS",
            line=dict(color="blue"),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=df_dem_price["Volume"],
            y=df_dem_price["Price"],
            mode="lines",
            name="DEM_WS",
            line=dict(color="red"),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=[min(df_sup_price["Volume"]), max(df_sup_price["Volume"])],
            y=[P_auction, P_auction],
            mode="lines",
            line=dict(color="gray", dash="dash"),
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=[q0],
            y=[P_auction],
            mode="markers",
            marker=dict(color="black", size=8),
            name="Auction clearing point",
        ),
        row=1,
        col=1,
    )

    fig.update_xaxes(title="Volume [MWh]", row=1, col=1)
    fig.update_yaxes(title="Price [€]", row=1, col=1)

    # -----------------------------------------------------------
    # Transformed supply + inelastic demand
    # -----------------------------------------------------------
    fig.add_trace(
        go.Scatter(
            x=V_sorted,
            y=P_sorted,
            mode="lines",
            name="Transformed supply",
            line=dict(color="blue"),
        ),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Scatter(
            x=[DEM_inelastic, DEM_inelastic],
            y=[min(P_sorted), max(P_sorted)],
            mode="lines",
            line=dict(color="red", dash="dash"),
            name="DEM_inelastic",
        ),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Scatter(
            x=[min(V_sorted), max(V_sorted)],
            y=[P_auction, P_auction],
            mode="lines",
            line=dict(color="gray", dash="dash"),
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    fig.update_xaxes(title="Volume [MWh]", row=1, col=2)
    fig.update_yaxes(title="Price [€]", row=1, col=2)

    # -----------------------------------------------------------
    # Slope segment around q0
    # -----------------------------------------------------------
    fig.add_trace(
        go.Scatter(
            x=V_sorted,
            y=P_sorted,
            mode="lines",
            name="Transformed supply",
            line=dict(color="blue"),
        ),
        row=1,
        col=3,
    )

    fig.add_trace(
        go.Scatter(
            x=[q_implied - volume_delta, q_implied + volume_delta],
            y=[P_minus, P_plus],
            mode="lines",
            name=f"Slope segment Δ={volume_delta}",
            line=dict(color="green"),
        ),
        row=1,
        col=3,
    )

    fig.add_trace(
        go.Scatter(
            x=[q_implied - volume_delta, q_implied, q_implied + volume_delta],
            y=[P_minus, P_clearing, P_plus],
            mode="markers",
            marker=dict(color="black", size=8),
            name="Slope points",
        ),
        row=1,
        col=3,
    )

    fig.add_trace(
        go.Scatter(
            x=[min(V_sorted), max(V_sorted)],
            y=[P_auction, P_auction],
            mode="lines",
            line=dict(color="gray", dash="dash"),
            showlegend=False,
        ),
        row=1,
        col=3,
    )

    fig.add_trace(
        go.Scatter(
            x=[q_implied, q_implied],
            y=[min(P_sorted), max(P_sorted)],
            mode="lines",
            line=dict(color="gray", dash="dash"),
            showlegend=False,
        ),
        row=1,
        col=3,
    )

    fig.add_trace(
        go.Scatter(
            x=[DEM_inelastic, DEM_inelastic],
            y=[min(P_sorted), max(P_sorted)],
            mode="lines",
            line=dict(color="red", dash="dash"),
            name="DEM_inelastic",
        ),
        row=1,
        col=3,
    )

    fig.update_xaxes(title="Volume [MWh]", row=1, col=3)
    fig.update_yaxes(title="Price [€]", row=1, col=3)

    # -----------------------------------------------------------
    # Layout
    # -----------------------------------------------------------
    fig.update_layout(showlegend=True, title="Supply/Demand Curve Transformations")

    fig.show()
