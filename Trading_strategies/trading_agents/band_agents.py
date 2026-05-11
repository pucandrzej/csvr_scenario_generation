import numpy as np
import plotly.graph_objects as go
from dataclasses import dataclass

from Trading_strategies.strategies_utils import (
    compute_weights,
    vanilla_band,
    weighted_band,
    add_curve,
    get_trust_threshold,
)


@dataclass(kw_only=True)
class BaseBandConfig:
    band_type: str
    scp: float
    p: float
    lambda_: float = 0.25
    trust_threshold_method: str = "3sigma"
    weights_method: str = "kernel"
    dev_plots: bool = False


@dataclass(kw_only=True)
class OneSidedBandsConfig(BaseBandConfig):
    direction: int


@dataclass(kw_only=True)
class TwoSidedBandsConfig(BaseBandConfig):
    pass


def one_sided_bands_strategy(
    y_actual,
    y_forecast,
    config: OneSidedBandsConfig,
):
    direction = config.direction
    band_type = config.band_type
    scp = config.scp
    p = config.p
    lambda_ = config.lambda_
    trust_threshold_method = config.trust_threshold_method
    weights_method = config.weights_method
    dev_plots = config.dev_plots

    if dev_plots:
        fig = go.Figure()
        x = np.arange(len(y_actual))

    T, N = y_forecast.shape

    if direction == -1:
        if band_type == "risk_seeking":
            band = vanilla_band(
                y_forecast, scp=scp, band_type="upper"
            )  # taking max of upper band maximizing the max expected price
        elif band_type == "risk_averse":
            band = vanilla_band(
                y_forecast, scp=scp, band_type="lower"
            )  # taking max of lower band maximizing the min expected price
    elif direction == 1:
        if band_type == "risk_seeking":
            band = vanilla_band(
                y_forecast, scp=scp, band_type="lower"
            )  # taking min of upper band minimizing the min expected price
        elif band_type == "risk_averse":
            band = vanilla_band(
                y_forecast, scp=scp, band_type="upper"
            )  # taking min of upper band minimizing the max expected price

    # initial plan of trading
    argmax = int(np.argmax(band))
    argmin = int(np.argmin(band))

    if direction == 1:  # we buy - we want to buy at min price possible
        planned_entry = argmin
    elif direction == -1:  # we sell - we want to sell at max price possible
        planned_entry = argmax

    initial_planned_entry = planned_entry

    basic_profit = y_actual[initial_planned_entry]
    if direction == 1:
        best_profit = np.min(y_actual)
        worst_profit = np.max(y_actual)
    elif direction == -1:
        best_profit = np.max(y_actual)
        worst_profit = np.min(y_actual)

    played = False

    profit = 0  # profit declaration

    # iterate over t=0..T-1 and adapt plan if a more profitable buy/sell points are detected
    for t in range(T):
        # compute weights using data observed up to t (inclusive)
        price_so_far = y_actual[: t + 1]
        forecast_so_far = y_forecast[: t + 1, :]

        residuals = (
            np.median(forecast_so_far, axis=1) - price_so_far
        )  # calc errors between median of trajectories and price observed so far
        trust_threshold, nonzero_mae = get_trust_threshold(
            residuals, trust_threshold_method
        )

        w = compute_weights(
            forecast_so_far,
            price_so_far,
            nonzero_mae,
            p,
            lambda_,
            weights_method,
        )

        # build conditional medians for future times > t
        future_count = T - (t + 1)
        if future_count <= 0:
            break

        if direction == -1:
            if band_type == "risk_seeking":
                cond_band = weighted_band(y_forecast[t + 1 :, :], w, scp, "upper")
            elif band_type == "risk_averse":
                cond_band = weighted_band(y_forecast[t + 1 :, :], w, scp, "lower")
        elif direction == 1:
            if band_type == "risk_seeking":
                cond_band = weighted_band(y_forecast[t + 1 :, :], w, scp, "lower")
            elif band_type == "risk_averse":
                cond_band = weighted_band(y_forecast[t + 1 :, :], w, scp, "upper")

        if dev_plots:
            if t == 0:
                for fore_idx in range(np.shape(y_forecast)[1]):
                    add_curve(
                        fig,
                        x,
                        y_forecast[:, fore_idx],
                        f"{fore_idx} forecast path",
                        "grey",
                    )
                add_curve(fig, x, band, "Band", "blue")
                add_curve(fig, x, y_actual, "Actual", "green")
            add_curve(fig, x[T - len(cond_band) :], cond_band, f"Band {t}", "red")

        # map back to absolute indices
        rel_argmax = int(np.argmax(cond_band))
        rel_argmin = int(np.argmin(cond_band))
        new_argmax = rel_argmax + (t + 1)
        new_argmin = rel_argmin + (t + 1)

        if direction == 1:
            desired_entry = new_argmin
        elif direction == -1:
            desired_entry = new_argmax

        if planned_entry > t:
            planned_entry_profit = cond_band[planned_entry - t - 1]
        elif planned_entry == t:
            planned_entry_profit = y_actual[planned_entry]

        if planned_entry > t:
            desired_entry_profit = cond_band[desired_entry - t - 1]
        elif planned_entry == t:
            desired_entry_profit = y_actual[desired_entry]

        # we shift the entering of position if we see more profit from changing it
        if desired_entry_profit - trust_threshold > planned_entry_profit:
            planned_entry = desired_entry

        # entry logic: if not in position and planned entry is now -> enter
        if planned_entry == t:
            played = True
            profit = y_actual[planned_entry]
            break

    # force an action at the end of the path if action was not performed in course of the path
    if not played:
        played = True
        profit = y_actual[-1]

    return profit, basic_profit, played, best_profit, worst_profit


def two_sided_bands_strategy(y_actual, y_forecast, config: TwoSidedBandsConfig):
    band_type = config.band_type
    scp = config.scp
    p = config.p
    lambda_ = config.lambda_
    trust_threshold_method = config.trust_threshold_method
    weights_method = config.weights_method
    dev_plots = config.dev_plots

    if dev_plots:
        fig = go.Figure()
        x = np.arange(len(y_actual))

    T, N = y_forecast.shape

    if band_type == "risk_seeking":
        max_band = vanilla_band(y_forecast, scp=scp, band_type="upper")
        min_band = vanilla_band(y_forecast, scp=scp, band_type="lower")
    elif band_type == "risk_averse":
        max_band = vanilla_band(y_forecast, scp=scp, band_type="lower")
        min_band = vanilla_band(y_forecast, scp=scp, band_type="upper")

    # initial plan of trading
    argmax = int(np.argmax(max_band))
    argmin = int(np.argmin(min_band))

    if argmin > argmax:
        planned_direction = -1
        planned_entry = argmax
        planned_exit = argmin
    else:
        planned_direction = 1
        planned_entry = argmin
        planned_exit = argmax

    direction = planned_direction

    initial_planned_entry = planned_entry
    initial_planned_exit = planned_exit

    basic_profit = (
        y_actual[initial_planned_exit] - y_actual[initial_planned_entry]
    ) * direction
    best_profit = np.max(y_actual) - np.min(y_actual)
    worst_profit = -best_profit

    # indicator if we are in position already
    in_position = False
    played = False

    # entry price and index
    entry_price = None

    profit = 0

    # observe t=0..T-1 and adapt plan if a more profitable buy/sell points are detected
    for t in range(T):
        # compute weights using data observed up to t (inclusive)
        price_so_far = y_actual[: t + 1]
        forecast_so_far = y_forecast[: t + 1, :]

        residuals = np.median(forecast_so_far, axis=1) - price_so_far
        trust_threshold, nonzero_mae = get_trust_threshold(
            residuals, trust_threshold_method
        )
        w = compute_weights(
            forecast_so_far,
            price_so_far,
            nonzero_mae,
            p,
            lambda_,
            weights_method,
        )

        # build conditional medians for future times > t
        future_count = T - (t + 1)
        if future_count <= 0:
            # no future points; if in position close at last observed price
            if in_position:
                exit_price = y_actual[t]
                profit += (exit_price - entry_price) * direction
                in_position = False
            break

        if band_type == "risk_seeking":
            cond_max_band = weighted_band(y_forecast[t + 1 :, :], w, scp, "upper")
            cond_min_band = weighted_band(y_forecast[t + 1 :, :], w, scp, "lower")
        elif band_type == "risk_averse":
            cond_max_band = weighted_band(y_forecast[t + 1 :, :], w, scp, "lower")
            cond_min_band = weighted_band(y_forecast[t + 1 :, :], w, scp, "upper")

        if dev_plots:
            if t == 0:
                add_curve(fig, x, max_band, "Max Band", "blue")
                add_curve(fig, x, min_band, "Min Band", "blue")
                add_curve(fig, x, y_actual, "Actual", "green")
            add_curve(
                fig, x[T - len(cond_max_band) :], cond_max_band, f"Max Band {t}", "red"
            )
            add_curve(
                fig, x[T - len(cond_min_band) :], cond_min_band, f"Min Band {t}", "red"
            )

        # map back to absolute indices
        rel_argmax = int(np.argmax(cond_max_band))
        rel_argmin = int(np.argmin(cond_min_band))
        new_argmax = rel_argmax + (t + 1)
        new_argmin = rel_argmin + (t + 1)

        # desired trading plan from conditional medians
        if new_argmin > new_argmax:
            desired_direction = -1
            desired_entry = new_argmax
            desired_exit = new_argmin
        else:
            desired_direction = 1
            desired_entry = new_argmin
            desired_exit = new_argmax

        if planned_entry > t:
            if (
                planned_direction == -1
            ):  # if we short we go from entry on max band to exit on min band
                planned_entry_profit = (
                    cond_min_band[planned_exit - t - 1]
                    - cond_max_band[planned_entry - t - 1]
                )
            else:  # if we long we go from entry on min band to exit on max band
                planned_entry_profit = (
                    cond_max_band[planned_exit - t - 1]
                    - cond_min_band[planned_entry - t - 1]
                )
        elif planned_entry == t:
            if (
                planned_direction == -1
            ):  # if we short we go from entry on max band to exit on min band
                planned_entry_profit = (
                    cond_min_band[planned_exit - t - 1] - y_actual[planned_entry]
                )
            else:  # if we long we go from entry on min band to exit on max band
                planned_entry_profit = (
                    cond_max_band[planned_exit - t - 1] - y_actual[planned_entry]
                )

        if desired_entry > t:
            if desired_direction == -1:
                desired_entry_profit = (
                    cond_min_band[desired_exit - t - 1]
                    - cond_max_band[desired_entry - t - 1]
                )
            else:
                desired_entry_profit = (
                    cond_max_band[desired_exit - t - 1]
                    - cond_min_band[desired_entry - t - 1]
                )
        elif desired_entry == t:
            if desired_direction == -1:
                desired_entry_profit = (
                    cond_min_band[desired_exit - t - 1] - y_actual[desired_entry]
                )
            else:
                desired_entry_profit = (
                    cond_max_band[desired_exit - t - 1] - y_actual[desired_entry]
                )

        # we shift the entering of position if we see more profit from changing it
        if (
            desired_exit != desired_entry
            and not in_position
            and desired_entry_profit * desired_direction - trust_threshold
            > planned_entry_profit * direction
        ):
            planned_entry = desired_entry
            planned_exit = desired_exit
            planned_direction = desired_direction

        # entry logic: if not in position and planned entry is now or in past -> enter
        if (not in_position) and (planned_entry == t):
            entry_price = y_actual[t]
            exit_index = planned_exit
            in_position = True
            played = True
            direction = planned_direction  # commit to direction at entry time

        if in_position:
            # check whether taking profit based on current weighted median and observed errors is profitable
            if (
                direction == -1
                and (y_actual[t] - entry_price) * direction
                > (min(cond_min_band) - entry_price) * direction + trust_threshold
            ) or (
                direction == 1
                and (y_actual[t] - entry_price) * direction
                > (max(cond_max_band) - entry_price) * direction + trust_threshold
            ):
                exit_price = y_actual[t]
                profit += (exit_price - entry_price) * direction
                in_position = False
                break

            # if planned exit is now -> check whether it is worth waiting and if not exit, otherwise update the exit time
            if exit_index == t:
                if (
                    direction == -1
                    and (y_actual[t] - entry_price) * direction
                    > (min(cond_min_band) - entry_price) * direction - trust_threshold
                ) or (
                    direction == 1
                    and (y_actual[t] - entry_price) * direction
                    > (max(cond_max_band) - entry_price) * direction - trust_threshold
                ):
                    exit_price = y_actual[t]
                    profit += (exit_price - entry_price) * direction
                    in_position = False
                    break
                else:
                    if direction == -1:
                        exit_index = new_argmin
                    elif direction == 1:
                        exit_index = new_argmax

    # end loop: if still in position close at last observation
    if in_position:
        exit_price = y_actual[-1]
        profit += (exit_price - entry_price) * direction

    return profit, basic_profit, played, best_profit, worst_profit
