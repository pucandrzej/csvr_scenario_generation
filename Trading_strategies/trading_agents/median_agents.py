import numpy as np
import plotly.graph_objects as go
from dataclasses import dataclass

from Trading_strategies.strategies_utils import (
    get_trust_threshold,
    compute_weights,
    weighted_median,
    add_curve,
)


@dataclass(kw_only=True)
class BaseMedianConfig:
    p: float
    lambda_: float = 0.25
    trust_threshold_method: str = "3sigma"
    weights_method: str = "kernel"
    dev_plots: bool = False


@dataclass(kw_only=True)
class OneSidedMedianConfig(BaseMedianConfig):
    direction: int


@dataclass(kw_only=True)
class TwoSidedMedianConfig(BaseMedianConfig):
    pass


def one_sided_median_trading_strategy(
    y_actual: np.ndarray,
    y_forecast: np.ndarray,
    config: OneSidedMedianConfig,
):
    """
    Minimal dynamic evolution-tracking.
    - initial entry/exit from unconditional median across paths
    - at each time t, compute path weights based on observed history
    - compute weighted medians for future times and replan entry/exit
    - simulate immediate fills: enter when planned_entry <= t, exit when planned_exit <= t
    - if direction flips while in position, close and flip immediately
    - returns profit (single number).

    y_actual: shape (T,)
    y_forecast: shape (T, Npaths)
    """

    direction = config.direction
    p = config.p
    lambda_ = config.lambda_
    trust_threshold_method = config.trust_threshold_method
    weights_method = config.weights_method
    dev_plots = config.dev_plots

    if dev_plots:
        fig = go.Figure()
        x = np.arange(len(y_actual))

    T, N = y_forecast.shape
    if y_actual.shape[0] != T:
        raise ValueError("Time dimension mismatch")

    # initial unconditional central forecast (median across paths per time)
    central = np.median(y_forecast, axis=1)

    # initial plan of trading
    argmax = int(np.argmax(central))
    argmin = int(np.argmin(central))

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

    # indicator if we are in position already
    played = False
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
            forecast_so_far, price_so_far, nonzero_mae, p, lambda_, weights_method
        )

        # build conditional medians for future times > t
        future_count = T - (t + 1)
        if future_count <= 0:
            break

        cond_medians = np.empty(future_count)
        for idx, s in enumerate(range(t + 1, T)):
            vals = y_forecast[s, :]
            cond_medians[idx] = weighted_median(vals, w)

        if dev_plots:
            if t == 0:
                add_curve(fig, x, central, "Median", "blue")
                add_curve(fig, x, y_actual, "Actual", "green")
            add_curve(
                fig,
                x[T - len(cond_medians) :],
                cond_medians,
                f"Conditional medians {t}",
                "red",
            )

        # map back to absolute indices
        rel_argmax = int(np.argmax(cond_medians))
        rel_argmin = int(np.argmin(cond_medians))
        new_argmax = rel_argmax + (t + 1)
        new_argmin = rel_argmin + (t + 1)

        if direction == 1:
            desired_entry = new_argmin
        elif direction == -1:
            desired_entry = new_argmax

        if planned_entry > t:
            planned_entry_profit = cond_medians[planned_entry - t - 1]
        elif planned_entry == t:
            planned_entry_profit = y_actual[planned_entry]

        if desired_entry > t:
            desired_entry_profit = cond_medians[desired_entry - t - 1]
        elif desired_entry == t:
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

    # no trade executed
    return profit, basic_profit, played, best_profit, worst_profit


def two_sided_median_trading_strategy(
    y_actual: np.ndarray,
    y_forecast: np.ndarray,
    config: TwoSidedMedianConfig,
):
    """
    Minimal dynamic evolution-tracking.
    - initial entry/exit from unconditional median across paths
    - at each time t, compute path weights based on observed history
    - compute weighted medians for future times and replan entry/exit
    - simulate immediate fills: enter when planned_entry <= t, exit when planned_exit <= t
    - if direction flips while in position, close and flip immediately
    - returns profit (single number).

    y_actual: shape (T,)
    y_forecast: shape (T, Npaths)
    """

    p = config.p
    lambda_ = config.lambda_
    trust_threshold_method = config.trust_threshold_method
    weights_method = config.weights_method
    dev_plots = config.dev_plots

    if dev_plots:
        fig = go.Figure()
        x = np.arange(len(y_actual))

    T, N = y_forecast.shape
    if y_actual.shape[0] != T:
        raise ValueError("Time dimension mismatch")

    # initial unconditional central forecast (median across paths per time)
    central = np.median(y_forecast, axis=1)

    # initial plan of trading
    argmax = int(np.argmax(central))
    argmin = int(np.argmin(central))

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
    worst_profit = -best_profit  # for speculator the worst profit is - best profit

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
                profit = (exit_price - entry_price) * direction
                in_position = False
            break

        cond_medians = np.empty(future_count)
        for idx, s in enumerate(range(t + 1, T)):
            vals = y_forecast[s, :]
            cond_medians[idx] = weighted_median(vals, w)

        if dev_plots:
            if t == 0:
                add_curve(fig, x, central, "Median", "blue")
                add_curve(fig, x, y_actual, "Actual", "green")
            add_curve(
                fig,
                x[T - len(cond_medians) :],
                cond_medians,
                f"Conditional medians {t}",
                "red",
            )

        # map back to absolute indices
        rel_argmax = int(np.argmax(cond_medians))
        rel_argmin = int(np.argmin(cond_medians))
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

        # prepare the planned profit: in case the planned entry is at t we already know the price
        if planned_entry > t:
            planned_entry_profit = (
                cond_medians[planned_exit - t - 1] - cond_medians[planned_entry - t - 1]
            )
        elif planned_entry == t:
            planned_entry_profit = (
                cond_medians[planned_exit - t - 1] - y_actual[planned_entry]
            )

        if desired_entry > t:
            desired_entry_profit = (
                cond_medians[desired_exit - t - 1] - cond_medians[desired_entry - t - 1]
            )
        elif desired_entry == t:
            desired_entry_profit = (
                cond_medians[desired_exit - t - 1] - y_actual[desired_entry]
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
                > (min(cond_medians) - entry_price) * direction + trust_threshold
            ) or (
                direction == 1
                and (y_actual[t] - entry_price) * direction
                > (max(cond_medians) - entry_price) * direction + trust_threshold
            ):
                exit_price = y_actual[t]
                profit = (exit_price - entry_price) * direction
                in_position = False
                break

            # if planned exit is now -> check whether it is worth waiting and if not exit, otherwise update the exit time
            if exit_index == t:
                if (
                    direction == -1
                    and (y_actual[t] - entry_price) * direction
                    > (min(cond_medians) - entry_price) * direction - trust_threshold
                ) or (
                    direction == 1
                    and (y_actual[t] - entry_price) * direction
                    > (max(cond_medians) - entry_price) * direction - trust_threshold
                ):
                    exit_price = y_actual[t]
                    profit = (exit_price - entry_price) * direction
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
        profit = (exit_price - entry_price) * direction

    return profit, basic_profit, played, best_profit, worst_profit
