import numpy as np
import plotly.graph_objects as go
from dataclasses import dataclass

from Trading_strategies.trading_agents.agents_utils import (  # all of the interchangeable strategies logic elements are separate functions to avoid repeating the same code
    seller_initial_trading_plan,
    speculative_initial_trading_plan,
    speculative_desired_trading_plan,
    get_weights_and_trust_threshold,
    seller_strategy_correction,
    speculative_strategy_correction,
)

from Trading_strategies.strategies_utils import (
    weighted_median,
    add_curve,
)


@dataclass(kw_only=True)
class BaseMedianConfig:
    """Configuration shared by one- and two-sided median strategies.

    Parameters
    ----------
    p : float
        Kernel shape parameter used for dynamic scenario weighting.
    lambda_ : float
        Exponential time-decay parameter for past forecast errors.
    trust_threshold_method : str
        Method used to calculate the strategy adaptation threshold.
    weights_method : str
        Method used to dynamically weight forecast scenarios.
    dev_plots : bool
        Whether to generate diagnostic strategy plots.
    """

    p: float
    lambda_: float = 0.25
    trust_threshold_method: str = "3sigma"
    weights_method: str = "kernel"
    dev_plots: bool = False


@dataclass(kw_only=True)
class OneSidedMedianConfig(BaseMedianConfig):
    """Configuration for the one-sided seller median strategy."""

    pass


@dataclass(kw_only=True)
class TwoSidedMedianConfig(BaseMedianConfig):
    """Configuration for the two-sided speculative median strategy."""

    pass


def one_sided_median_trading_strategy(
    y_actual: np.ndarray,
    y_forecast: np.ndarray,
    config: OneSidedMedianConfig,
):
    """Simulate the dynamically updated median strategy for a seller.

    The initial selling time is selected from the unconditional ensemble
    median. As prices are observed, scenario weights and future weighted
    medians are updated, and the planned sale is changed when the expected
    improvement exceeds the trust threshold. An unsold position is settled
    at the final trajectory price.

    Parameters
    ----------
    y_actual : np.ndarray
        Realized price trajectory of shape (T,).
    y_forecast : np.ndarray
        Ensemble forecast trajectories of shape (T, N).
    config : OneSidedMedianConfig
        Scenario weighting and strategy adaptation parameters.

    Returns
    -------
    tuple
        Dynamic profit, static strategy profit, action indicator, and
        best- and worst-case realized profits.
    """

    dev_plots = config.dev_plots

    if dev_plots:
        fig = go.Figure()
        x = np.arange(len(y_actual))

    T, _ = y_forecast.shape
    if y_actual.shape[0] != T:
        raise ValueError("Time dimension mismatch")

    # initial unconditional central forecast (median across paths per time)
    central = np.median(y_forecast, axis=1)

    # initial plan of trading
    argmax = int(np.argmax(central))

    planned_entry, basic_profit, best_profit, worst_profit = (
        seller_initial_trading_plan(y_actual.copy(), argmax)
    )

    # indicator if we are in position already
    played = False
    profit = 0

    # observe t=0..T-1 and adapt plan if a more profitable buy/sell points are detected
    for t in range(T):
        w, trust_threshold = get_weights_and_trust_threshold(
            y_actual.copy(), y_forecast.copy(), t, config
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

        break_condition, profit, planned_entry, played = seller_strategy_correction(
            cond_medians.copy(), planned_entry, y_actual.copy(), trust_threshold, profit, played, t
        )

        if break_condition:
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
    """Simulate the dynamically updated median strategy for a spread trader.

    The initial long or short trade is determined from the extrema of the
    unconditional ensemble median. As prices are observed, scenario weights
    and future weighted medians are updated, allowing the entry plan to
    change before execution and the exit to occur early or be postponed
    according to the trust threshold.

    Parameters
    ----------
    y_actual : np.ndarray
        Realized price trajectory of shape (T,).
    y_forecast : np.ndarray
        Ensemble forecast trajectories of shape (T, N).
    config : TwoSidedMedianConfig
        Scenario weighting and strategy adaptation parameters.

    Returns
    -------
    tuple
        Dynamic profit, static strategy profit, action indicator, and
        best- and worst-case realized profits.
    """

    dev_plots = config.dev_plots

    if dev_plots:
        fig = go.Figure()
        x = np.arange(len(y_actual))

    T, _ = y_forecast.shape
    if y_actual.shape[0] != T:
        raise ValueError("Time dimension mismatch")

    # initial unconditional central forecast (median across paths per time)
    central = np.median(y_forecast, axis=1)

    # initial plan of trading
    argmax = int(np.argmax(central))
    argmin = int(np.argmin(central))

    (
        planned_direction,
        planned_entry,
        planned_exit,
        direction,
        basic_profit,
        best_profit,
        worst_profit,
    ) = speculative_initial_trading_plan(y_actual, argmin, argmax)

    # indicator if we are in position already
    in_position = False
    played = False

    # entry price and index
    entry_price = None
    exit_index = None

    profit = 0

    # observe t=0..T-1 and adapt plan if a more profitable buy/sell points are detected
    for t in range(T):
        w, trust_threshold = get_weights_and_trust_threshold(
                    y_actual.copy(), y_forecast.copy(), t, config
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

        desired_direction, desired_entry, desired_exit, new_argmin, new_argmax = (
            speculative_desired_trading_plan(cond_medians, cond_medians, t)
        )

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

        minimum_of_forecast = min(cond_medians)
        maximum_of_forecast = max(cond_medians)

        (
            break_condition,
            profit,
            exit_index,
            planned_entry,
            planned_exit,
            planned_direction,
            entry_price,
            played,
            direction,
            in_position,
        ) = speculative_strategy_correction(
            desired_exit,
            desired_entry,
            in_position,
            entry_price,
            desired_entry_profit,
            desired_direction,
            trust_threshold,
            planned_entry,
            planned_exit,
            exit_index,
            planned_direction,
            planned_entry_profit,
            direction,
            y_actual,
            minimum_of_forecast,
            maximum_of_forecast,
            new_argmin,
            new_argmax,
            profit,
            played,
            t,
        )

        if break_condition:
            break

    # end loop: if still in position close at last observation
    if in_position:
        exit_price = y_actual[-1]
        profit = (exit_price - entry_price) * direction

    return profit, basic_profit, played, best_profit, worst_profit
