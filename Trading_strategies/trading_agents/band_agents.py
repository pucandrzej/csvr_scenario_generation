import numpy as np
import plotly.graph_objects as go
from dataclasses import dataclass

from Trading_strategies.strategies_utils import (
    vanilla_band,
    weighted_band,
    add_curve,
)

from Trading_strategies.trading_agents.agents_utils import (  # all of the interchangeable strategies logic elements are separate functions to avoid repeating the same code
    seller_initial_trading_plan,
    speculative_initial_trading_plan,
    speculative_desired_trading_plan,
    get_weights_and_trust_threshold,
    seller_strategy_correction,
    speculative_strategy_correction,
)


@dataclass(kw_only=True)
class BaseBandConfig:
    """Configuration shared by one- and two-sided band strategies.

    Parameters
    ----------
    band_type : str
        Risk attitude determining which simultaneous band is used.
    scp : float
        Simultaneous coverage probability of the forecast band.
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

    band_type: str
    scp: float
    p: float
    lambda_: float = 0.25
    trust_threshold_method: str = "3sigma"
    weights_method: str = "kernel"
    dev_plots: bool = False


@dataclass(kw_only=True)
class OneSidedBandsConfig(BaseBandConfig):
    """Configuration for the one-sided seller band strategy."""

    pass


@dataclass(kw_only=True)
class TwoSidedBandsConfig(BaseBandConfig):
    """Configuration for the two-sided speculative band strategy."""

    pass


def one_sided_bands_strategy(
    y_actual,
    y_forecast,
    config: OneSidedBandsConfig,
):
    """Simulate the dynamically updated band strategy for a seller.

    The initial selling time is selected from a simultaneous forecast band.
    As prices are observed, scenario weights and the conditional band are
    updated and the planned sale is changed when the expected improvement
    exceeds the trust threshold. An unsold position is settled at the final
    trajectory price.

    Parameters
    ----------
    y_actual : np.ndarray
        Realized price trajectory of shape (T,).
    y_forecast : np.ndarray
        Ensemble forecast trajectories of shape (T, N).
    config : OneSidedBandsConfig
        Band construction, weighting, and adaptation parameters.

    Returns
    -------
    tuple
        Dynamic profit, static strategy profit, action indicator, and
        best- and worst-case realized profits.
    """
    band_type = config.band_type
    scp = config.scp
    dev_plots = config.dev_plots

    if dev_plots:
        fig = go.Figure()
        x = np.arange(len(y_actual))

    T, N = y_forecast.shape

    if band_type == "risk_seeking":
        band = vanilla_band(
            y_forecast, scp=scp, band_type="upper"
        )  # taking max of upper band maximizing the max expected price
    elif band_type == "risk_averse":
        band = vanilla_band(
            y_forecast, scp=scp, band_type="lower"
        )  # taking max of lower band maximizing the min expected price

    # initial plan of trading
    argmax = int(np.argmax(band))
    planned_entry, basic_profit, best_profit, worst_profit = (
        seller_initial_trading_plan(y_actual.copy(), argmax)
    )

    played = False

    profit = 0  # profit declaration

    # iterate over t=0..T-1 and adapt plan if a more profitable buy/sell points are detected
    for t in range(T):
        w, trust_threshold = get_weights_and_trust_threshold(
            y_actual.copy(), y_forecast.copy(), t, config
        )
        # build conditional medians for future times > t
        future_count = T - (t + 1)
        if future_count <= 0:
            break

        if band_type == "risk_seeking":
            cond_band = weighted_band(y_forecast[t + 1 :, :], w, scp, "upper")
        elif band_type == "risk_averse":
            cond_band = weighted_band(y_forecast[t + 1 :, :], w, scp, "lower")

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

        break_condition, profit, played = seller_strategy_correction(
            cond_band, planned_entry, y_actual, trust_threshold, profit, played, t
        )

        if break_condition:
            break

    # force an action at the end of the path if action was not performed in course of the path
    if not played:
        played = True
        profit = y_actual[-1]

    return profit, basic_profit, played, best_profit, worst_profit


def two_sided_bands_strategy(y_actual, y_forecast, config: TwoSidedBandsConfig):
    """Simulate the dynamically updated band strategy for a spread trader.

    The strategy initially plans a long or short trade from extrema of the
    simultaneous forecast bands. As prices are observed, scenario weights
    and conditional bands are updated, allowing the entry plan to change
    before execution and the exit to occur early or be postponed according
    to the trust threshold.

    Parameters
    ----------
    y_actual : np.ndarray
        Realized price trajectory of shape (T,).
    y_forecast : np.ndarray
        Ensemble forecast trajectories of shape (T, N).
    config : TwoSidedBandsConfig
        Band construction, weighting, and adaptation parameters.

    Returns
    -------
    tuple
        Dynamic profit, static strategy profit, action indicator, and
        best- and worst-case realized profits.
    """
    band_type = config.band_type
    scp = config.scp
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

    (
        planned_direction,
        planned_entry,
        planned_exit,
        direction,
        basic_profit,
        best_profit,
        worst_profit,
    ) = speculative_initial_trading_plan(y_actual.copy(), argmin, argmax)

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

        desired_direction, desired_entry, desired_exit, new_argmin, new_argmax = (
            speculative_desired_trading_plan(
                cond_min_band.copy(), cond_max_band.copy(), t
            )
        )

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

        minimum_of_forecast = min(cond_min_band)
        maximum_of_forecast = max(cond_max_band)

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
