import numpy as np

from Trading_strategies.strategies_utils import (
    compute_weights,
    get_trust_threshold,
)


def seller_initial_trading_plan(y_actual, argmax_forecast):
    planned_entry = argmax_forecast

    basic_profit = y_actual[planned_entry]

    best_profit = np.max(y_actual)
    worst_profit = np.min(y_actual)

    return planned_entry, basic_profit, best_profit, worst_profit


def speculative_initial_trading_plan(y_actual, argmin_forecast, argmax_forecast):
    # initial trading results based on model forecast
    if argmin_forecast > argmax_forecast:
        planned_direction = -1
        planned_entry = argmax_forecast
        planned_exit = argmin_forecast
    else:
        planned_direction = 1
        planned_entry = argmin_forecast
        planned_exit = argmax_forecast

    direction = planned_direction

    initial_planned_entry = planned_entry
    initial_planned_exit = planned_exit

    basic_profit = (
        y_actual[initial_planned_exit] - y_actual[initial_planned_entry]
    ) * direction
    best_profit = np.max(y_actual) - np.min(y_actual)
    worst_profit = -best_profit

    return (
        planned_direction,
        planned_entry,
        planned_exit,
        direction,
        basic_profit,
        best_profit,
        worst_profit,
    )


def get_weights_and_trust_threshold(y_actual, y_forecast, t, config):
    # compute weights using data observed up to t (inclusive)
    price_so_far = y_actual[: t + 1]
    forecast_so_far = y_forecast[: t + 1, :]

    residuals = np.median(forecast_so_far, axis=1) - price_so_far
    trust_threshold, nonzero_mae = get_trust_threshold(
        residuals, config.trust_threshold_method
    )
    w = compute_weights(
        forecast_so_far,
        price_so_far,
        nonzero_mae,
        config.p,
        config.lambda_,
        config.weights_method,
    )

    return w, trust_threshold


def speculative_desired_trading_plan(cond_min_forecast, cond_max_forecast, t):
    # map back to absolute indices
    rel_argmax = int(np.argmax(cond_max_forecast))
    rel_argmin = int(np.argmin(cond_min_forecast))
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

    return desired_direction, desired_entry, desired_exit, new_argmin, new_argmax


def seller_strategy_correction(
    adjusted_forecast, planned_entry, y_actual, trust_threshold, profit, played, t
):
    # map back to absolute indices
    rel_argmax = int(np.argmax(adjusted_forecast))
    new_argmax = rel_argmax + (t + 1)

    desired_entry = new_argmax

    if planned_entry > t:
        planned_entry_profit = adjusted_forecast[planned_entry - t - 1]
    elif planned_entry == t:
        planned_entry_profit = y_actual[planned_entry]

    if desired_entry > t:
        desired_entry_profit = adjusted_forecast[desired_entry - t - 1]
    elif desired_entry == t:
        desired_entry_profit = y_actual[desired_entry]

    # we shift the entering of position if we see more profit from changing it
    if desired_entry_profit - trust_threshold > planned_entry_profit:
        planned_entry = desired_entry

    # entry logic: if not in position and planned entry is now -> enter
    if planned_entry == t:
        played = True
        profit = y_actual[planned_entry]
        return True, profit, planned_entry, played

    return False, profit, planned_entry, played


def speculative_strategy_correction(
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
):

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
            > (minimum_of_forecast - entry_price) * direction + trust_threshold
        ) or (
            direction == 1
            and (y_actual[t] - entry_price) * direction
            > (maximum_of_forecast - entry_price) * direction + trust_threshold
        ):
            exit_price = y_actual[t]
            profit = (exit_price - entry_price) * direction
            in_position = False
            return (
                True,
                profit,
                exit_index,
                planned_entry,
                planned_exit,
                planned_direction,
                entry_price,
                played,
                direction,
                in_position,
            )  # returning True as a first element tells the code above to break the loop over trajectory steps

        # if planned exit is now -> check whether it is worth waiting and if not exit, otherwise update the exit time
        if exit_index == t:
            if (
                direction == -1
                and (y_actual[t] - entry_price) * direction
                > (minimum_of_forecast - entry_price) * direction - trust_threshold
            ) or (
                direction == 1
                and (y_actual[t] - entry_price) * direction
                > (maximum_of_forecast - entry_price) * direction - trust_threshold
            ):
                exit_price = y_actual[t]
                profit = (exit_price - entry_price) * direction
                in_position = False
                return (
                    True,
                    profit,
                    exit_index,
                    planned_entry,
                    planned_exit,
                    planned_direction,
                    entry_price,
                    played,
                    direction,
                    in_position,
                )  # returning True as a first element tells the code above to break the loop over trajectory steps
            else:
                if direction == -1:
                    exit_index = new_argmin
                elif direction == 1:
                    exit_index = new_argmax

    return (
        False,
        profit,
        exit_index,
        planned_entry,
        planned_exit,
        planned_direction,
        entry_price,
        played,
        direction,
        in_position,
    )
