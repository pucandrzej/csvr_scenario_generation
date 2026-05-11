import numpy as np


def naive_1(y_actual, direction, one_sided):
    """
    Naive that always sells in 1st step for seller and sells in 1st step and buys at last step for speculator
    """

    if float(direction) == -1 and one_sided:
        profit = y_actual[0]
        best_profit = max(y_actual)
        worst_profit = min(y_actual)
    elif float(direction) == 0 and not one_sided:
        profit = y_actual[0] - y_actual[-1]
        best_profit = np.abs(max(y_actual) - min(y_actual))
        worst_profit = -best_profit
    else:
        raise ValueError("Buyer is not implemented.")

    # no trade executed
    return profit, 0, 1, best_profit, worst_profit


def naive_30(y_actual, direction, one_sided):
    """
    Naive that always sells in last step for seller and sells in last step and buys at first step for speculator
    """

    if float(direction) == -1 and one_sided:
        profit = y_actual[-1]
        best_profit = max(y_actual)
        worst_profit = min(y_actual)
    elif float(direction) == 0 and not one_sided:
        profit = y_actual[-1] - y_actual[0]
        best_profit = np.abs(max(y_actual) - min(y_actual))
        worst_profit = -best_profit
    else:
        raise ValueError("Buyer is not implemented.")

    # no trade executed
    return profit, 0, 1, best_profit, worst_profit
