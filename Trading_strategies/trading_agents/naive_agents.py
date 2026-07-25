import numpy as np


def naive_1(y_actual, one_sided):
    """
    Naive that always sells in 1st step for seller and sells in 1st step and buys at last step for speculator
    """

    if one_sided:
        profit = y_actual[0]
        best_profit = max(y_actual)
        worst_profit = min(y_actual)
    else:
        profit = y_actual[0] - y_actual[-1]
        best_profit = np.abs(max(y_actual) - min(y_actual))
        worst_profit = -best_profit

    # no trade executed
    return profit, 0, 1, best_profit, worst_profit


def naive_30(y_actual, one_sided):
    """
    Naive that always sells in last step for seller and sells in last step and buys at first step for speculator
    """

    if one_sided:
        profit = y_actual[-1]
        best_profit = max(y_actual)
        worst_profit = min(y_actual)
    else:
        profit = y_actual[-1] - y_actual[0]
        best_profit = np.abs(max(y_actual) - min(y_actual))
        worst_profit = -best_profit

    # no trade executed
    return profit, 0, 1, best_profit, worst_profit
