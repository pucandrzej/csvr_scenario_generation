import functools
import time
import numpy as np


def my_mae(X, Y):
    return np.mean(np.abs(X - Y))


def timing(func):
    """Decorator to measure execution time of a function."""

    @functools.wraps(func)
    def wrapper_timing(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"⏱️ Function '{func.__name__}' executed in {end - start:.2f} seconds")
        return result

    return wrapper_timing


def analysis_pinball_loss(y_true, y_pred_quantile, q):
    """
    Calculate Pinball Loss for quantile predictions.

    Parameters:
    - y_true: array-like, actual values
    - y_pred_quantile: array-like, predicted q-th quantiles
    - q: float, quantile level (e.g., 0.1, 0.5, 0.9)

    Returns:
    - float, mean pinball loss
    """
    y_true = np.array(y_true)
    y_pred_quantile = np.array(y_pred_quantile)

    error = y_true - y_pred_quantile
    loss = np.where(error >= 0, q * error, (1 - q) * (-error))
    return np.mean(loss)
