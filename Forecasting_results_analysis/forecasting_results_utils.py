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
        print(f"Function '{func.__name__}' executed in {end - start:.2f} seconds")
        return result

    return wrapper_timing
