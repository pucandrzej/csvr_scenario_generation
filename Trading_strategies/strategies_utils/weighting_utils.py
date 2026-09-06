import numpy as np


def compute_weights(
    forecast_so_far, observed_so_far, kernel_width, p, lambda_, weights_method="kernel"
):
    """
    Compute normalized path weights given forecasts and observations.

    Parameters
    ----------
    forecast_so_far : ndarray
        Shape (t+1, N-paths). Forecast history for each path.
    observed_so_far : ndarray
        Shape (t+1,). Observed historical values.
    kernel_width : float
        Kernel width parameter.
    p : float
        Norm degree (p=2 → Gaussian kernel).
    lambda_ : float
        Time-decay factor for weighting past errors.
    weights_method : {'kernel','mae'}
        Weighting scheme selection.

    Returns
    -------
    ndarray
        Normalized weights of shape (Npaths,).
    """
    # compute errors per path
    diffs = forecast_so_far - observed_so_far[:, None]

    times = np.arange(len(observed_so_far))
    age = len(observed_so_far) - 1 - times
    w_time = np.exp(-lambda_ * age)
    w_time /= np.sum(w_time)

    if weights_method == "kernel":
        # we are calculating the kernel on TIME DIMENSION vectors: so each observation from trajectory is a vector element, we apply decay weight and sum over it
        err = np.sum(w_time[:, None] * (np.abs(diffs)) ** 2, axis=0)
        log_weights = -kernel_width * (err ** (p / 2))  # for p = 2 it is gaussian kernel
        raw = np.exp(log_weights - log_weights.max())

        # scale the weights so that they sum to 1
        s = np.sum(raw)
        if (
            s == 0 or not np.isfinite(s)
        ):  # it can happen that the error is large and we are not able to scale efficiently with any of our paths
            N = raw.size
            return np.ones(N) / N

        weights = raw / s
    elif weights_method == "mae":
        diffs = np.where(
            diffs == 0, 1e-6, diffs
        )  # handle the rare cases (~30) where naive sampling gave us exact price (can happen sometimes especially in night hours)
        all_paths_mae = np.mean(np.abs(diffs), axis=0)
        inverse_mae = 1 / all_paths_mae
        weights = inverse_mae / np.sum(inverse_mae)

    return weights


def weighted_median(values, weights):
    """
    Compute the weighted median of a set of values.

    Parameters
    ----------
    values : array-like
        Data values.
    weights : array-like
        Corresponding weights.

    Returns
    -------
    float
        Median using weighted Hazen interpolation.
    """
    return batch_weighted_quantiles(values, weights, [0.5])[0]


def _calc_band(M, Y, idx, band_type):
    """
    Compute an upper or lower prediction band for a given index.

    Parameters
    ----------
    M : array-like
        Summary statistic per path (max or min).
    Y : ndarray
        Forecast ensemble of shape (T, Npaths).
    idx : int
        Index corresponding to quantile level.
    band_type : {'upper', 'lower'}
        Band type selection.

    Returns
    -------
    ndarray
        Band values for each time step.
    """
    if band_type == "upper":
        lt = np.argsort(M)[: idx + 1]
        B = np.max(Y[:, lt], axis=1)
    elif band_type == "lower":
        lt = np.argsort(M)[::-1][: idx + 1]
        B = np.min(Y[:, lt], axis=1)
    return B


def vanilla_band(Y, scp, band_type):
    """
    Compute a vanilla (unweighted) prediction band.

    Parameters
    ----------
    Y : ndarray
        Forecast ensemble (T x Npaths).
    scp : float
        Probability level of the band.
    band_type : {'upper', 'lower'}
        Which band to compute.

    Returns
    -------
    ndarray
        Prediction band across time.
    """
    m = np.shape(Y)[1]
    if band_type == "upper":
        M = np.max(Y, axis=0)
    elif band_type == "lower":
        M = np.min(Y, axis=0)

    levels = np.arange(1, m + 1) / m

    idx = min([np.searchsorted(levels, scp), len(levels) - 1])
    if levels[idx] == scp or idx == 0:
        B = _calc_band(M, Y, idx, band_type=band_type)
    else:
        B1 = _calc_band(M, Y, idx - 1, band_type=band_type)
        B2 = _calc_band(M, Y, idx, band_type=band_type)
        l1, l2 = levels[idx - 1], levels[idx]
        denominator = l2 - l1
        nominator_1 = B1 * (l2 - scp)
        nominator_2 = B2 * (scp - l1)
        B = (nominator_1 + nominator_2) / denominator
    return B


def weighted_band(Y, weights, scp, band_type):
    """
    Compute a weighted prediction band.

    Parameters
    ----------
    Y : ndarray
        Forecast ensemble (T x Npaths).
    weights : array-like
        Path weights summing to 1.
    scp : float
        Coverage probability.
    band_type : {'upper','lower'}

    Returns
    -------
    ndarray
        Weighted prediction band.
    """
    keep = weights > 0
    Y = Y[:, keep]
    weights = weights[keep]

    if band_type == "upper":
        M = np.max(Y, axis=0)
    elif band_type == "lower":
        M = np.min(Y, axis=0)
    levels, M_sorted, Y_sorted = _sort_values_and_weights(
        weights, M, Y, band_type=band_type
    )

    idx = min([np.searchsorted(levels, scp), len(levels) - 1])

    if levels[idx] == scp or idx == 0:
        B = _calc_weighted_band(Y_sorted, idx, band_type=band_type)
    else:
        B1 = _calc_weighted_band(Y_sorted, idx - 1, band_type=band_type)
        B2 = _calc_weighted_band(Y_sorted, idx, band_type=band_type)

        l1, l2 = levels[idx - 1], levels[idx]
        denominator = l2 - l1
        nominator_1 = B1 * (l2 - scp)
        nominator_2 = B2 * (scp - l1)
        if denominator == 0:  # edge case that should not happen
            raise ValueError("Denominator for weighted path is 0")
        B = (nominator_1 + nominator_2) / denominator

    return B


def _sort_values_and_weights(weights, values, Y, band_type):
    """
    Sort values, weights, and ensemble paths for weighted band calculation.

    Parameters
    ----------
    weights : array-like
        Path weights.
    values : array-like
        Max/min summary stats per path.
    Y : ndarray
        Forecast ensemble (T x Npaths).
    band_type : {'upper','lower'}

    Returns
    -------
    levels : ndarray
        Cumulative weight levels.
    values_sorted : ndarray
        Sorted values.
    Y_sorted : ndarray
        Forecast ensemble sorted by values.
    """
    if band_type == "upper":
        i = np.argsort(values)
    elif band_type == "lower":
        i = np.argsort(values)[::-1]
    values_sorted = values[i]
    w_sorted = weights[i]
    Y_sorted = Y[:, i]
    levels = np.cumsum(w_sorted)
    return levels, values_sorted, Y_sorted


def _calc_weighted_band(Y, idx, band_type):
    """
    Compute a weighted band given pre-sorted paths.

    Parameters
    ----------
    Y : ndarray
        Forecast ensemble.
    idx : int
        Index corresponding to weight level.
    band_type : {'upper','lower'}

    Returns
    -------
    ndarray
        Weighted prediction band.
    """
    if band_type == "upper":
        B = np.max(Y[:, : idx + 1], axis=1)
    elif band_type == "lower":
        B = np.min(Y[:, : idx + 1], axis=1)
    return B


def batch_weighted_quantiles(values, weights, qs, *, presorted=False):
    """Weighted Hazen quantiles; zero weights ignored, endpoints clipped.

    Expects finite 1-D values and quantile levels in [0, 1]. Weights need
    not sum to one. Set presorted=True when values are already sorted.
    Equal weights use the same interpolation formula as unequal weights.
    """
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    q = np.atleast_1d(np.asarray(qs, dtype=float))

    if np.any(w < 0) or not np.all(np.isfinite(w)):
        raise ValueError("Weights must be finite and nonnegative")

    if not presorted:
        order = np.argsort(v)
        v, w = v[order], w[order]

    keep = w > 0
    v, w = v[keep], w[keep]
    if w.size == 0:
        raise ValueError("At least one positive weight is required")

    cumulative = np.cumsum(w)
    positions = (cumulative - 0.5 * w) / cumulative[-1]
    return np.interp(q, positions, v)
