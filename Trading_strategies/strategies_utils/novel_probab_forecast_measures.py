import numpy as np


def weighted_classification_accuracy(y_actual, y_forecast, naive):
    """
    Compute Weighted Classification Accuracy (WCA) and vanilla CA.

    Parameters
    ----------
    y_actual : array-like
        Observed realizations.
    y_forecast : ndarray
        Forecast ensemble (T x N-paths).
    naive : array-like
        Naive benchmark.

    Returns
    -------
    wca : float
        Weighted classification accuracy.
    vanilla_ca : float
        Unweighted classification accuracy.
    """

    y_forecast_median = np.median(y_forecast, axis=1)
    indic_fore = np.where(
        y_forecast_median > naive, 1, np.where(y_forecast_median < naive, -1, 0)
    )
    indic_actual = np.where(y_actual > naive, 1, np.where(y_actual < naive, -1, 0))

    # Compute weights: Higher MAE means higher penalty
    mae = np.abs(y_actual - y_forecast_median)
    weights = 1 + (
        (mae - np.min(mae)) / (np.max(mae) - np.min(mae))
    )  # Normalize by mean MAE

    # Compute weighted misclassification
    misclassified = (indic_actual != indic_fore).astype(int)
    weighted_misclassification = np.sum(weights * misclassified)

    # Compute WCA
    wca = 1 - (weighted_misclassification / np.sum(weights))

    # Compute vanilla (unweighted) CA
    vanilla_ca = 1 - np.sum(misclassified) / len(misclassified)

    return wca, vanilla_ca


def probabilistic_weighted_classification_accuracy(y_actual, y_forecast, naive):
    """
    Compute the Probabilistic Weighted Classification Accuracy (PWCA)
    and vanilla CA.

    Parameters
    ----------
    y_actual : array-like
        Observed values.
    y_forecast : ndarray
        Forecast ensemble.
    naive : array-like
        Naive benchmark.

    Returns
    -------
    wca : float
        Probabilistic weighted classification accuracy.
    vanilla_ca : float
        Unweighted classification accuracy.
    """

    y_forecast_median = np.median(y_forecast, axis=1)
    indic_fore = np.where(
        y_forecast_median > naive, 1, np.where(y_forecast_median < naive, -1, 0)
    )
    indic_actual = np.where(y_actual > naive, 1, np.where(y_actual < naive, -1, 0))

    # Compute weights: Higher MAE means higher penalty
    mae = np.abs(y_actual - y_forecast_median)
    weights = 1 + (
        (mae - np.min(mae)) / (np.max(mae) - np.min(mae))
    )  # Normalize by mean MAE

    # Correct weights by the probability
    y_na = naive.reshape(-1, 1)
    y_fc = y_forecast
    below = (y_fc < y_na).mean(axis=1)
    equal = (y_fc == y_na).mean(axis=1)
    above = (y_fc > y_na).mean(axis=1)
    probab = np.where(indic_fore == -1, below, np.where(indic_fore == 0, equal, above))

    # Compute weighted misclassification
    misclassified = (indic_actual != indic_fore).astype(int)
    weighted_misclassification = np.sum(weights * probab * misclassified)

    # Compute WCA
    wca = 1 - (weighted_misclassification / np.sum(weights * probab))

    # Compute vanilla (unweighted) CA
    vanilla_ca = 1 - np.sum(misclassified) / len(misclassified)

    return wca, vanilla_ca
