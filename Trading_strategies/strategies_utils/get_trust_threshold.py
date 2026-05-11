import numpy as np


def get_trust_threshold(residuals, trust_threshold_method):
    """Compute the trust threshold for trading decision change in dynamic strategies."""
    raw_median_mae = np.mean(np.abs(residuals))

    if len(residuals) > 1:
        sigma = np.std(residuals)
        iqr = np.subtract(*np.percentile(residuals, [75, 25]))
        ipr = np.subtract(*np.percentile(residuals, [95, 5]))
    else:
        sigma = np.sqrt(np.sum((residuals) ** 2))
        iqr = np.mean(np.abs(residuals))
        ipr = np.mean(np.abs(residuals))

    if trust_threshold_method == "3sigma":
        trust_threshold = 3 * sigma
    elif trust_threshold_method == "iqr":
        trust_threshold = iqr
    elif trust_threshold_method == "5_95_IPR":
        trust_threshold = ipr
    elif trust_threshold_method == "mae":
        trust_threshold = raw_median_mae

    nonzero_mae = max([raw_median_mae, 0.01])

    return trust_threshold, nonzero_mae
