"""A collection of helpers for path cSVR implementation; declaration of custom kernels"""

import numpy as np
import scipy
from numba import njit


@njit
def my_mae(X, Y):
    return np.mean(np.abs(X - Y))


@njit
def calc_interm_kernel(interm_data_1, interm_data_2, are_equal=True, norm: int = 2):
    """
    Compute a pairwise distance matrix between vectors for use as a kernel exponent.
    Note that the loops could be vectorized, but this would lead to excessive RAM usage.

    Parameters
    ----------
    interm_data : array_like
        A 2D array of shape (n_samples, n_features) containing the input vectors.
    norm : int, optional
        The norm to use for distance computation. Supported values are:
        - ``2`` : Euclidean (L2) distance (default)
        - ``1`` : Manhattan (L1) distance

    Returns
    -------
    plain_kernel : ndarray
        A 2D array of shape (n_samples, n_samples) containing the pairwise distances.
        The matrix is symmetric with zeros on the diagonal.
    """
    plain_kernel = np.zeros((len(interm_data_1), len(interm_data_2)))
    for i in range(len(interm_data_1)):
        for j in range(len(interm_data_2)):
            if not are_equal or i < j:
                if norm == 2:
                    plain_kernel[i, j] = np.sqrt(
                        np.sum((interm_data_1[i] - interm_data_2[j]) ** 2)
                    )  # L2 distance
                elif norm == 1:
                    plain_kernel[i, j] = np.sum(
                        np.abs(interm_data_1[i] - interm_data_2[j])
                    )  # L1 distance
    if are_equal:
        plain_kernel = plain_kernel + plain_kernel.T
    return plain_kernel


def multifore_corrected_laplace_kernel(
    X, Y, q_kernel, q_data, q_data_naive, q_kernel_naive, is_train=None
):
    """RAM inefficient way of training and then forecasting using cSVR with one correcting variable.
    It allows for forecasting multiple scenarios at once using broadcasting.
    """
    normal_quant = scipy.stats.norm.ppf(q_kernel_naive, loc=0, scale=1)
    if is_train:  # training case
        intermediate_kernel = calc_interm_kernel(X[:, 1:], X[:, 1:])
        width = np.log(2 - 2 * q_kernel) / np.quantile(intermediate_kernel, q_data)

        naive_vec_standardized = X[:, 0]
        intermediate_naive_kernel = (
            naive_vec_standardized[:, None] - naive_vec_standardized
        ) ** 2
        sigma = np.quantile(intermediate_naive_kernel, q_data_naive) / normal_quant

    else:  # test case - Y is the training data, X is the test data
        intermediate_kernel_train = calc_interm_kernel(Y[:, 1:], Y[:, 1:])
        intermediate_kernel = np.sqrt(
            np.sum((X[:, None, 1:] - Y[None, :, 1:]) ** 2, axis=2)
        )  # use the broadcasting to calculate multiple scenarios at once
        width = np.log(2 - 2 * q_kernel) / np.quantile(
            intermediate_kernel_train, q_data
        )

        naive_vec_standardized = Y[:, 0]
        intermediate_naive_kernel = (X[:, None, 0] - Y[None, :, 0]) ** 2
        intermediate_naive_kernel_train = (
            naive_vec_standardized[:, None] - naive_vec_standardized
        ) ** 2
        sigma = (
            np.quantile(intermediate_naive_kernel_train, q_data_naive) / normal_quant
        )

    kernel = np.exp(
        width * intermediate_kernel - 1 / (2 * sigma**2) * intermediate_naive_kernel
    )

    return kernel


def corrected_laplace_kernel(
    X, Y, q_kernel, q_data, q_data_naive, q_kernel_naive, is_train=False
):
    """
    RAM efficient way of training and then forecasting using cSVR with one correcting variable and
    one forecasted point at a time.
    Forecasting multiple points at a time requires more RAM and is slower - we do not consider it for historical simulation.
    """
    normal_quant = scipy.stats.norm.ppf(q_kernel_naive, loc=0, scale=1)
    if is_train:  # training case - more than one day to forecast and Y is ignored
        intermediate_kernel = calc_interm_kernel(X[:, 1:], X[:, 1:])
        width = np.log(2 - 2 * q_kernel) / np.quantile(intermediate_kernel, q_data)

        naive_vec_standardized = X[:, 0]
        intermediate_naive_kernel = (
            naive_vec_standardized[:, None] - naive_vec_standardized
        ) ** 2
        sigma = np.quantile(intermediate_naive_kernel, q_data_naive) / normal_quant

    else:  # test case - Y is the training data, X is the test data
        intermediate_kernel_train = calc_interm_kernel(Y[:, 1:], Y[:, 1:])
        if np.shape(X)[0] == 1:
            intermediate_kernel = np.sqrt(np.sum((Y[:, 1:] - X[:, 1:]) ** 2, axis=1))
        else:
            intermediate_kernel = calc_interm_kernel(
                X[:, 1:], Y[:, 1:], are_equal=np.array_equal(X, Y)
            )  # calc_interm_kernel assumes we only enter this part if we forecast in-sample
        width = np.log(2 - 2 * q_kernel) / np.quantile(
            intermediate_kernel_train, q_data
        )

        naive_vec_standardized = Y[:, 0]
        intermediate_naive_kernel = (naive_vec_standardized - X[:, :1]) ** 2
        intermediate_naive_kernel_train = (
            naive_vec_standardized[:, None] - naive_vec_standardized
        ) ** 2
        sigma = (
            np.quantile(intermediate_naive_kernel_train, q_data_naive) / normal_quant
        )

    kernel = np.exp(
        width * intermediate_kernel - 1 / (2 * sigma**2) * intermediate_naive_kernel
    )

    return kernel
