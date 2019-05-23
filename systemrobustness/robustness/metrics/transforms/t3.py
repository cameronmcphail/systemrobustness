"""Contains T3 performance transformations (aggregation to robustness)

Generally the selected and transformed performance values, f, are
transformed to an expected value of performance.

However, supplementary metrics may consider the variance in f,
or higher-order moments of f.
"""

import numpy as np


def f_mean(f):
    """Calculate robustness as mean of f

    Parameters
    ----------
    f : np.ndarray, shape=(m, n)
        Transformed performance values to be maximised.
        m decision alternatives and n scenarios

    Returns
    -------
    np.ndarray, shape=(m, )
        The robustness value for each of the m decision alternatives
    """
    R = np.mean(f, axis=1)
    return R


def f_sum(f):
    """Calculate robustness as sum of f (/ n_scenarios)

    Parameters
    ----------
    f : np.ndarray, shape=(m, n)
        Transformed performance values to be maximised.
        m decision alternatives and n scenarios

    Returns
    -------
    np.ndarray, shape=(m, )
        The robustness value for each of the m decision alternatives
    """
    R = np.sum(f, axis=1) / f.shape[1]
    return R


def f_w_sum(f, weights):
    """Calculate robustness as weighted sum of f

    Parameters
    ----------
    f : np.ndarray, shape=(m, n)
        Transformed performance values to be maximised.
        m decision alternatives and n scenarios
    weights : np.ndarray, shape=(n, )
        Weights to apply to each scenario
        E.g. for n=3, you could use
        weights = [0.5, 0.25, 0.25]

    Returns
    -------
    np.ndarray, shape=(m, )
        The robustness value for each of the m decision alternatives
    """
    R = np.matmul(f, weights)
    return R


def f_variance(f):
    """Calculate robustness as variance of f

    Parameters
    ----------
    f : np.ndarray, shape=(m, n)
        Transformed performance values to be maximised.
        m decision alternatives and n scenarios

    Returns
    -------
    np.ndarray, shape=(m, )
        The robustness value for each of the m decision alternatives
    """
    # Calculate variance with ddof=1
    # (variance of sample, not population)
    R = np.var(f, axis=1, ddof=1)
    return R


def f_mean_variance(f):
    """Calculate robustness as a combination of mean and variance of f

    Parameters
    ----------
    f : np.ndarray, shape=(m, n)
        Transformed performance values to be maximised.
        m decision alternatives and n scenarios

    Returns
    -------
    np.ndarray, shape=(m, )
        The robustness value for each of the m decision alternatives
    """
    mean_f = f_mean(f)
    # Calculate variance with ddof=1
    # (std deviation of sample, not population)
    std_dev_f = np.std(f, axis=1, ddof=1)
    # +1 is to ensure no divide by 0
    R = np.divide((mean_f + 1), (std_dev_f + 1))
    return R


def f_skew(f, reverse=False):
    """Calculate robustness as the skew of f

    It assumes that it is best to have most values skewed towards
    high performance, with a larger tail for low performance.

    Parameters
    ----------
    f : np.ndarray, shape=(m, 3)
        Transformed performance values to be maximised.
        m decision alternatives and 3 scenarios.
        Those 3 scenarios must be (in order) the 10th, 50th and 90th
        percentiles, where the 10th percentile, q10, is f where only
        10% of f is worse than q10. If wanting the reverse of this,
        see arg 'reverse' below.
    reverse : bool, optional
        Reverses the skew calculation to have a preference for a
        larger tail of higher-performance values.
        (The default is False, which implies it is best to have a skew
        towards high f with a larger tail towards low f).

    Returns
    -------
    np.ndarray, shape=(m, )
        The robustness value for each of the m decision alternatives
    """
    # q10 < q50 < q90
    q10 = f[:, 0]
    q50 = f[:, 1]
    q90 = f[:, 2]
    # If q50 is closer to q90 than q10, most of the values are
    # skewed towards the higher-performance end of f
    # i.e. If q50 > mu50, most values of f are skewed higher
    if not reverse:
        R = q50 - ((q90 + q10) / 2.)
    else:
        R = ((q90 + q10) / 2.) - q50
    normaliser = (q90 - q10) / 2.
    R = np.divide(R, normaliser)
    return R


def f_kurtosis(f):
    """Calculate robustness as the kurtosis of f

    Parameters
    ----------
    f : np.ndarray, shape=(m, 4)
        Transformed performance values.
        m decision alternatives and 4 scenarios.
        Those 4 scenarios must be (in order) the 10th, 25th, 75th and
        90th percentiles, where the 10th percentile, q10, is f where
        only 10% of f is worse than q10.

    Returns
    -------
    np.ndarray, shape=(m, )
        The robustness value for each of the m decision alternatives
    """
    # q10 < q25 < q75 < q90
    q10 = f[:, 0]
    q25 = f[:, 1]
    q75 = f[:, 2]
    q90 = f[:, 3]
    R = np.divide((q90 - q10), (q75 - q25))
    return R
