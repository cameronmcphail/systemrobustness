"""Contains the T1 functions (performance value transformations).

In general, T1 will take one of 3 forms:
    1. Identity transform - for understanding actual performance
    2. Regret transform - for understanding cost of making wrong decision
    3. Satisficing transform - for understanding how often constraints
       are satisfied.
It is expected that after T1, the aim is to maximise performance.
i.e. Even for the identity transform, if minimising, then values
will be returned as negative.
"""

import numpy as np


def identity(f, maximise=True):
    """Keeps values the same unless minimising.

    If minimising, values are made negative so that the aim
    becomes to maximise performance.

    Parameters
    ----------
    f : np.ndarray, shape=(m, n)
        Performance values, f, for m decision alternatives
        and n scenarios.
    maximise : bool
        Is the performance metric to be maximised or minimised.
        (The default is True, which implies high values of f are better
        than low values of f).

    Returns
    -------
    np.ndarray, shape=(m, n)
        Transformed performance values, f', for m decision alternatives
        and n scenarios
    """
    return f if maximise else -f


def regret_from_best_da(f, maximise=True):
    """T1: Regret from best decision alternative

    Returns negative regret, so that from this point on,
    the aim is to maximise the negative regret (towards 0).

    Parameters
    ----------
    f : np.ndarray, shape=(m, n)
        Performance values, f, for m decision alternatives
        and n scenarios.
    maximise : bool
        Is the performance metric to be maximised or minimised.
        (The default is True, which implies high values of f are better
        than low values of f).

    Returns
    -------
    np.ndarray, shape=(m, n)
        Transformed performance values, f', for m decision alternatives
        and n scenarios
    """
    _f = identity(f, maximise=maximise)
    best_decision_alternatives = np.amax(_f, axis=0)
    regret = _f - best_decision_alternatives
    return regret


def regret_from_values(f, values, maximise=True):
    """T1: Regret from given values

    For a given decision alternative, this function compares its performance in
    each scenario to given performance value for that decision alternative
    Returns negative regret, so that from this point on,
    the aim is to maximise the negative regret (towards 0).

    Parameters
    ----------
    f : np.ndarray, shape=(m, n)
        Performance values, f, for m decision alternatives
        and n scenarios.
    values : np.ndarray, shape=(n, )
        The values to compare the performance values to. i.e. The
        values you would regret not getting, relative to f.
    maximise : bool
        Is the performance metric to be maximised or minimised.
        (The default is True, which implies high values of f are better
        than low values of f).

    Returns
    -------
    np.ndarray, shape=(m, n)
        Transformed performance values, f', for m decision alternatives
        and n scenarios
    """
    regret = np.subtract(f, values)
    # Take into account whether f is to be minimised or maximised.
    regret = identity(regret, maximise=maximise)
    return regret


def regret_from_median(f, maximise=True):
    """T1: Regret from median values

    For a given decision alternative, this function compares its performance in
    each scenario to median performance for that decision alternative

    For a given decision alternative, this function compares its performance in
    each scenario to given performance value for that decision alternative
    Returns negative regret, so that from this point on,
    the aim is to maximise the negative regret (towards 0).

    Parameters
    ----------
    f : np.ndarray, shape=(m, n)
        Performance values, f, for m decision alternatives
        and n scenarios.
    maximise : bool
        Is the performance metric to be maximised or minimised.
        (The default is True, which implies high values of f are better
        than low values of f).

    Returns
    -------
    np.ndarray, shape=(m, n)
        Transformed performance values, f', for m decision alternatives
        and n scenarios
    """
    median_f = np.median(f, axis=1, keepdims=True)
    regret = np.subtract(f, median_f)
    regret = identity(regret, maximise=maximise)
    return regret


def satisfice(f, maximise=True, threshold=0.0, accept_equal=True):
    """Transform performance how many scenarios are satisficed

    Parameters
    ----------
    f : np.ndarray, shape=(m, n)
        Performance values, f, for m decision alternatives
        and n scenarios.
    maximise : bool
        Is the performance metric to be maximised or minimised.
        (The default is True, which implies high values of f are better
        than low values of f).
    threshold : float, optional
        A minimum value where f >= threshold to be satisficed
        (The default is 0.0, which implies that any f value above 0 is
        of satisfactory performance).
    accept_equal : bool, optional
        Whether or not an f value equal to the threshold is acceptable.
        (The default is True, which implies a >= comparison, whereas
        False would imply a > comparison).

    Returns
    -------
    np.ndarray, shape=(m, n)
        Transformed performance values, f', for m decision alternatives
        and n scenarios
    """
    _f = identity(f, maximise=maximise)
    c = threshold if maximise else -threshold
    _f = _f >= c if accept_equal else _f > c
    _f = np.where(_f, 1., 0.)
    return _f
