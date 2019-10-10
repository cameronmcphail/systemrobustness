"""Helper functions for calculating robustness from performance values"""

import numpy as np


def f_to_R(f_df, f_maximise):
    """Calculates robustness from performance values.

    Uses a set of performance values, `f`, determined from simulations
    across multiple decision alternatives, `l`, different scenarios,
    `s`, and calculates robustness, `R`, using a variety of given
    robustness metrics.

    Parameters
    ----------
    f_df : pandas.DataFrame
        A dataframe of performance values, `f`, with indexes for the
        scenario, `s`, and decision alternative, `l`.
        Columns: `['s_idx', 'l_idx', '<f1_name>', '<f2_name>', ...]`
    f_maximise : dict
        A mapping of performance metric (f) names to whether the aim
        of that metric is to be maximised.
        E.g. `{'<f1_name>': bool, '<f2_name>': bool, ...}`

    Returns
    -------
    pandas.DataFrame
        A dataframe of robustness values, `R`, with indexes for the
        decision alternative, `l`, and a column for the performance
        metric name, ``f_name``.
        Columns: `['l_idx', 'f_name', '<R1_name>', '<R2_name>', ...]`
    """
    # Extract the names of the f metrics
    f_metrics = [col for col in f_df.columns if col not in ['s_idx', 'l_idx']]
    # Check that the same metrics are in f_maximise
    for f_metric in f_metrics:
        assert f_metric in f_maximise

    # Sort dataframe to ensure consistency
    f_df = sort_f_df(f_df)

    # Get the number of scenarios and decision alternatives, and
    # check that the s_idx and l_idx indexes are valid.
    n_s, n_l = get_f_df_details(f_df)
    
    # Loop through performance metrics
    for f_metric in f_metrics:
        x = 42


def sort_f_df(f_df):
    """Sorts f_df by s_idx first then by l_idx.

    E.g. for scenario 0, see all decision alternatives in order,
    then scenario 1, scenario 2, etc.

    Parameters
    ----------
    f_df : pandas.DataFrame
        A dataframe of performance values, `f`, with indexes for the
        scenario, `s`, and decision alternative, `l`.
        Columns: `['s_idx', 'l_idx', '<f1_name>', '<f2_name>', ...]`
    """
    # This will sort first by s_idx then by l_idx, both from 0 to ...
    f_df.sort_values(['s_idx', 'l_idx'], ascending=[True, True])
    return f_df


def get_f_df_details(f_df):
    """Gets the number of unique s_idx and l_idx values in f_df.

    Also checks that for each s_idx, each unique l_idx exists
    (and vice versa).
    
    Parameters
    ----------
    f_df : pandas.DataFrame
        A dataframe of performance values, `f`, with indexes for the
        scenario, `s`, and decision alternative, `l`.
        Columns: `['s_idx', 'l_idx', '<f1_name>', '<f2_name>', ...]`

    Returns
    -------
    n_s : int
        Number of scenarios, `s`
    n_l : int
        Number of decision alternatives, `l`
    """
    # Check that df is sorted
    f_df = sort_f_df(f_df)

    s_idxs = f_df['s_idx'].unique()
    l_idxs = f_df['l_idx'].unique()

    for s_idx in s_idxs:
        relevant_rows = f_df.loc[f_df['s_idx'] == s_idx]
        relevant_l_idxs = relevant_rows['l_idx'].values
        assert np.all_close(relevant_l_idxs, l_idxs)

    n_s = len(s_idxs)
    n_l = len(l_idxs)
    return n_s, n_l
