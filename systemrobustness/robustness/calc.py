"""Helper functions for calculating robustness from performance values"""

import numpy as np
import pandas as pd

from metrics import t1, t2, t3, custom_R_metric, guidance_to_R


def f_to_R(f_df, R_dict):
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
        Must include any other variables to be used during calculation
        of robustness (referred to as 'varX_name' below).
        Columns: `['s_idx', 'l_idx', '<f1_name>', '<f2_name>', ...,
                   '<var1_name>', '<var2_name>', ...]`
    R_dict : dict of dict
        A mapping of robustness metric (`R`) names to information
        about those robustness metrics including
        'f': string
            the corresponding performance metric to use
        'maximise': bool
            whether the aim of that performance metric is to be maximised
        'func': func
            the robustness metric function
        'kws': list of string
            a list of keyword arguments required for calculating R
        Note that all performance metric names must be listed here.
        E.g. `{'<R1_name>': {'f': <f1_name>, 'maximise': <bool>, 'func': <func>, 'kws': [<kw1>, <kw2>, ...]},
               '<R2_name>': {'f': <f1_name>, 'maximise': <bool>, 'func': <func>, 'kws': [<kw1>, <kw2>, ...]}, ...}`

    Returns
    -------
    pandas.DataFrame
        A dataframe of robustness values, `R`, with indexes for the
        decision alternative, `l`, and a column for the performance
        metric name, ``f_name``.
        Columns: `['l_idx', 'f_name', '<R1_name>', '<R2_name>', ...]`
    """
    # Extract the names of the f metrics
    f_metrics = [R_dict[R_name]['f'] for R_name in R_dict]
    df_cols = [col for col in f_df.columns]
    # Check that the same metrics are in f_maximise
    for f_metric in f_metrics:
        assert f_metric in df_cols

    # Sort dataframe to ensure consistency
    f_df = sort_f_df(f_df)

    # Get the scenario and decision alternative idxs, and
    # check that the s_idx and l_idx indexes are valid.
    s_idxs, l_idxs = get_f_df_details(f_df)

    # Loop through performance metrics
    R = {}
    for R_metric in R_dict:
        f_metric = R_dict[R_metric]['f']
        # Check that required data exists
        kwargs = {}
        for kw in R_dict[R_metric]['kws']:
            assert kw in df_cols
            kwargs[kw] = np.reshape(
                f_df.iloc[:, f_df.columns.get_loc(kw)].values,
                newshape=(l_idxs.size, s_idxs.size))
        f = np.reshape(
            f_df.iloc[:, f_df.columns.get_loc(f_metric)].values,
            newshape=(l_idxs.size, s_idxs.size))
        kwargs['maximise'] = R_dict[R_metric]['maximise']
        R[R_metric] = R_dict[R_metric]['func'](f, **kwargs)
    return R



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
    f_df.sort_values(['l_idx', 's_idx'], ascending=[True, True])
    return f_df


def get_f_df_details(f_df):
    """Gets the unique s_idx and l_idx values in f_df.

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
    s_idxs : list of int
        Number of scenario (`s`) idxs
    l_idxs : list of int
        List decision alternative (`l`) idxs
    """
    # Check that df is sorted
    f_df = sort_f_df(f_df)

    s_idxs = f_df['s_idx'].unique()
    l_idxs = f_df['l_idx'].unique()

    for s_idx in s_idxs:
        relevant_rows = f_df.loc[f_df['s_idx'] == s_idx]
        relevant_l_idxs = relevant_rows['l_idx'].values
        assert np.allclose(relevant_l_idxs, l_idxs)

    return s_idxs, l_idxs


if __name__ == '__main__':
    df = pd.DataFrame.from_dict({
        's_idx': [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2],
        'l_idx': [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
        'return': [-4, 4, 12, -2, 3, 8, 3, 2, 1, 3, 3, 3]
    })
    info = {
        'Maximin': {
            'f': 'return',
            'maximise': True,
            'func': custom_R_metric(t1.identity, t2.worst_case, t3.f_mean),
            'kws': []},
        'Minimax regret': {
            'f': 'return',
            'maximise': True,
            'func': custom_R_metric(t1.regret_from_best_da, t2.worst_case, t3.f_mean),
            'kws': []},
        'Custom R': {
            'f': 'return',
            'maximise': True,
            'func': guidance_to_R(),
            'kws': []}
    }
    R = f_to_R(df, info)

    x = 42
