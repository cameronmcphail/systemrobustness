"""Implements example from EM Workbench package.

Example contained in:

Kwakkel, J.H., 2017. The Exploratory Modeling Workbench: An open source
toolkit for exploratory modeling, scenario discovery, and
(multi-objective) robust decision making. Environ. Model. Softw. 96,
239–250. https://doi.org/10.1016/j.envsoft.2017.06.054

Altered to use custom robustness metrics from the System Robustness package.
"""

import functools
import numpy as np
from ema_workbench import (
    MultiprocessingEvaluator, Model, RealParameter, Constant, ScalarOutcome)
from ema_workbench.em_framework.samplers import sample_uncertainties
from ema_workbench.examples.lake_model import lake_problem
from systemrobustness.robustness.metrics import t1, t2, t3, custom_R_metric


def get_custom_R_metrics():
    """Returns the custom robustness metrics similar to original example."""
    # A metric to minimise the average max_P across all scenarios
    laplace_principle = functools.partial(
        custom_R_metric(t1.identity, t2.all_scenarios, t3.f_mean),
        maximise=False)
    # A metric to minimise the range of values of max_P across all scenarios
    # Note that this is different to the original example's metric which was
    # the standard deviation.
    range_metric = functools.partial(
        custom_R_metric(t1.identity, t2.all_scenarios, t3.f_range),
        maximise=False)
    # A metric to maximise the average reliability and minimise
    # the variance in reliability. Note that this metric is
    # found in the literature but is not included in the guidance
    # from the System Robustness paper. Despite this, the System Robustness
    # package can still create the metric.
    mean_variance = functools.partial(
        custom_R_metric(t1.identity, t2.all_scenarios, t3.f_mean_variance),
        maximise=True)
    # A metric to maximise the reliability, but is risk-averse, since
    # it looks at the 10th percentile (i.e. the scenario where only 10%
    # of reliabilities are lower)
    percentile10 = functools.partial(
        custom_R_metric(t1.identity, t2.select_percentiles, t3.f_mean),
        maximise=True,
        t2_kwargs={'percentiles': [0.1]})

    MAXIMIZE = ScalarOutcome.MAXIMIZE
    MINIMIZE = ScalarOutcome.MINIMIZE

    # Note that we want to minimise max_P, so we define this in the
    # robustness metrics above (maximise=False), and this changes
    # the sign of the robustness metric, so that we can always
    # make the objective to MAXIMIZE robustness.
    robustness_functions = [
        ScalarOutcome(
            'mean p',
            kind=MAXIMIZE,
            variable_name='max_P',
            function=laplace_principle),
        ScalarOutcome(
            'range p',
            kind=MAXIMIZE,
            variable_name='max_P',
            function=range_metric),
        ScalarOutcome(
            'sn reliability',
            kind=MAXIMIZE,
            variable_name='reliability',
            function=mean_variance),
        ScalarOutcome(
            '10th percentile utility',
            kind=MAXIMIZE,
            variable_name='reliability',
            function=percentile10)]

    return robustness_functions


def get_original_R_metrics():
    """Returns the Robustness metrics from original example."""
    # a percentile-based minimax robustness function
    percentile10 = functools.partial(np.percentile, q=10)
    MAXIMIZE = ScalarOutcome.MAXIMIZE
    MINIMIZE = ScalarOutcome.MINIMIZE

    robustness_functions = [
        ScalarOutcome(
            'mean p',
            kind=MINIMIZE,
            variable_name='max_P',
            function=np.mean),
        ScalarOutcome(
            'std p',
            kind=MINIMIZE,
            variable_name='max_P',
            function=np.std),
        ScalarOutcome(
            'sn reliability',
            kind=MAXIMIZE,
            variable_name='reliability',
            function=signal_to_noise),
        ScalarOutcome(
            '10th percentile utility',
            kind=MAXIMIZE,
            variable_name='reliability',
            function=percentile10)]

    return robustness_functions


def signal_to_noise(data):
    """A robustness metric defined for the original example."""
    mean = np.mean(data)
    std = np.std(data)
    sn = mean/std
    return sn


def get_lake_model():
    """Returns a fully formulated model of the lake problem."""
    # instantiate the model
    lake_model = Model('lakeproblem', function=lake_problem)
    lake_model.time_horizon = 100

    # specify uncertainties
    lake_model.uncertainties = [RealParameter('b', 0.1, 0.45),
                                RealParameter('q', 2.0, 4.5),
                                RealParameter('mean', 0.01, 0.05),
                                RealParameter('stdev', 0.001, 0.005),
                                RealParameter('delta', 0.93, 0.99)]

    # set levers, one for each time step
    lake_model.levers = [RealParameter(str(i), 0, 0.1) for i in
                         range(lake_model.time_horizon)]

    # specify outcomes
    lake_model.outcomes = [ScalarOutcome('max_P',),
                           ScalarOutcome('utility'),
                           ScalarOutcome('inertia'),
                           ScalarOutcome('reliability')]

    # override some of the defaults of the model
    lake_model.constants = [Constant('alpha', 0.41),
                            Constant('nsamples', 150)]
    return lake_model


def optimize_lake_problem(use_original_R_metrics=False):
    robustness_functions = (
        get_original_R_metrics()
        if use_original_R_metrics
        else get_custom_R_metrics())

    lake_model = get_lake_model()

    n_scenarios = 10  # for demo purposes only, should in practice be higher
    scenarios = sample_uncertainties(lake_model, n_scenarios)
    nfe = 1000

    with MultiprocessingEvaluator(lake_model) as evaluator:
        robust_results = evaluator.robust_optimize(
            robustness_functions,
            scenarios,
            nfe=nfe,
            population_size=25,
            epsilons=[0.1,] * len(robustness_functions))
    print(robust_results)

if __name__ == '__main__':
    optimize_lake_problem(
        use_original_R_metrics=False)
