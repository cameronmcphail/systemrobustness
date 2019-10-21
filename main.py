from systemrobustness.examples.investment_simple import investment_simple_example
from systemrobustness.examples.em_workbench_lake_model import optimize_lake_problem

if __name__ == '__main__':
    #investment_simple_example()
    optimize_lake_problem(
        use_original_R_metrics=False)
