import numpy as np
import mlrose_hiive

RAW_OUTPUT_DIRECTORY = 'problems_output'
SEED = 42
QUEENS_NAME = 'queens'
MAX_K_COLOR_NAME = 'max_k_color'

# TODO: Use inverse fitness functions and maximize them!!!
# TODO: Use fitness function wrapper to count function invocations


def run_rhc(problem, experiment_name, max_attempts=50):
    rhc = mlrose_hiive.RHCRunner(
        problem=problem,
        experiment_name=experiment_name,
        output_directory=RAW_OUTPUT_DIRECTORY,
        seed=SEED,
        iteration_list=2 ** np.arange(10),
        max_attempts=max_attempts,
        restart_list=[2, 4, 8, 16, 32, 64, 128, 256]
    )
    df_run_stats, df_run_curves = rhc.run()
    return df_run_stats, df_run_curves


def run_sa(problem, experiment_name, max_attempts=50):
    sa = mlrose_hiive.SARunner(problem=problem,
                               experiment_name=experiment_name,
                               output_directory=RAW_OUTPUT_DIRECTORY,
                               seed=SEED,
                               iteration_list=2 ** np.arange(14),
                               max_attempts=max_attempts,
                               temperature_list=[1, 10, 50, 100, 250, 500, 1000, 2500, 5000, 10000])
    df_run_stats, df_run_curves = sa.run()
    return df_run_stats, df_run_curves


def run_ga(problem, experiment_name, max_attempts=50):
    ga = mlrose_hiive.GARunner(
        problem=problem,
        experiment_name=experiment_name,
        output_directory=RAW_OUTPUT_DIRECTORY,
        seed=SEED,
        iteration_list=2 ** np.arange(12),
        max_attempts=max_attempts,
        population_sizes=[150, 200, 300],
        mutation_rates=[0.4, 0.5, 0.6]
    )
    df_run_stats, df_run_curves = ga.run()
    return df_run_stats, df_run_curves


def run_mimic(problem, experiment_name, max_attempts=50):
    mimic = mlrose_hiive.MIMICRunner(
        problem=problem,
        experiment_name=experiment_name,
        output_directory=RAW_OUTPUT_DIRECTORY,
        seed=SEED,
        iteration_list=2 ** np.arange(10),
        max_attempts=max_attempts,
        population_sizes=[10, 200, 500, 1000],
        keep_percent_list=[0.25, 0.5, 0.75]
    )

    df_run_stats, df_run_curves = mimic.run()
    return df_run_stats, df_run_curves


def run_all_queens():
    problem = mlrose_hiive.QueensGenerator.generate(seed=SEED, size=8)

    rhc_curves, rhc_stats = run_rhc(problem, QUEENS_NAME)
    # sa_curves, sa_stats = run_sa(problem, QUEENS_NAME)
    # ga_curves, ga_stats = run_ga(problem, QUEENS_NAME)
    # mimic_curves, mimic_stats = run_mimic(problem, QUEENS_NAME)
    print(QUEENS_NAME + ' was run.')


def run_all_max_k_color():
    problem = mlrose_hiive.MaxKColorGenerator.generate(seed=SEED)

    rhc_curves, rhc_stats = run_rhc(problem, MAX_K_COLOR_NAME, max_attempts=5)
    sa_curves, sa_stats = run_sa(problem, MAX_K_COLOR_NAME)
    ga_curves, ga_stats = run_ga(problem, MAX_K_COLOR_NAME)
    mimic_curves, mimic_stats = run_mimic(problem, MAX_K_COLOR_NAME)
    print(MAX_K_COLOR_NAME + ' was run.')


if __name__ == '__main__':
    # run_all_queens()
    run_all_max_k_color()
