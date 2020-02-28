import pandas as pd
import matplotlib.pyplot as plt
from four_peaks import optimization

PLOTS_FOLDER = 'plots'


def plot_iterations_vs_fitness(size=50):
    for algo in ['rhc', 'sa', 'ga', 'mimic']:
        df = pd.read_csv(f'{optimization.STATS_FOLDER}/{algo}_{size}_stats.csv')
        plt.plot(df['fitness'], label=algo)
    plt.title(f'{optimization.PROBLEM_NAME.capitalize()} with size {size} - Iterations vs Fitness')
    plt.xlabel('Iterations')
    plt.ylabel('Fitness')
    plt.legend()
    plt.grid()
    plt.savefig(f'{PLOTS_FOLDER}/{optimization.PROBLEM_NAME}_size{size}_iterations_vs_fitness.png')
    plt.clf()

    print(f'{optimization.PROBLEM_NAME} iterations vs fitness plotted.')


def plot_iterations_vs_fitness_for_all_sizes():
    for size in optimization.SIZE_VALUES:
        plot_iterations_vs_fitness(size)


def plot_mean_invocations_per_iteration():
    stats = pd.DataFrame()
    for algo in ['rhc', 'sa', 'ga', 'mimic']:
        df = pd.read_csv(f'{optimization.STATS_FOLDER}/{algo}_50_stats.csv')
        stats[f'{algo.upper()}'] = df['n_invocations'] / df.shape[0]

    stats = stats.iloc[0]
    stats.plot.bar(color=['blue', 'orange', 'green', 'red'])
    plt.grid()
    plt.ylabel('Mean number of invocations')
    plt.title(f'{optimization.PROBLEM_NAME.capitalize()} - Fitness function invocations per iteration')
    plt.savefig(f'{PLOTS_FOLDER}/invocations_per_iteration.png')
    plt.clf()
    print(f'Invocations per iteration plotted.')


def plot_time():
    stats = pd.DataFrame()
    for algo in ['rhc', 'sa', 'ga', 'mimic']:
        df = pd.read_csv(f'{optimization.STATS_FOLDER}/{algo}_50_stats.csv')
        stats[f'{algo.upper()}'] = df['time']

    stats = stats.iloc[0]
    stats.plot.bar(color=['blue', 'orange', 'green', 'red'])
    plt.grid()
    plt.ylabel('Elapsed time in seconds')
    plt.title(f'{optimization.PROBLEM_NAME.capitalize()} - Time to solve problem')
    plt.savefig(f'{PLOTS_FOLDER}/time_to_solve_problem.png')
    plt.clf()
    print(f'Time to solve problem plotted.')


if __name__ == '__main__':
    plot_iterations_vs_fitness_for_all_sizes()
    plot_mean_invocations_per_iteration()
    plot_time()
