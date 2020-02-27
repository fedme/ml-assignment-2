import pandas as pd
import matplotlib.pyplot as plt
from knapsack import optimization

PLOTS_FOLDER = 'plots'


def plot_iterations_vs_fitness(size=100):
    for algo in ['rhc', 'sa', 'ga', 'mimic']:
        df = pd.read_csv(f'{optimization.STATS_FOLDER}/{algo}_{size}_stats.csv')
        plt.plot(df['fitness'], label=algo)
    plt.title(f'{optimization.PROBLEM_NAME.capitalize()} - Iterations vs Fitness')
    plt.xlabel('Iterations')
    plt.ylabel('Fitness')
    plt.legend()
    plt.grid()
    plt.savefig(f'{PLOTS_FOLDER}/{optimization.PROBLEM_NAME}_{size}_iterations_vs_fitness.png')
    plt.clf()

    print(f'{optimization.PROBLEM_NAME} iterations vs fitness plotted.')


def plot_iterations_vs_fitness_for_all_sizes():
    for size in optimization.SIZE_VALUES:
        plot_iterations_vs_fitness(size)


if __name__ == '__main__':
    plot_iterations_vs_fitness_for_all_sizes()
