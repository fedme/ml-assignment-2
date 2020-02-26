import pandas as pd
import matplotlib.pyplot as plt
from continuous_peaks import optimization


def plot_iterations_vs_fitness():
    for algo in ['rhc', 'sa', 'ga', 'mimic']:
        df = pd.read_csv(f'{algo}_stats.csv')
        plt.plot(df['fitness'], label=algo)
    plt.title(f'{optimization.PROBLEM_NAME.capitalize()} - Iterations vs Fitness')
    plt.xlabel('Iterations')
    plt.ylabel('Fitness')
    plt.xlim()
    plt.legend()
    plt.grid()
    plt.savefig(f'{optimization.PROBLEM_NAME}_iterations_vs_fitness.png')

    print(f'{optimization.PROBLEM_NAME} iterations vs fitness plotted.')


if __name__ == '__main__':
    plot_iterations_vs_fitness()
