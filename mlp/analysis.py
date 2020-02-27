import pandas as pd
import matplotlib.pyplot as plt

PLOTS_FOLDER = 'plots'
STATS_FOLDER = 'stats'


def plot_iterations_vs_mean_cv_score():
    stats = pd.read_csv(f'{STATS_FOLDER}/backprop_stats.csv', index_col='max_iters', usecols=['max_iters'])
    for algo in ['rhc', 'sa', 'ga', 'gd', 'backprop']:
        df = pd.read_csv(f'{STATS_FOLDER}/{algo}_stats.csv', index_col='max_iters')
        stats[f'{algo.upper()}'] = df['mean_cv_score']

    stats.plot(marker='o')
    plt.title(f'MLP - Iterations vs F1 Score')
    plt.xlabel('Iterations')
    plt.ylabel('Mean CV F1 Score')
    plt.xlim()
    plt.legend()
    plt.grid()
    plt.savefig(f'{PLOTS_FOLDER}/mlp_iterations_vs_f1_score.png')

    print(f'Iterations vs F1 score plotted.')


def plot_iterations_vs_time():
    stats = pd.read_csv(f'{STATS_FOLDER}/backprop_stats.csv', index_col='max_iters', usecols=['max_iters'])
    for algo in ['rhc', 'sa', 'ga', 'gd', 'backprop']:
        df = pd.read_csv(f'{STATS_FOLDER}/{algo}_stats.csv', index_col='max_iters')
        stats[f'{algo.upper()}'] = df['train_time']

    stats.plot(marker='o')
    plt.title(f'MLP - Iterations vs Training time')
    plt.xlabel('Iterations')
    plt.ylabel('Training time (seconds)')
    plt.xlim()
    plt.legend()
    plt.grid()
    plt.savefig(f'{PLOTS_FOLDER}/mlp_iterations_vs_time.png')

    print(f'Iterations vs time plotted.')


if __name__ == '__main__':
    plot_iterations_vs_mean_cv_score()
    plot_iterations_vs_time()
