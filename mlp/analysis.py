import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mlp import mlp_training
from sklearn.metrics import plot_confusion_matrix

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
    plt.clf()

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
    plt.clf()

    print(f'Iterations vs time plotted.')


def plot_mlp_confusion_matrix(algo, clf, x_test, y_test):
    # Plot non-normalized confusion matrix
    print(f'Plotting confusion matrix for {algo} MLP...')
    np.set_printoptions(precision=2)
    disp = plot_confusion_matrix(clf, x_test, y_test,
                                 display_labels=['< 50k', '> 50k'],
                                 cmap=plt.cm.get_cmap('Blues'),
                                 normalize='true')
    disp.ax_.set_title(f'{algo.capitalize()} - Normalized confusion matrix')
    plt.savefig(f'{PLOTS_FOLDER}/mlp_{algo}_confusion_matrix.png')
    plt.clf()


def plot_confusion_matrices():
    mlp_training.load_data_ac()
    x_train = mlp_training.x_train
    y_train = mlp_training.y_train
    x_test = mlp_training.x_test
    y_test = mlp_training.y_test

    # algo = 'backprop'
    # print(f'Training MLP with {algo}...')
    # mlp = mlp_training.get_mlp(algo, 600)
    # mlp.fit(x_train, y_train)
    # plot_mlp_confusion_matrix(algo, mlp, x_test, y_test)

    # algo = 'gd'
    # print(f'Training MLP with {algo}...')
    # mlp = mlp_training.get_mlp(algo, 200)
    # mlp.fit(x_train, y_train)
    # plot_mlp_confusion_matrix(algo, mlp, x_test, y_test)

    # algo = 'ga'
    # print(f'Training MLP with {algo}...')
    # mlp = mlp_training.get_mlp(algo, 200)
    # mlp.fit(x_train, y_train)
    # plot_mlp_confusion_matrix(algo, mlp, x_test, y_test)

    # algo = 'rhc'
    # print(f'Training MLP with {algo}...')
    # mlp = mlp_training.get_mlp(algo, 100)
    # mlp.fit(x_train, y_train)
    # plot_mlp_confusion_matrix(algo, mlp, x_test, y_test)

    algo = 'sa'
    print(f'Training MLP with {algo}...')
    mlp = mlp_training.get_mlp(algo, 200)
    mlp.fit(x_train, y_train)
    plot_mlp_confusion_matrix(algo, mlp, x_test, y_test)


if __name__ == '__main__':
    # plot_iterations_vs_mean_cv_score()
    # plot_iterations_vs_time()
    plot_confusion_matrices()
