from timeit import default_timer as timer
import pandas as pd
import mlrose_hiive
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier

SEED = 42
DATA_FOLDER = 'data'
STATS_FOLDER = 'stats'
x_train, y_train, x_test, y_test = None, None, None, None


def load_data_ac():
    global x_train, y_train, x_test, y_test

    # Load csv files
    x_train_df = pd.read_csv(f'{DATA_FOLDER}/adult_census_x_train.csv')
    y_train_df = pd.read_csv(f'{DATA_FOLDER}/adult_census_y_train.csv')
    x_test_df = pd.read_csv(f'{DATA_FOLDER}/adult_census_x_test.csv')
    y_test_df = pd.read_csv(f'{DATA_FOLDER}/adult_census_y_test.csv')

    # Convert features to numpy arrays
    x_train = x_train_df.to_numpy()
    y_train = y_train_df.to_numpy()
    x_test = x_test_df.to_numpy()
    y_test = y_test_df.to_numpy()


def compute_mean_cv_score(classifier, x_train, y_train, cv=5, scoring='accuracy'):
    return cross_val_score(classifier, x_train, y_train, cv=cv, scoring=scoring).mean()


def mlp_backprop(max_iters):
    return MLPClassifier(
        solver='adam',
        max_iter=max_iters,
        learning_rate='adaptive',
        hidden_layer_sizes=(10, 10, 10),
        alpha=0.01,
        activation='tanh',
        random_state=42
    )


def mlp_gd(max_iters):
    return mlrose_hiive.NeuralNetwork(
        hidden_nodes=[10, 10, 10],
        algorithm='gradient_descent',
        activation='tanh',
        learning_rate=0.001,  # 0.1, 0.01, 0.001, 0.0001, 0.00001
        early_stopping=True,
        random_state=SEED,
        max_iters=max_iters
    )


def mlp_sa(max_iters):
    return mlrose_hiive.NeuralNetwork(
        hidden_nodes=[10, 10, 10],
        algorithm='simulated_annealing',
        activation='tanh',
        learning_rate=0.1,  # 0.1, 0.01, 0.001, 0.0001, 0.000001
        max_iters=max_iters,
        early_stopping=True,
        schedule=mlrose_hiive.GeomDecay(),  # ArithDecay, ExpDecay, GeomDecay
        random_state=SEED,
        max_attempts=100  # 10, 100, 500
    )


def mlp_rhc(max_iters):
    return mlrose_hiive.NeuralNetwork(
        hidden_nodes=[10, 10, 10],
        activation='tanh',
        learning_rate=0.01,  # 0.1, 0.01, 0.001, 0.0001, 0.000001
        max_iters=max_iters,
        early_stopping=True,
        random_state=SEED,
        restarts=30,  # 5, 10, ...
        max_attempts=5  # 5, 10, ...
    )


def mlp_ga(max_iters):
    return mlrose_hiive.NeuralNetwork(
        hidden_nodes=[10, 10, 10],
        max_iters=max_iters,
        algorithm='genetic_alg',
        activation='tanh',
        learning_rate=0.001,  # 0.1, 0.01, 0.001, 0.0001, 0.000001
        pop_size=200,  # 100, 150, 200, 300
        mutation_prob=0.1,  # 0.05, 0.1, 0.3
        early_stopping=True,
        random_state=SEED
    )


def get_mlp(algo, max_iters):
    if algo == 'backprop':
        return mlp_backprop(max_iters)
    if algo == 'gd':
        return mlp_gd(max_iters)
    if algo == 'sa':
        return mlp_sa(max_iters)
    if algo == 'rhc':
        return mlp_rhc(max_iters)
    if algo == 'ga':
        return mlp_ga(max_iters)


def train_mlp_and_compute_stats(algo):
    cv = 2
    max_iters_values = [10, 50, 100, 200, 300, 600, 1000]
    mean_cv_scores = []
    times = []

    print(f'Started training MLP with {algo}.')

    # Train for different values of max_iters
    for max_iters in max_iters_values:
        print(f'Run with max_iters={max_iters}')

        # Build MLP
        mlp = get_mlp(algo, max_iters)

        # Train MLP and compute mean CV score
        start = timer()
        mean_cv_score = compute_mean_cv_score(mlp, x_train, y_train, cv=cv, scoring='f1')
        end = timer()
        elapsed_time = (end - start) / cv

        mean_cv_scores.append(mean_cv_score)
        times.append(elapsed_time)

    # Save stats to csv file
    df = pd.DataFrame(index=max_iters_values)
    df['max_iters'] = max_iters_values
    df['mean_cv_score'] = mean_cv_scores
    df['train_time'] = times
    df.to_csv(f'{STATS_FOLDER}/{algo}_stats.csv', index=False)

    print(f'MLP with {algo} stats run finished.')


def quick_test_mlp(algo, max_iters):
    mlp = get_mlp(algo, max_iters)
    print(f'Quick testing MLP [algo={algo}] [max_iters={max_iters}]')
    mean_cv_score = compute_mean_cv_score(mlp, x_train, y_train, cv=2, scoring='f1')
    print(f'MLP [algo={algo}] [max_iters={max_iters}] scored {mean_cv_score}')


if __name__ == '__main__':
    load_data_ac()

    train_mlp_and_compute_stats('ga')
    # quick_test_mlp('ga', 200)
