import pandas as pd
import mlrose_hiive
from sklearn.preprocessing import OneHotEncoder

DATA_FOLDER = '../data'
fm_x_train, fm_y_train, fm_x_test, fm_y_test = None, None, None, None


def load_data():
    global fm_x_train, fm_y_train, fm_x_test, fm_y_test

    # Load csv files
    fm_x_train_df = pd.read_csv(f'{DATA_FOLDER}/fashion_mnist_x_train.csv')
    fm_y_train_df = pd.read_csv(f'{DATA_FOLDER}/fashion_mnist_y_train.csv')
    fm_x_test_df = pd.read_csv(f'{DATA_FOLDER}/fashion_mnist_x_test.csv')
    fm_y_test_df = pd.read_csv(f'{DATA_FOLDER}/fashion_mnist_y_test.csv')

    # Convert features to numpy arrays
    fm_x_train = fm_x_train_df.to_numpy()
    fm_x_test = fm_x_test_df.to_numpy()

    # One hot encode target values
    one_hot = OneHotEncoder()
    fm_y_train = one_hot.fit_transform(fm_y_train_df.to_numpy().reshape(-1, 1)).todense()
    fm_y_test = one_hot.transform(fm_y_test_df.to_numpy().reshape(-1, 1)).todense()

    # Check dimensions
    assert fm_x_train.shape[0] == fm_y_train.shape[0] == 6000
    assert fm_x_test.shape[0] == fm_y_test.shape[0] == 1000
    assert fm_x_train.shape[1] == fm_x_test.shape[1] == 784
    assert fm_y_train.shape[1] == fm_y_test.shape[1] == 5

    print()


if __name__ == '__main__':
    load_data()
    print()
