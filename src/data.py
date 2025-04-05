import math
import numpy as np
from sklearn.datasets import load_breast_cancer


def fetch_and_format_dataset():
    dataset = load_breast_cancer(as_frame=True).frame
    dataset.columns = [feature.replace(' ', '_') for feature in dataset.columns]
    return dataset


def min_max_scaling(dataset):
    copy = dataset.copy()
    maximum = np.max(copy, axis=0)
    minimum = np.min(copy, axis=0)

    scaled_dataset = np.subtract(copy, minimum)/np.subtract(maximum, minimum)
    return scaled_dataset


def test_train_split(dataset, test_perc=0.2, random_state=2025):
    # Set seed for reproducible shuffling
    np.random.seed(random_state)

    # Shuffle the dataset randomly
    shuffled_dataset = np.random.permutation(dataset)

    # Calculate split
    test_size = math.ceil(test_perc * shuffled_dataset.shape[0])

    # Split the dataset into training and test set
    X_test = shuffled_dataset[:test_size, :-1]
    y_test = shuffled_dataset[:test_size, -1:].reshape((-1,)).astype(int)

    X = shuffled_dataset[test_size:, :-1]
    y = shuffled_dataset[test_size:, -1:].reshape((-1,)).astype(int)

    return X, X_test, y, y_test


if __name__ == '__main__':
    data = min_max_scaling(fetch_and_format_dataset())
    Xtr, Xte, ytr, yte = test_train_split(data)

    print(f"Training Data Shape: {Xtr.shape}")
    print(f"Training Labels Shape: {ytr.shape}")
    print(f"Test Data Shape: {Xte.shape}")
    print(f"Test Labels Shape: {yte.shape}")
    print(data)
    print(data)
