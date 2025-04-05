import numpy as np
from model import LogisticRegression
from data import fetch_and_format_dataset, test_train_split


if __name__ == '__main__':
    data = fetch_and_format_dataset()
    X, X_test, y, y_test = test_train_split(data)

    model = LogisticRegression()
    threshold = 0.4
    alpha = 1e-4
    epochs = 350

    model.train(X, y, alpha, epochs)

    training_results = model.predict(X, threshold)
    test_results = model.predict(X_test, threshold)

    training_accuracy = np.round(np.mean(training_results == y), 3)
    test_accuracy = np.round(np.mean(test_results == y_test), 3)

    print(training_accuracy)
    print(test_accuracy)
