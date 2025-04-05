import numpy as np


class LogisticRegression:
    def __init__(self):
        self.output = None
        self.W = None
        self.b = None

    def train(self, X, y, alpha=1e-2, epochs=1000, threshold=0.5):
        for epoch in range(epochs):
            self.forward(X, threshold)
            self.backward(X, y, alpha)

    def predict(self, X, threshold):
        scores = np.dot(X, self.W) + self.b
        exponent = 1 + np.exp(-scores)
        self.output = 1 / exponent
        return (self.output >= threshold).astype(int)

    def forward(self, X, threshold):
        num_examples, num_features = X.shape[0], X.shape[1]

        if self.W is None or self.b is None:
            self.b = 0
            self.W = np.random.normal(loc=0, scale=1e-1, size=(num_features,))

        self.predict(X, threshold)

    def backward(self, X, y, alpha):
        num_examples = X.shape[0]
        dW = np.dot(X.T, self.output - y) / num_examples
        db = np.sum(self.output - y) / num_examples
        self.W -= alpha * dW
        self.b -= alpha * db

    def evaluate(self, y):
        return -np.sum(y * np.log(self.output), (1-y) * np.log(1-self.output))
