import numpy as np
import pandas as pd


class LinearRegressionScratch:
    """
    Author @Abishek J
    A simple linear regression model that uses gradient descent for optimization.
    weights -> The weights of the model.
    bias    -> The bias of the model.
    sample  -> The number of rows in the dataset.
    feature ->The number of columns in the dataset.
    dw      -> The gradient of the loss function with respect to the weights.
    db      -> The gradient of the loss function with respect to the bias.

    """

    def __init__(self):
        self.weights = None
        self.bias = None

    def fit(self, x, y, learning_rate=0.01, epochs=1500):
        sample, feature = x.shape
        self.weights = np.zeros(feature)
        self.bias = 0

        for n in range(epochs):
            y_pred = np.dot(x, self.weights) + self.bias

            dw = (1 / sample) * np.dot(x.T, (y_pred - y))
            db = (1 / sample) * np.sum(y_pred - y)

            self.weights -= learning_rate * dw
            self.bias -= learning_rate * db

    def predict(self, x):
        y_pred = np.dot(x, self.weights) + self.bias
        return y_pred
