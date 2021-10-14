"""Linear Regression"""

import numpy as np
import pandas as pd
import sklearn.linear_model as sklm
from typing import List, Optional
from utils import min_max_normalize, squared_error

class LinearRegression():
    def __init__(self,
                 alpha: int = 1e-10,
                 num_iter: int = 10000,
                 early_stop: int = 1e-50,
                 intercept: bool = True,
                 init_weight: Optional[np.ndarray] = None
                ):
        """
        Args:
            alpha: learning rate, default=1e-10
            num_iter: number of training iterations, default=10000
            early_stop: early stop error value, default=1e-50
            intercept: fit an intercept, default=True
            init_weight: Nx1 matrix, used for testing, default=None
        """
        self.model_name = 'Linear Regression'
        self.alpha = alpha
        self.num_iter = num_iter
        self.early_stop = early_stop
        self.intercept = intercept
        self.init_weight = init_weight


    def fit(self, X_train, y_train) -> None:
        """Trains the model given training data and labels.

        Args:
            X_train: matrix or 2-D array of input features
            y_train: matrix or 2-D array of target values
        """
        self.X = np.mat(X_train)
        self.y = np.mat(y_train).T

        if self.intercept:
            ones = np.ones(shape=(self.X.shape[0], 1))
            self.X = np.hstack((ones, self.X))

        # randomly initialize weights in [-1, 1]
        self.coef = np.random.uniform(-1, 1, size=(self.X.shape[1], 1))

        # train model
        self.gradient_descent()


    def gradient(self) -> None:
        """Computes gradient for gradient descent for linear regression using squared-error loss.
        """
        y_hat = np.dot(self.X, self.coef)
        self.grad_coef = -2 * np.dot(self.X.T, (self.y - y_hat))


    def gradient_descent(self) -> None:
        """Performs gradient descent to update model weights num_iter times.
        """
        self.loss = []

        for i in range(self.num_iter):

            ones = np.ones(shape=(1, self.y.shape[0]))

            previous_y_hat = np.dot(self.X, self.coef)
            previous_error = np.dot(ones, np.square(self.y - previous_y_hat))

            self.gradient()

            temp_coef = self.coef - (self.alpha * self.grad_coef)

            current_y_hat = np.dot(self.X, temp_coef)
            current_error = np.dot(ones, np.square(self.y - current_y_hat))

            error_diff = abs(previous_error - current_error)
            if error_diff < self.early_stop or abs(error_diff / previous_error) < self.early_stop:
                self.coef = temp_coef
                return

            if current_error <= previous_error:
                self.alpha = self.alpha * 1.3
                self.coef = temp_coef
            else:
                self.alpha = self.alpha * 0.9

            self.loss.append(current_error)

            if i % 10000 == 0:
                print(f'Iteration: {i}, Coef: {self.coef}, Loss: {current_error}')


    def ind_predict(self, X_test: np.ndarray) -> np.ndarray:
        """Returns the predicted output for an individual instance of test input data.

        Args:
            X_test: one instance of X_test, (1 x num_features) array
        """
        return np.array(np.dot(np.mat(X_test), self.coef)).flatten()[0]


    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Returns predicated output for all test input data.

        Args:
            X_test: test input data, (N x num_features) array
        """
        X_test = np.mat(X_test)
        if self.intercept:
            ones = np.ones(shape=(X_test.shape[0], 1))
            X_test = np.hstack((ones, X_test))
        return np.array([self.ind_predict(x) for x in X_test])


if __name__ == '__main__':
    X_train = np.array(np.mat(np.arange(1, 1000, 5)).T)
    # X_train = min_max_normalize(X_train)
    y_train = np.array((30 * X_train)).flatten() + 20

    model = LinearRegression(alpha=1, num_iter=10000000)
    model.fit(X_train, y_train)
    print(model.coef)

    ref_model = sklm.LinearRegression()
    ref_model.fit(X_train, y_train)
    print(ref_model.coef_)