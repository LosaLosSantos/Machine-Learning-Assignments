"""Module implementing Linear Regression using the closed-form solution."""

import numpy as np


class LinearRegression:
    """A class to implement the Linear Regression model."""

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "LinearRegression":
        """
        Fits the Linear Regression model to the input data using the normal equation.

        :param X: A numpy array of shape (n, m) representing the input features.
        :param Y: A numpy array of shape (n,) representing the target values.
        :return: Returns self with fitted parameters.
        """
        n, m = X.shape
        # Adding a column of ones for the intercept term
        theta_0 = np.hstack((np.ones((n, 1)), X))
        # Solution for Linear Regression
        self.theta = np.linalg.inv(theta_0.T @ theta_0) @ theta_0.T @ Y
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the target values for the given input data.

        :param X: A numpy array of shape (k, m) representing the input features.
        :return: A numpy array of shape (k,) representing the predicted values.
        """
        k, m = X.shape
        # Adding a column of ones for the intercept term
        theta_0 = np.hstack((np.ones((k, 1)), X))
        # Return the predicted values
        return theta_0 @ self.theta
