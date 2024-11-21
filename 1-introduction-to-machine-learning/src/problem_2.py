"""Module implementing Ridge Regression using the closed-form solution."""

import numpy as np


class RidgeRegression:
    """A class to implement the Ridge Regression model."""

    def __init__(self, alpha: float = 0.0):
        """
        Initialize the Ridge Regression model.

        :param alpha: The regularization parameter for the Ridge Regression model.
        """
        self.alpha = alpha

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "RidgeRegression":
        """
        Fits the Ridge Regression model to the input data using the closed-form solution.

        :param X: A numpy array of shape (n, m) representing the input features.
        :param Y: A numpy array of shape (n,) representing the target values.
        :return: Returns self with fitted parameters.
        """
        n, m = X.shape
        # Adding a column of ones for the intercept term
        theta_0 = np.hstack((np.ones((n, 1)), X))

        # Identity matrix for regularization, excluding the intercept term
        I = np.eye(m + 1)
        I[0, 0] = 0

        # Solution for Ridge Regression
        self.theta = np.linalg.inv(theta_0.T @ theta_0 + self.alpha * I) @ theta_0.T @ Y
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the target values for the given input data using the fitted Ridge Regression model.

        :param X: A numpy array of shape (k, m) representing the input features.
        :return: A numpy array of shape (k,) representing the predicted values.
        """
        k, m = X.shape
        # Adding a column of ones for the intercept term
        theta_0 = np.hstack((np.ones((k, 1)), X))
        # Return the predicted values
        return theta_0 @ self.theta
