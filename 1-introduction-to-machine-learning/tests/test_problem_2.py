import numpy as np
import pytest
from sklearn.linear_model import Ridge

from src.problem_1 import LinearRegression
from src.problem_2 import RidgeRegression


def test_ridge_one_dimensional():
    X = np.array([1, 3, 2, 5]).reshape((4, 1))
    Y = np.array([2, 5, 3, 8])

    alpha = 0.3
    X_test = np.array([1, 2, 10]).reshape((3, 1))
    expected = Ridge(alpha).fit(X, Y).predict(X_test)

    actual = RidgeRegression(alpha).fit(X, Y).predict(X_test)
    assert list(actual) == pytest.approx(list(expected), rel=1e-5)


def test_ridge_multi_dimensional():
    X = np.array([1, 2, 3, 5, 4, 5, 4, 3, 3, 3, 2, 5]).reshape((4, 3))
    Y = np.array([2, 5, 3, 8])

    X_test = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1, 2, 5, 7, -2, 0, 3]).reshape((5, 3))

    alpha = 0.4
    expected = Ridge(alpha).fit(X, Y).predict(X_test)

    actual = RidgeRegression(alpha).fit(X, Y).predict(X_test)
    assert list(actual) == pytest.approx(list(expected), rel=1e-3)


def test_ridge_0_lambda():
    X = np.array([1, 3, 2, 5]).reshape((4, 1))
    Y = np.array([2, 5, 3, 8])

    alpha = 0
    X_test = np.array([1, 2, 10]).reshape((3, 1))
    expected = Ridge(alpha).fit(X, Y).predict(X_test)

    actual = RidgeRegression(alpha).fit(X, Y).predict(X_test)
    assert list(actual) == pytest.approx(list(expected), rel=1e-5)


def test_ridge_0_local_lambda():
    X = np.array([1, 3, 2, 5]).reshape((4, 1))
    Y = np.array([2, 5, 3, 8])

    alpha = 0
    X_test = np.array([1, 2, 10]).reshape((3, 1))
    expected = LinearRegression().fit(X, Y).predict(X_test)

    actual = RidgeRegression(alpha).fit(X, Y).predict(X_test)
    assert list(actual) == pytest.approx(list(expected), rel=1e-5)
