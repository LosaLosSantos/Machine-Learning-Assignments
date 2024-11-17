import numpy as np
import pytest
from sklearn.linear_model import LinearRegression as SklearnLinearRegression

from src.problem_1 import LinearRegression


def test_lr_one_dimensional():
    X = np.array([1, 3, 2, 5]).reshape((4, 1))
    Y = np.array([2, 5, 3, 8])
    a = np.array([1, 2, 10]).reshape((3, 1))

    expected = SklearnLinearRegression().fit(X, Y).predict(a)
    actual = LinearRegression().fit(X, Y).predict(a)

    assert list(actual) == pytest.approx(list(expected))


def test_ridge_multi_dimensional():
    X = np.array([1, 2, 3, 5, 4, 5, 4, 3, 3, 3, 2, 5]).reshape((4, 3))
    Y = np.array([2, 5, 3, 8])
    a = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1, 2, 5, 7, -2, 0, 3]).reshape((5, 3))

    expected = SklearnLinearRegression().fit(X, Y).predict(a)
    actual = LinearRegression().fit(X, Y).predict(a)

    assert list(actual) == pytest.approx(list(expected))
