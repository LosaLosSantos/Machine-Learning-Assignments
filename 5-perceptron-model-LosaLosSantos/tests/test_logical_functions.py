import numpy as np
import pytest

from src.perceptron import Perceptron


@pytest.fixture
def model():
    """
    Fixture providing a fresh Perceptron instance for each test.

    Returns:
        Perceptron: Initialized with learning_rate=0.1 and n_iterations=100
    """
    return Perceptron(learning_rate=0.1, n_iterations=100)


def test_AND(model):
    """
    Test Perceptron's ability to learn AND gate logic.

    Truth table for AND:
        0 0 → 0
        0 1 → 0
        1 0 → 0
        1 1 → 1
    """
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 0, 0, 1])

    model.fit(X, y)
    predictions = model.predict(X)
    assert np.array_equal(predictions, y)


def test_OR(model):
    """
    Test Perceptron's ability to learn OR gate logic.

    Truth table for OR:
        0 0 → 0
        0 1 → 1
        1 0 → 1
        1 1 → 1
    """
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 1])

    model = Perceptron()
    model.fit(X, y)
    predictions = model.predict(X)
    assert np.array_equal(predictions, y)


def test_XOR(model):
    """
    Test Perceptron's limitation with XOR gate logic.

    Truth table for XOR:
        0 0 → 0
        0 1 → 1
        1 0 → 1
        1 1 → 0

    Note: This test is expected to fail as XOR is not linearly separable
    """
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])

    model.fit(X, y)
    predictions = model.predict(X)
    assert not np.array_equal(predictions, y)
