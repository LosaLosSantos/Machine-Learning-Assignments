from typing import Callable, List

import numpy as np


class MomentumGradientDescentOptimizer:
    def __init__(
        self, learning_rate: float = 0.01, max_iterations: int = 1000, tolerance: float = 1e-6, momentum: float = 0.9
    ) -> None:
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.momentum = momentum

    def optimize(
        self,
        cost_function: Callable[[np.ndarray], float],
        gradient_function: Callable[[np.ndarray], np.ndarray],
        initial_params: List[float],
    ) -> tuple:
        """
        Performs Momentum Gradient Descent optimization.

        Args:
        cost_function: function that computes the cost
        gradient_function: function that computes the gradient of the cost
        initial_params: Initial parameter values

        Returns:
        optimal_params: Optimized parameter values
        cost_history: List of cost values at each iteration (params, cost)
        """
        params = np.array(initial_params)
        cost_history = []  # Keep track of parameters and their cost
        velocity = np.zeros_like(params)

        for _ in range(self.max_iterations):
            # Compute the current cost
            cost = cost_function(params)
            cost_history.append((params.copy(), cost))

            # Compute the gradient
            gradient = np.array(gradient_function(params))

            # vt is the velocity vector
            velocity = self.momentum * velocity + (1 - self.momentum) * gradient

            # Update the parameters
            new_params = params - self.learning_rate * velocity

            # Check for convergence (using the norm of the difference)
            if np.linalg.norm(new_params - params) < self.tolerance:
                break

            params = new_params

        return params, cost_history


class NesterovAcceleratedGradientOptimizer:
    def __init__(
        self, learning_rate: float = 0.01, max_iterations: int = 1000, tolerance: float = 1e-6, momentum: float = 0.9
    ) -> None:
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.momentum = momentum

    def optimize(
        self,
        cost_function: Callable[[np.ndarray], float],
        gradient_function: Callable[[np.ndarray], np.ndarray],
        initial_params: List[float],
    ) -> tuple:
        """
        Performs Nesterov Accelerated Gradient Descent optimization.

        Args:
        cost_function: function that computes the cost
        gradient_function: function that computes the gradient of the cost
        initial_params: Initial parameter values

        Returns:
        optimal_params: Optimized parameter values
        cost_history: List of cost values at each iteration (params, cost)
        """
        params = np.array(initial_params)
        cost_history = []  # Keep track of parameters and their cost
        velocity = np.zeros_like(params)

        for _ in range(self.max_iterations):
            # Compute the current cost
            cost = cost_function(params)
            cost_history.append((params.copy(), cost))

            # Compute the gradient
            gradient = np.array(gradient_function(params - self.learning_rate * self.momentum * velocity))

            # vt is the velocity vector
            velocity = self.momentum * velocity + (1 - self.momentum) * gradient

            # Update the parameters
            new_params = params - self.learning_rate * velocity

            # Check for convergence (using the norm of the difference)
            if np.linalg.norm(new_params - params) < self.tolerance:
                break

            params = new_params

        return params, cost_history


class AdaptiveGradientOptimizer:
    def __init__(
        self, learning_rate: float = 0.01, max_iterations: int = 1000, tolerance: float = 1e-6, e: float = 1e-8
    ) -> None:
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.e = e

    def optimize(
        self,
        cost_function: Callable[[np.ndarray], float],
        gradient_function: Callable[[np.ndarray], np.ndarray],
        initial_params: List[float],
    ) -> tuple:
        """
        Performs Adaptive Gradient Descent optimization.

        Args:
        cost_function: function that computes the cost
        gradient_function: function that computes the gradient of the cost
        initial_params: Initial parameter values

        Returns:
        optimal_params: Optimized parameter values
        cost_history: List of cost values at each iteration (params, cost)
        """
        params = np.array(initial_params)
        cost_history = []  # Keep track of parameters and their cost
        G = np.zeros_like(params)

        for _ in range(self.max_iterations):
            # Compute the current cost
            cost = cost_function(params)
            cost_history.append((params.copy(), cost))

            # Compute the gradient
            gradient = np.array(gradient_function(params))

            # Sum of squared gradients up to time t
            G = G + gradient**2

            # Update the parameters
            new_params = params - ((self.learning_rate) / (np.sqrt(G + self.e))) * gradient

            # Check for convergence (using the norm of the difference)
            if np.linalg.norm(new_params - params) < self.tolerance:
                break

            params = new_params

        return params, cost_history


class RMSpropOptimizer:
    def __init__(
        self,
        learning_rate: float = 0.01,
        max_iterations: int = 1000,
        tolerance: float = 1e-6,
        e: float = 1e-8,
        beta: float = 0.9,
    ) -> None:
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.e = e
        self.beta = beta

    def optimize(
        self,
        cost_function: Callable[[np.ndarray], float],
        gradient_function: Callable[[np.ndarray], np.ndarray],
        initial_params: List[float],
    ) -> tuple:
        """
        Performs RMSprop optimization.

        Args:
        cost_function: function that computes the cost
        gradient_function: function that computes the gradient of the cost
        initial_params: Initial parameter values

        Returns:
        optimal_params: Optimized parameter values
        cost_history: List of cost values at each iteration (params, cost)
        """
        params = np.array(initial_params)
        cost_history = []  # Keep track of parameters and their cost
        E = np.zeros_like(params)

        for _ in range(self.max_iterations):
            # Compute the current cost
            cost = cost_function(params)
            cost_history.append((params.copy(), cost))

            # Compute the gradient
            gradient = np.array(gradient_function(params))

            # Exponential moving average of squared gradients
            E = self.beta * E + (1 - self.beta) * gradient**2

            # Update the parameters
            new_params = params - ((self.learning_rate) / (np.sqrt(E + self.e))) * gradient

            # Check for convergence
            if np.linalg.norm(new_params - params) < self.tolerance:
                break

            params = new_params

        return params, cost_history


class AdaptiveMomentEstimationOptimizer:
    def __init__(
        self,
        learning_rate: float = 0.01,
        max_iterations: int = 1000,
        tolerance: float = 1e-6,
        e: float = 1e-8,
        b1: float = 0.9,
        b2: float = 0.999,
    ) -> None:
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.e = e
        self.b1 = b1
        self.b2 = b2

    def optimize(
        self,
        cost_function: Callable[[np.ndarray], float],
        gradient_function: Callable[[np.ndarray], np.ndarray],
        initial_params: List[float],
    ) -> tuple:
        """
        Performs Adaptive Moment Estimation optimization.

        Args:
        cost_function: function that computes the cost
        gradient_function: function that computes the gradient of the cost
        initial_params: Initial parameter values

        Returns:
        optimal_params: Optimized parameter values
        cost_history: List of cost values at each iteration (params, cost)
        """
        params = np.array(initial_params)
        cost_history = []  # Keep track of parameters and their cost
        m = np.zeros_like(params)  # first moment estimate (mean of gradients)
        v = np.zeros_like(params)  # second moment estimate (variance of gradients)

        for t in range(self.max_iterations):
            # Compute the current cost
            cost = cost_function(params)
            cost_history.append((params.copy(), cost))

            # Compute the gradient
            gradient = np.array(gradient_function(params))

            # Moments estimate
            m = self.b1 * m + (1 - self.b1) * gradient
            v = self.b2 * v + (1 - self.b2) * gradient**2

            # mt and vt are bias-corrected moment estimates
            mt = m / (1 - self.b1 ** (t + 1))
            vt = v / (1 - self.b2 ** (t + 1))

            # Update the parameters
            new_params = params - ((self.learning_rate) / (np.sqrt(vt) + self.e)) * mt

            # Check for convergence
            if np.linalg.norm(new_params - params) < self.tolerance:
                break

            params = new_params

        return params, cost_history
