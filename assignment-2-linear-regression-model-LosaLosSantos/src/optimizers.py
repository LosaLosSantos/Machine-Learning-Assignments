from typing import Callable, List

import numpy as np


class GradientDescentOptimizer:
    def __init__(self, learning_rate: float = 0.01, max_iterations: int = 1000, tolerance: float = 1e-6) -> None:
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    def optimize(
        self,
        cost_function: Callable[[np.ndarray], float],
        gradient_function: Callable[[np.ndarray], np.ndarray],
        initial_params: List[float],
    ) -> tuple:
        """
        Performs gradient descent optimization.

        Args:
        cost_function: function that computes the cost
        cost_function: function that computes the gradient of the cost
        initial_params: Initial parameter values

        Returns:
        optimal_params: Optimized parameter values
        cost_history: List of cost values at each iteration (params, cost)
        """
        params = np.array(initial_params)
        cost_history = []  # Keep track of parameters and their cost

        for _ in range(self.max_iterations):
            # Compute the current cost
            cost = cost_function(params)
            cost_history.append((params.copy(), cost))

            # Compute the gradient
            gradient = np.array(gradient_function(params))

            # Update the parameters
            new_params = params - self.learning_rate * gradient

            # Check for convergence (using the norm of the difference)
            if np.linalg.norm(new_params - params) < self.tolerance:
                break

            params = new_params

        return params, cost_history
