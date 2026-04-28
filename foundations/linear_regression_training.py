import numpy as np
from numpy.typing import NDArray

class Solution:
    def get_derivative(
        self,
        model_prediction: NDArray[np.float64],
        ground_truth: NDArray[np.float64],
        N: int,
        X: NDArray[np.float64],
        desired_weight: int
    ) -> float:
        # note that N is just len(X)
        return -2 * np.dot(ground_truth - model_prediction, X[:, desired_weight]) / N

    def get_model_prediction(
        self,
        X: NDArray[np.float64],
        weights: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return np.squeeze(np.matmul(X, weights))

    learning_rate = 0.01

    def train_model(
        self,
        X: NDArray[np.float64],
        Y: NDArray[np.float64],
        num_iterations: int,
        initial_weights: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        # make a float copy of the weights
        weights = np.array(initial_weights, dtype=np.float64)

        N = len(X)

        for _ in range(num_iterations):
            # 1. predictions for current weights
            preds = self.get_model_prediction(X, weights)

            # 2. update each weight separately
            for j in range(len(weights)):
                grad_j = self.get_derivative(preds, Y, N, X, j)
                weights[j] -= self.learning_rate * grad_j

        return np.round(weights, 5)