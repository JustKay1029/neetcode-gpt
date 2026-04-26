import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        epsilon = 1e-7
        pred = np.clip(y_pred, epsilon, 1 - epsilon)
        k = 0
        for i in range(len(y_true)):
            k += y_true[i]*np.log(pred[i]) + (1 - y_true[i])*np.log(1 - pred[i])
        ans = -k/len(y_true)
        return round(ans,4)

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        epsilon = 1e-7
        pred = np.clip(y_pred, epsilon, 1 - epsilon)
        k = 0 
        rows, cols = y_true.shape
        for i in range(rows):  
            for j in range(cols):
                k += y_true[i,j]*np.log(pred[i,j])
        ans = -k/rows
        return round(ans,4)
