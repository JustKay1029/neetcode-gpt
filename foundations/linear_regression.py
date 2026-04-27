import numpy as np
from numpy.typing import NDArray

class Solution:

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        # X is (n, m), weights is (m,) -> return (n,) predictions
        # Round to 5 decimal places
        predictions = []
        for row in X:
            dot = round(sum(row[j]* weights[j] for j in range(len(weights))),5)
            predictions.append(dot) 
        return predictions

    def get_error(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64]) -> float:
        # Compute mean squared error between predictions and ground truth
        # Round to 5 decimal places
        n = len(model_prediction)
        total = 0
        for i in range(n):
            diff = model_prediction[i][0] - ground_truth[i][0]
            total += diff*diff
        mse = total/n
        return round(mse, 5)