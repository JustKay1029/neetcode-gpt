import numpy as np
from numpy.typing import NDArray


class Solution:
    def sigmoid(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: 1 / (1 + e^(-z))
        y = 1 / (1 + np.exp(-z))
        return np.round(y, 5)

    def relu(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: max(0, z) element-wise
        ans = []
        for i in z:
            if i <= 0:
                ans.append(0.0)
            else:
                ans.append(i)
        return ans

