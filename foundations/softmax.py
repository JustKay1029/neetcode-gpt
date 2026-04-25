import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        list_e = []
        k = z-max(z)
        for i in k:
            list_e.append(np.exp(i))
        s = sum(list_e)
        values = list_e/s
        ans = []
        for value in values:
            ans.append(round(value,4))
    
        return ans