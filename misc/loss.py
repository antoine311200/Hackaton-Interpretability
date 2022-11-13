from __future__ import annotations
import itertools

from typing import List, Tuple, Union


from scipy import linalg
import numpy as np
from opt_einsum import contract


class MSELoss:

    def __init__(self):
        pass

    def forward(self, y_pred, y_true):
        return np.mean((y_pred - y_true) ** 2)

    def backward(self, y_pred, y_true):
        return 2 * (y_pred - y_true)