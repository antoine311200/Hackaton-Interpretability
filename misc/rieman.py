from __future__ import annotations
import itertools

from typing import List, Tuple, Union


from scipy import linalg
import numpy as np
from opt_einsum import contract

from tensortrain import TensorTrain

class RiemanianManifold:

    def project(self, X: TensorTrain, W: TensorTrain) -> TensorTrain:

        W_left = W.copy().left_canonicalize()
        W_right = W.copy().right_canonicalize()
        
        cores = []
        for i in range(X.n_cores):
            is_left = i == 0
            is_right = i == X.n_cores - 1

            core = np.zeros(shape=(
                X.cores[i].shape[0] if not is_left else 1,
                *(self.cores[i].shape[1:-1]),
                X.cores[i].shape[-1] if not is_right else 1
            ))

            for inp in range(X.cores[i].shape[1]):
                U = W_left.cores[i][:, inp, :]
                V = W_right.cores[i][:, inp, :]
                dU = RiemanianManifold.delta(X, W)
                if is_left:
                    core[:, inp, :] = np.block([dU, U])
                elif is_right:
                    core[:, inp, :] = np.block([V, dU]).T
                else:
                    core[:, inp, :] = np.block([[V, np.zeros(V.shape[0], U.shape[1])], [dU, U]])

    @staticmethod
    def delta(X: TensorTrain, W: TensorTrain, k: int) -> np.ndarray:
        einsum_structure = []
        for i in range(X.n_cores):
            if i != k:
                einsum_structure += [
                    X.cores[i], [i+1, X.n_cores+2+i, i+2],
                    W.cores[i], [2*X.n_cores+2+i, X.n_cores+2+i, 3*X.n_cores+3+i, 2*X.n_cores+3+i]
                ]
            else:
                einsum_structure += [
                    X.cores[i], [i+1, X.n_cores+2+i, i+2]
                ]

        output_indices = [1, X.n_cores+1, X.n_cores+2+k, 2*X.n_cores+2, 2*X.n_cores+2+k, 2*X.n_cores+3+k, 3*X.n_cores+2]
        output_indices += list(range(3*X.n_cores+3, 3*X.n_cores+3+k)) + list(range(3*X.n_cores+4+k, 4*X.n_cores+2))

        einsum_structure += [output_indices]

        return contract(*einsum_structure, optimize='auto-hq').squeeze()

        