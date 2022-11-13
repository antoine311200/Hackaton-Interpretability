from __future__ import annotations
import itertools

from typing import List, Tuple, Union


from scipy import linalg
import numpy as np
from opt_einsum import contract

from tensortrain import TensorTrain

class Linear:

    def __init__(self, input_rank: List[np.int16], output_rank: List[np.int16], tucker_rank: List[np.int16], has_bias: bool = False) -> None:
        self.tucker_rank = tucker_rank
        self.has_bias = has_bias

        self.input_rank = input_rank
        self.output_rank = output_rank

        self.build()

    
    def build(self):

        t = np.arange(np.multiply.reduce(self.input_rank+self.output_rank)).reshape(self.input_rank+self.output_rank)#np.random.randn(*(input_rank+output_rank))
        self.W = TensorTrain.from_tensor(t, tucker_rank=self.tucker_rank, type='operator')

        if self.has_bias:
            u = np.arange(np.multiply.reduce(self.output_rank)).reshape(self.output_rank)
            self.B = TensorTrain.from_tensor(u, tucker_rank=self.tucker_rank)
        else:
            self.B = None

    def __call__(self, x: np.ndarray) -> TensorTrain:
        return self.forward(x)

    def forward(self, X: TensorTrain) -> TensorTrain:
        return (self.W @ X + self.B).rounding(self.W.tucker_rank)


if __name__ == '__main__':
    l = Linear([3, 3,], [5, 5], [3,], has_bias=True)
    print([core.shape for core in l.W.cores])


    w = np.arange(3*3*5*5).reshape((3, 3, 5, 5)).astype(float)
    w2 = np.arange(3*3*5*5).reshape((3, 3, 5, 5))
    x = np.arange(3*3).reshape(3, 3, )
    y = np.matmul(x.reshape(3*3), l.W.to_numpy().reshape(3*3, 5*5)).reshape(5, 5) + l.B.to_numpy().reshape(5, 5)
    # y2 = np.matmul(x.reshape(5*5), w.reshape((5*5, 5*5))).reshape(5, 5)

    X = TensorTrain.from_tensor(x, tucker_rank=[3,])
    # W = TensorTrain.from_tensor(w, tucker_rank=[3,], type='operator')
    # W2 = TensorTrain.from_tensor(w2, tucker_rank=[3,], type='operator')

    Y = l(X)

    print(Y.to_numpy())
    print(y)

    # print(y.reshape(5, 5)[1, 1])
    # print(Y[[[1, 1]]])

    # print(y)
    # print(y2)
    # print(Y.to_numpy())

    # print((W+W2).to_numpy())

    # print([core.shape for core in Y.cores])

