from __future__ import annotations
import itertools

from typing import List, Tuple, Union


from scipy import linalg
import numpy as np
from opt_einsum import contract



class TensorTrain:

    def __init__(self, cores: List[np.ndarray] = [], tucker_rank: List[np.int16] = []) -> None:
        
        self.tucker_rank = tucker_rank

        self.cores = cores
        self.n_cores = len(tucker_rank) + 1

        self.type = 'state' if len(cores[0].shape) == 3 else 'operator'


    def __add__(self, other: TensorTrain, tucker_rank: List[np.int16] = None) -> TensorTrain:
        """Adds two TensorTrains.
        
        Args:
            other (TensorTrain): The other TensorTrain.
            
        Returns:
            TensorTrain: The sum of the two TensorTrains.
        """

        if other is None:
            return self

        if tucker_rank is not None:
            if self.tucker_rank != other.tucker_rank and len(self.cores[0].shape) == len(other.cores[0].shape):
                raise ValueError('Tucker ranks of the TensorTrains are not equal.')

        cores = []
        for i in range(self.n_cores):
            is_left = i == 0
            is_right = i == self.n_cores - 1

            core = np.zeros(shape=(
                self.cores[i].shape[0] + other.cores[i].shape[0] if not is_left else 1,
                *(self.cores[i].shape[1:-1]),
                self.cores[i].shape[-1] + other.cores[i].shape[-1] if not is_right else 1
            ))

            if len(self.cores[0].shape) == 3:
                for inp in range(self.cores[i].shape[1]):
                    A = self.cores[i][:, inp, :]
                    B = other.cores[i][:, inp, :]
                    if is_left:
                        core[:, inp, :] = np.block([A, B])
                    elif is_right:
                        core[:, inp, :] = np.block([A.T, B.T]).T
                    else:
                        core[:, inp, :] = linalg.block_diag(A, B)
            else:
                for inp in range(self.cores[i].shape[1]):
                    for out in range(self.cores[i].shape[2]):
                        A = self.cores[i][:, inp, out, :]
                        B = other.cores[i][:, inp, out, :]
                        if is_left:
                            core[:, inp, out, :] = np.block([A, B])
                        elif is_right:
                            core[:, inp, out, :] = np.block([A.T, B.T]).T
                        else:
                            core[:, inp, out, :] = linalg.block_diag(A, B)
            # if is_left:
            #     core[:, ..., :] = np.block([self.cores[i][:, ..., :], other.cores[i][:, ..., :]])
            # elif is_right:
            #     core[:, ..., :] = np.block([self.cores[i][:, ..., :].T, other.cores[i][:, ..., :].T])
            # else:
            #     core[:, ..., :] = np.block([
            #         [self.cores[i][:, ..., :], np.zeros(shape=self.cores[i].shape)],
            #         [np.zeros(shape=self.cores[i].shape) ,other.cores[i][:, ..., :]]
            #     ])

            # print(core)

            cores.append(core)

        tt = TensorTrain(cores, self.tucker_rank)

        if tucker_rank is not None:
            if tucker_rank != self.tucker_rank:
                tt.rounding(tucker_rank)

        return tt

    
    def __matmul__(self, other: TensorTrain) -> TensorTrain:
        """Multiplies two TensorTrains.
        
        Args:
            other (TensorTrain): The other TensorTrain.
            
        Returns:
            TensorTrain: The product of the two TensorTrains.
        """

        cores = []
        tucker_rank = []

        if other.type == 'state':
            for i in range(self.n_cores):
                # core  = contract('ijkl, mjn -> imknl', self.cores[i], other.cores[i], optimize='optimal')
                core = contract(other.cores[i], [1,2,3], self.cores[i], [4,2,6,5], [1,4,6,3,5])

                core = core.reshape((
                    self.cores[i].shape[0] * other.cores[i].shape[0],
                    self.cores[i].shape[2],
                    self.cores[i].shape[3] * other.cores[i].shape[2]
                ))
                cores.append(core)
                tucker_rank.append(core.shape[2])
        else:
            for i in range(self.n_cores):
                core  = contract('ijkl, mkno -> imjnol', self.cores[i], other.cores[i], optimize='optimal')
                core = core.reshape((
                    self.cores[i].shape[0] * other.cores[i].shape[0],
                    self.cores[i].shape[1],
                    self.cores[i].shape[2],
                    self.cores[i].shape[3] * other.cores[i].shape[3]
                ))
                cores.append(core)
                tucker_rank.append(core.shape[3])
        
        return TensorTrain(cores, tucker_rank[:-1])


    def __getitem__(self, key):
        n_keys = len(key)

        # if len(key_inp) != self.n_cores:
        #     raise Exception("Input indices do not match the number of sites")
        # if len(key_out) != self.n_cores:
        #     raise Exception("Output indices do not match the number of sites")

        # print(key)

        if n_keys == 2:
            return self.retrieve(key[0], key[1])
        elif n_keys == 1:
            # print("ok")
            return self.retrieve(key[0])


    def rounding(self, new_tucker_rank: List[np.int16]) -> None:
        """Rounds the TensorTrain."""

        self.tucker_rank = new_tucker_rank
        new_tucker_rank = [1] + new_tucker_rank + [1]

        for i in range(self.n_cores-1):
            L = self.to_matrix(self.cores[i], 'left')
            R = self.to_matrix(self.cores[i+1], 'right')

            Q, S = np.linalg.qr(L)
            Q = Q[:, :new_tucker_rank[i+1]]
            S = S[:new_tucker_rank[i+1], :]

            W = S @ R

            shape_curr = (*self.cores[i].shape[:-1], new_tucker_rank[i+1])
            shape_next = (new_tucker_rank[i+1], *self.cores[i+1].shape[1:])

            self.cores[i] = self.to_tensor(Q, shape_curr)
            self.cores[i+1] = self.to_tensor(W, shape_next)

        return self
        

    def left_canonicalize(self) -> None:
        """Left canonicalizes the TensorTrain."""

        for i in range(self.n_cores-1):
            L = self.to_matrix(self.cores[i], 'left')
            R = self.to_matrix(self.cores[i+1], 'right')

            U, B = np.linalg.qr(L)
            W = B @ R

            self.cores[i] = self.to_tensor(U, self.cores[i].shape)
            self.cores[i+1] = self.to_tensor(W, self.cores[i+1].shape)

    def right_canonicalize(self) -> None:
        """Right canonicalizes the TensorTrain."""

        for i in range(self.n_cores-1, 0, -1):
            L = self.to_matrix(self.cores[i-1], 'left')
            R = self.to_matrix(self.cores[i], 'right')

            W, B = np.linalg.qr(R.T)
            U = L @ B.T

            self.cores[i-1] = self.to_tensor(U, self.cores[i-1].shape)
            self.cores[i] = self.to_tensor(W.T, self.cores[i].shape)

    
    def copy(self) -> TensorTrain:
        """Copies the TensorTrain."""

        return TensorTrain(self.cores, self.tucker_rank)


    @staticmethod
    def from_tensor(tensor: np.ndarray, tucker_rank: List[np.int16], type: str = None) -> None:
        """Constructs a TensorTrain from a tensor.

        Args:
            tensor (np.ndarray): The tensor to be decomposed.
            tucker_rank (List[np.int16]): The Tucker rank of the TensorTrain.

        Returns:
            TensorTrain: The TensorTrain decomposition of the tensor.
        """

        if type == 'operator':
            shape = tuple(zip(tensor.shape[:len(tensor.shape)//2], tensor.shape[len(tensor.shape)//2:]))
        else:
            shape = tensor.shape

        print('Tensor train shape ->', shape)

        cores = []
        n_cores = len(tucker_rank) + 1
        tucker_rank = [1] + tucker_rank + [1]

        T = tensor
        for i in range(n_cores-1):
            L = T.reshape(tucker_rank[i]*np.multiply.reduce(shape[i]), -1)

            Q, R = np.linalg.qr(L)
            Q = Q[:, :tucker_rank[i+1]]
            R = R[:tucker_rank[i+1], :]
            T = R

            if type == 'operator':
                cores.append(Q.reshape(tucker_rank[i], *shape[i], tucker_rank[i+1]))
            else:
                cores.append(Q.reshape(tucker_rank[i], shape[i], tucker_rank[i+1]))

        if type == 'operator':
            cores.append(T.reshape(tucker_rank[-2], *shape[-1], tucker_rank[-1]))
        else:
            cores.append(T.reshape(tucker_rank[-2], shape[-1], tucker_rank[-1]))

        tt = TensorTrain(cores, tucker_rank[1:-1])
        return tt

    
    def retrieve(self, indices_inp, indices_out = None) -> np.ndarray:
        """Retrieves the tensor from the TensorTrain."""
        # indices_inp = kwargs.get('indices_inp', None)
        # indices_out = kwargs.get('indices_out', None)

        einsum_structure = []

        # print(indices_inp)

        for i in range(self.n_cores):
            if indices_out is not None:
                einsum_structure.append(self.cores[i][:, indices_inp[i], indices_out[i], :])
                einsum_structure.append([i, i+1])
            else:
                einsum_structure.append(self.cores[i][:, indices_inp[i], :])
                einsum_structure.append([i, Ellipsis, i+1])
        return contract(*einsum_structure)

    
    def to_numpy(self):
        """Converts the TensorTrain to a numpy array."""

        if len(self.cores[0].shape) == 4:
            input_shape = ()
            output_shape = ()

            for i in range(self.n_cores):
                input_shape += (self.cores[i].shape[1], )
                output_shape += (self.cores[i].shape[2], )

            tensor = np.zeros(shape=(*input_shape, *output_shape))

            print("Shape", input_shape, output_shape)

            range_inp = [range(inp) for inp in input_shape]
            range_out = [range(out) for out in output_shape]
            
            for inp in itertools.product(*range_inp):
                for out in itertools.product(*range_out):
                    tensor[(*inp, *out)] = self[inp, out]
            
            return tensor
        else:
            input_shape = ()
            
            for i in range(self.n_cores):
                input_shape += (self.cores[i].shape[1], )

            tensor = np.zeros(shape=input_shape)

            range_inp = [range(inp) for inp in input_shape]
            
            for inp in itertools.product(*range_inp):
                tensor[inp] = self[[inp]]
            
            return tensor

    

    @staticmethod
    def to_matrix(tensor: np.ndarray, mode: str) -> np.ndarray:
        """Convert a tensor to a matrix by unfolding along a given mode.

        Args:
            tensor: The tensor to be unfolded.
            mode: The mode along which the tensor is unfolded.

        Returns:
            The unfolded tensor.
        """
        if mode == 'left':
            return np.reshape(tensor, (np.multiply.reduce(tensor.shape[:-1]), -1))
        elif mode == 'right':
            return np.reshape(tensor, (-1, np.multiply.reduce(tensor.shape[1:])))


    @staticmethod
    def to_tensor(matrix: np.ndarray, shape: List[int]) -> np.ndarray:
        """Convert a matrix to a tensor by folding along a given mode.

        Args:
            matrix: The matrix to be folded.
            shape: The shape of the tensor.
            mode: The mode along which the tensor is folded.

        Returns:
            The folded tensor.
        """
        return matrix.reshape(shape)





if __name__ == "__main__":
    tensor = np.random.rand(4, 4, 4, 4)
    tucker_rank = [2, 2, 2]
    tt = TensorTrain.from_tensor(tensor, tucker_rank)

    tt.left_canonicalize()
    print(np.diag(
        tt.cores[1].reshape(tt.cores[1].shape[0]*tt.cores[1].shape[1], -1).T
        @ 
        tt.cores[1].reshape(tt.cores[1].shape[0]*tt.cores[1].shape[1], -1)
    ))

    tt.right_canonicalize()
    print(np.diag(
        tt.cores[1].reshape(-1, tt.cores[1].shape[1]*tt.cores[1].shape[2])
        @ 
        tt.cores[1].reshape(-1, tt.cores[1].shape[1]*tt.cores[1].shape[2]).T
    ))




    tensor = np.random.rand(4, 4, 4, 4, 2, 2, 2, 2)
    tucker_rank = [4, 4, 4]
    tto = TensorTrain.from_tensor(tensor, tucker_rank, type='operator')

    tto.right_canonicalize()
    print(np.diag(
        tto.cores[1].reshape(-1,np.multiply.reduce(tto.cores[1].shape[1:]))
        @ 
        tto.cores[1].reshape(-1, np.multiply.reduce(tto.cores[1].shape[1:])).T
    ))
    
    tto.left_canonicalize()
    print(np.diag(
        tto.cores[1].reshape(np.multiply.reduce(tto.cores[1].shape[:-1]), -1).T
        @ 
        tto.cores[1].reshape(np.multiply.reduce(tto.cores[1].shape[:-1]), -1)
    ))

    tto.rounding([2, 2, 2])
    for core in tto.cores:
        print(core.shape)
    
    print(np.diag(
        tto.cores[1].reshape(np.multiply.reduce(tto.cores[1].shape[:-1]), -1).T
        @ 
        tto.cores[1].reshape(np.multiply.reduce(tto.cores[1].shape[:-1]), -1)
    ))


    tensor1 = np.arange(4*4*4*4).reshape(4, 4, 4, 4)
    tucker_rank1 = [2, 2, 2]
    
    tensor2 = np.arange(4*4*4*4).reshape(4, 4, 4, 4)
    tucker_rank2 = [2, 2, 2]

    tt1 = TensorTrain.from_tensor(tensor1, tucker_rank1)
    tt2 = TensorTrain.from_tensor(tensor2, tucker_rank2)

    # print(tt1.to_numpy())
    # print(tensor1[1, 1, 1, 1])
    # print(tt1[[[1, 1, 1, 1]]])

    tt3 = tt1 + tt2
    # print(tt3.to_numpy())
    # print(tensor1+tensor2)