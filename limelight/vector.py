"""Gathers classes represent text vectors."""
import abc
import torch
import numpy as np


class TextVectors(metaclass=abc.ABCMeta):
    """
    """

    @abc.abstractmethod
    def raw(self):
        """Return the matrix that represent feature vectors.."""


class DenseTextVectors(TextVectors):
    """
    """

    def __init__(self, dense_vectors):
        """Take a `numpy.ndarray`.

        Parameters
        ----------
        dense_vectors: numpy.ndarray

        """
        self.vectors = dense_vectors

    def raw(self):
        """Return the holding dense matrix."""
        return self.vectors

    def as_torch_tensor(self):
        """Convert :py:attr:`vectors` to a `torch.Tensor`."""
        if isinstance(self.vectors, torch.Tensor):
            return self.vectors
        if isinstance(self.vectors, np.ndarray):
            return torch.from_numpy(self.vectors)
        raise NotImplementedError(
            f'Failed to a {type(self.vectors)} object to a torch.Tensor one.')


class SparseTextVectors(TextVectors):
    """
    """

    def __init__(self, sparse_vectors):
        """

        Parameters
        ----------
        sparse_vectors

        """
        self.vectors = sparse_vectors

    def raw(self):
        """Return the holding sparse matrix."""
        return self.vectors
