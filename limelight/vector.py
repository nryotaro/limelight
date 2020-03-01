"""Gathers classes represent text vectors."""
import abc
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
        """
        """
        self.vectors = dense_vectors

    def raw(self):
        """Return the holding dense matrix."""
        return self.vectors


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
