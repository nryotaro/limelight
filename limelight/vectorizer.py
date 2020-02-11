"""gathers utilities to build feature vectors from text documents."""
import abc
from collections.abc import Sequence
import joblib
import sklearn.feature_extraction.text as t


class Vectorizer(metaclass=abc.ABCMeta):
    """
    """

    @abc.abstractmethod
    def fit(self, texts):
        """Fit on `texts`.

        Parameters
        ----------
        texts: Sequence[str]

        """

    @abc.abstractmethod
    def dump(self, filename: str):
        """Write this object to a file."""


class TfidfVectorizer(Vectorizer):
    """TfidfVectorizer."""

    def __init__(self):
        """
        """
        self.vectorizer = t.TfidfVectorizer()

    def fit(self, texts):
        """Overwrite the parent method.

        Parameters
        ----------
        texts: : Sequence[str]

        """

        self.vectorizer.fit(texts)

    def dump(self, filename):
        joblib.dump(self, filename)
