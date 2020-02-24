"""Gathers utilities to build feature vectors from text documents."""
import abc
import joblib
from greentea.text import Texts
import sklearn.feature_extraction.text as t
import sklearn.feature_selection as s
import sklearn.ensemble as e
import sklearn.linear_model as li
from .vector import TextVectors, SparseTextVectors
from .theme import Themes


class Vectorizer(metaclass=abc.ABCMeta):
    """Transform texts to feature vectors."""

    @abc.abstractmethod
    def fit(self, texts, themes=None, **kwargs):
        """Fit on `texts`.

        Parameters
        ----------
        texts: Sequence[str]

        Returns
        -------
        self

        """

    @abc.abstractmethod
    def transform(self, texts: Texts) -> TextVectors:
        """Transform texts to feature vectors."""

    @abc.abstractmethod
    def dump(self, filename: str):
        """Write this object to a file."""

    @abc.abstractmethod
    def load(self, filename: str):
        """Load a :py:class:`Vectorizer` from `filename`."""


class TfidfVectorizer(Vectorizer):
    """TfidfVectorizer."""

    def __init__(self):
        """Create a TfidfVectorizer object."""
        self.vectorizer = t.TfidfVectorizer()

    def fit(self, texts: Texts, themes=None, **kwargs):
        """Overwrite the parent method.

        Parameters
        ----------
        texts: Texts

        """
        raw_texts = texts.raw_texts()
        return self.vectorizer.fit(raw_texts)

    def transform(self, texts: Texts) -> SparseTextVectors:
        """Transform texts to feature vectors."""
        vectors = self.vectorizer.transform(texts.raw_texts())
        return SparseTextVectors(vectors.astype('float32'))

    def dump(self, filename):
        """Write itself to `filename`."""
        joblib.dump(self, filename)

    def load(self, filename):
        """Overwrite the parent method."""
        return joblib.load(filename)


class FeatureSelectedVectorizer(Vectorizer):
    """
    """

    def __init__(
            self,
            vectorizer: Vectorizer,
            select_from_model: s.SelectFromModel):
        """Take a :py:class:`Vectorizer` and :py:class:`SelectFromModel`."""
        self.vectorizer = vectorizer
        self.select_from_model = select_from_model

    def fit(self, texts: Texts, themes: Themes, **kwargs):
        """Fit the `FeatureSelectedVectorizer`.

        Parameters
        ----------
        texts: Texts

        themes: Themes

        Returns
        -------
        self

        """
        raw_texts = texts.raw_texts()
        feature_vectors = self.vectorizer.transform(raw_texts)
        raw_feature_vectors = feature_vectors.raw()
        matrix = themes.get_index_matrix()
        return self.select_from_model.fit(raw_feature_vectors, matrix)

    @classmethod
    def create_random_forrest(
            cls, vectorizer: Vectorizer, max_features: int):
        """Create a :py:class:`FeatureSelectedVectorizer`.

        It uses a `RandomForestClassifier` as a base estimator.

        """
        estimator = e.RandomForestClassifier()
        selector = s.SelectFromModel(estimator, max_features)
        return FeatureSelectedVectorizer(vectorizer, selector)
