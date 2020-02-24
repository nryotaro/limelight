"""Expose transformers."""
from typing import Tuple
from greentea.text import Text
from .news import DataPointSource
from .types import T
from .theme import Theme


class TextTransformer:
    """Transform :py:class:`DataPointSource` to `Text`."""

    def __call__(self, data_point_source: DataPointSource) -> Text:
        """Read text from a source."""
        return data_point_source.read_text()


class RawTransformer:
    """Return `str`."""

    def __call__(self, text: Text) -> str:
        """Transfrom :py:class:`Text` to str."""
        return text.text


class RawTextThemeTransformer:
    """Transform a :py:class:`DataPointSourbce`."""

    def __call__(
            self, data_point_source: DataPointSource) -> Tuple[str, Theme]:
        """Return a tuple of `str` and :py:class:`Theme`."""
        text = data_point_source.read_text()
        theme = data_point_source.get_theme()
        return (text.text, theme)


class NopTransformer:
    """Identity function."""

    def __call__(self, t: T) -> T:
        """Do nothing."""
        return t


class TextThemeTransformer:
    """Transform source to a pair of :py:class:`Text` and :py:class:`Theme`."""

    def __call__(
            self,
            data_point_source: DataPointSource) -> Tuple[Text, Theme]:
        """Transform an argument to the corresponding tuple."""
        text = data_point_source.read_text()
        theme = data_point_source.get_theme()
        return (text, theme)
