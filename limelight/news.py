"""Provide classes to represent news text."""
import codecs
import os
import csv
from dataclasses import dataclass
from typing import List, Callable
from greentea.text import Text
from greentea.first_class_collection import FirstClassSequence
from .theme import Theme
from .types import T


@dataclass
class DataPointId:
    """Identify a datapoint."""

    datapoint_id: int

    def get_raw(self) -> int:
        """Return the primitive value."""
        return self.datapoint_id

    def get_as_str(self) -> str:
        """Return the id as `str` value."""
        return str(self.get_raw())


@dataclass
class DataPointMeta:
    """A metadata of a datapoint.

    Attributes
    ----------
    datapoint_id

    theme

    """

    datapoint_id: DataPointId
    theme: Theme

    def combine(self, function: Callable[[DataPointId, Theme], T]) -> T:
        """Trans form the pair by `function`."""
        return function(self.datapoint_id, self.theme)

    def get_theme_name(self) -> str:
        """See :py:meth:`Theme.get_theme_name`."""
        return self.theme.get_theme_name()

    def get_id_str(self) -> str:
        """See :py:meth:`DataPointId.get_as_str`."""
        return self.datapoint_id.get_as_str()

    def return_as_dict(self) -> dict:
        """Return a dict that represents this object."""
        return {
            'theme': self.get_theme_name(),
            'id': self.datapoint_id.get_raw()
        }

    @classmethod
    def from_dict(cls, source: dict):
        """Create :py:class:`DataPointMeta` from a dict value."""
        data_point_id = DataPointId(int(source['id']))
        theme = Theme.create(source['theme'])
        return DataPointMeta(data_point_id, theme)


@dataclass
class DataPointSource:
    """Represent the location of a data point.

    Attributes
    ----------
    directory: str

    data_point_meta: DataPointMeta

    """

    directory: str
    data_point_meta: DataPointMeta

    def read_text(self) -> Text:
        """Read a text from a file."""
        point_id = self.data_point_meta.get_id_str()
        theme = self.data_point_meta.get_theme_name()
        path = os.path.join(self.directory, theme, point_id)
        with codecs.open(path, encoding='utf-8', errors='ignore') as f:
            return Text(f.read())

    def return_as_dict(self) -> dict:
        """Return a dict the represents this object."""
        dict_value = self.data_point_meta.return_as_dict()
        dict_value['directory'] = self.directory
        return dict_value

    @classmethod
    def from_dict(cls, source: dict):
        """Create :py:class:`DataPointSource` from a dict value."""
        meta = DataPointMeta.from_dict(source)
        return DataPointSource(source['directory'], meta)

    def get_theme(self) -> Theme:
        """Return the theme.

        Returns
        -------
        Theme

        """
        return self.data_point_meta.theme


@dataclass
class DataPointSources(FirstClassSequence):
    """A collection of :py:class:`DataPointSource`s.

    Attributes
    ----------
    items: List[DataPointSource]

    """

    items: List[DataPointSource]

    @property
    def sequence(self):
        """Return :py:attr:`items`."""
        return self.items

    def save_csv(self, filename) -> None:
        """Write them in csv format."""
        with open(filename, 'w') as csvfile:
            writer = csv.DictWriter(
                csvfile,
                fieldnames=self._fieldnames())
            writer.writeheader()
            self._write(writer)

    def _write(self, writer):
        for source in self.items:
            writer.writerow(source.return_as_dict())

    @classmethod
    def read_csv(cls, filename: str):
        """Read a file from `filename` into :py:class:`DataPointSources`."""
        with open(filename) as csvfile:
            reader = csv.DictReader(csvfile, fieldnames=cls._fieldnames())
            next(reader)
            return DataPointSources(
                [DataPointSource.from_dict(record) for record in reader])

    @classmethod
    def _fieldnames(cls):
        return ['directory', 'theme', 'id']
