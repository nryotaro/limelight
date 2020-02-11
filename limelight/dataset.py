"""Expose classes relevant to dataset."""
import os
import os.path
import csv
import re
from dataclasses import dataclass
from typing import List, Callable
import torch.utils.data as d
from .theme import Theme
from .types import T
from greentea.text import Text
from greentea.first_class_collection import FirstClassSequence


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
        return self.get_id_str()

    def return_as_dict(self) -> dict:
        """Return a dict that represents this object."""
        return {
            'theme': self.get_theme_name(),
            'id': self.datapoint_id.get_raw()
        }

    @classmethod
    def from_dict(cls, source: dict):
        """Create :py:class:`DataPointMeta` from a dict value."""
        data_point_id = DataPointId(source['id'])
        theme = Theme.create(source['theme'])
        return DataPointMeta(data_point_id, theme)


@dataclass
class DataPointSource:
    """Represent the location of a data point."""

    directory: str
    data_point_meta: DataPointMeta

    def read_text(self) -> Text:
        """Read a text from a file."""
        point_id = self.data_point_meta.get_id_str()
        theme = self.data_point_meta.get_theme_name()
        path = os.path.join(self.directory, theme, point_id)
        with open(path) as f:
            return Text(f.read())

    def return_as_dict(self) -> dict:
        """Return a dict the represents this object."""
        dict_value = self.return_as_dict()
        dict_value['directory'] = self.directory
        return dict_value

    def from_dict(self, source: dict):
        """Create :py:class:`DataPointSource` from a dict value."""
        meta = DataPointMeta.from_dict(source)
        return DataPointSource(source['directory'], meta)


@dataclass
class DataPointSources(FirstClassSequence):
    """A collection of :py:class:`DataPointSource`s."""

    items: List[DataPointSource]

    def save_csv(self, filename) -> None:
        """Write them in csv format."""
        with open(filename, 'w') as csvfile:
            writer = csv.DictWriter(
                csvfile,
                fieldnames=self._fieldnames())
            writer.writeheader()

        for source in self.items:
            writer.writerow(source.return_as_dict())

    @classmethod
    def read_csv(cls, filename: str):
        """Read a file from `filename` into :py:class:`DataPointSources`."""
        with open(filename) as csvfile:
            reader = csv.DictReader(csvfile, fieldnames=cls._fieldnames())
            return DataPointSources(
                [DataPointSource.from_dict(record) for record in reader])

    @classmethod
    def _fieldnames(cls):
        return ['directory', 'theme', 'id']


class TextTransformer:
    """Text reader."""

    def __call__(self, data_point_source: DataPointSource) -> Text:
        """Read text from a source."""
        return data_point_source.read_text()


class RawTransformer:
    """Return `str`."""

    def __call__(self, text: Text) -> str:
        """Transfrom :py:class:`Text` to str."""
        return text.text


class NopTransformer:
    """Identity function."""

    def __call__(self, t: T) -> T:
        """Do nothing."""
        return t


@dataclass
class Dataset(d.Dataset):
    """20newsgroups dataset.

    Attributes
    ----------
    sources: DataPointSources

    transformer: Callable[[DataPointSource], T]

    """

    sources: DataPointSources
    transformer: Callable[[DataPointSource], T]

    def __len__(self) -> int:
        """Return the size."""
        return len(self.sources)

    def __getitem__(self, index):
        """Access the specified item."""
        found = self.sources.__getitem__(index)
        if isinstance(found, DataPointSource):
            return self.transformer(found)
        return Dataset(found, self.transformer)

    def update_transformer(self, transformer: Callable[[DataPointSource], T]):
        """Update :py:attr:`transformer`."""
        return Dataset(self.sources, transformer)

    @classmethod
    def create(cls, dirname: str, transformer=NopTransformer()):
        """Create :py:class:`Dataset` from a directory."""
        sources = DataPointSources(
            [data_point_meta for theme in Theme
             for data_point_meta in cls._load_ids(dirname, theme)])
        return Dataset(sources, transformer)

    @classmethod
    def _load_ids(cls, dirname, theme: Theme) -> List[DataPointMeta]:
        theme_dir = os.path.join(dirname, theme.get_theme_name())
        return [DataPointSource(dirname,
                                DataPointMeta(DataPointId(point_id), theme))
                for point_id in os.listdir(theme_dir)
                if re.match(r'\d+', point_id)]

    def save_sources_as_csv(self, filename) -> None:
        """Save :py:attr:`sources` as a CSV file."""
        self.sources.save_csv(filename)

    @classmethod
    def read_sources_from_csv(
            cls,
            filename: str,
            transformer=NopTransformer()):
        """Read :py:class:`Dataset` from a file."""
        sources = DataPointSources.read_csv(filename)
        return Dataset(sources, transformer)
