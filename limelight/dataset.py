"""Expose classes relevant to dataset."""
import os
import os.path
import re
from dataclasses import dataclass
from collections.abc import Sequence
from typing import List, Callable
import torch.utils.data as d
from sklearn.model_selection import train_test_split
from .theme import Theme
from .types import T
from .transformer import NopTransformer
from .news import DataPointSource, DataPointMeta, DataPointId, DataPointSources


@dataclass
class Dataset(d.Dataset, Sequence):
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
        abs_dirname = os.path.abspath(dirname)
        sources = DataPointSources(
            [data_point_meta for theme in Theme
             for data_point_meta in cls._load_ids(abs_dirname, theme)])
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

    def train_test_split(self):
        """Split dataset into train and test."""
        train, test = train_test_split(self)
        return Dataset(DataPointSources(train), self.transformer), \
            Dataset(DataPointSources(test), self.transformer)
