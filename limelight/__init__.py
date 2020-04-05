"""Expose the entrypoints."""
import click
import numpy as np
from greentea.log import LogConfiguration
from greentea.text import Texts
from .theme import Themes
from .news import DataPointSources
from .dataset import Dataset
from .downloader import Initializer
from .transformer import TextTransformer, TextThemeTransformer
from .vectorizer import \
    TfidfVectorizer, \
    Vectorizer, \
    FeatureSelectedVectorizer, \
    LogisticRegressionFsVectorizer


@click.group()
@click.option('-v', '--verbose', is_flag=True)
def main(verbose: bool):
    """Group."""
    LogConfiguration(verbose, 'limelight').configure()


@main.command()
@click.argument('destination')
def download(destination: str):
    """Download 20newsgroups dataset.

    DESTINATION    directory.

    """
    Initializer(destination).prepare()


@main.command()
@click.argument('dataset', type=Dataset.create)
@click.argument('train')
@click.argument('test')
def split(dataset, train: str, test: str):
    """Split dataset into train and test."""
    train_dataset, test_dataset = dataset.train_test_split()
    train_dataset.save_sources_as_csv(train)
    test_dataset.save_sources_as_csv(test)


@main.command()
@click.argument('train', type=DataPointSources.read_csv)
@click.argument('location')
def sparsevec(train: DataPointSources, location: str):
    """Train a sparse vectorizer.

    TRAIN   A CSV file that the `split` subcommnad emitted.
    """
    texts = Texts(Dataset(train, TextTransformer()))
    vectorizer = TfidfVectorizer()
    vectorizer.fit(texts)
    vectorizer.dump(location)


@main.command()
@click.argument('train', type=DataPointSources.read_csv)
@click.argument('vectorizer', type=Vectorizer.load)
@click.argument('location')
def featuresel(train, vectorizer, location: str):
    """Create a vectorizer apply Feature selection to a base vectorizer."""
    dataset = np.array(Dataset(train, TextThemeTransformer()))
    texts = Texts(dataset[:, 0])
    themes = Themes(dataset[:, 1])
    train_vectorizer = LogisticRegressionFsVectorizer.create(
        vectorizer, 20000
    )
    train_vectorizer.fit(texts, themes)
    train_vectorizer.dump(location)


@main.command()
@click.argument('vectorizer', type=Vectorizer.load)
@click.argument('train', type=Dataset.read_sources_from_csv)
@click.argument('location')
def train(vectorizer, train, location):
    """Train a classifier."""
    number_of_features = vectorizer.get_num_of_features()

