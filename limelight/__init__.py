"""Expose the entrypoints."""
import click
import numpy as np
from greentea.log import LogConfiguration
from greentea.text import Texts
from .theme import Themes
from .news import DataPointSources
from .dataset import Dataset
from .transformer import TextTransformer, TextThemeTransformer
from .vectorizer import TfidfVectorizer, Vectorizer, FeatureSelectedVectorizer


@click.group()
@click.option('-v', '--verbose', is_flag=True)
def main(verbose: bool):
    """Group."""
    LogConfiguration(verbose, 'limelight').configure()


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
    """Train a sparse vectorizer."""
    texts = Dataset(train, TextTransformer())
    vectorizer = TfidfVectorizer()
    vectorizer.fit(texts)
    vectorizer.dump(location)


@main.command()
@click.argument('train', type=DataPointSources.read_csv)
@click.argument('vectorizer', type=Vectorizer.load)
@click.argument('location')
def featuresel(train, vectorizer, location: str):
    """
    """
    dataset = np.array(Dataset(train, TextThemeTransformer()))
    texts = Texts(dataset[:, 0])
    themes = Themes(dataset[:, 1])
    train_vectorizer = FeatureSelectedVectorizer.create_random_forest(
        vectorizer, 20000)
    train_vectorizer.fit(texts, themes)
    train_vectorizer.dump(location)
