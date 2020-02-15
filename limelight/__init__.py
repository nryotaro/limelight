"""Expose the entrypoints."""
import click
from greentea.log import LogConfiguration
from torchvision.transforms import Compose
from .dataset import Dataset, DataPointSources, \
    TextTransformer, RawTransformer
from .vectorizer import TfidfVectorizer


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
    transformer = Compose([TextTransformer(), RawTransformer()])
    texts = Dataset(train, transformer)
    vectorizer = TfidfVectorizer()
    vectorizer.fit(texts)
    vectorizer.dump(location)
