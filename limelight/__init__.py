"""Expose the entrypoints."""
import click
from sklearn.model_selection import train_test_split
from greentea.log import LogConfiguration
from .dataset import Dataset


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
    train_dataset, test_dataset = train_test_split(dataset)
    train_dataset.save_sources_as_csv(train)
    test_dataset.save_sources_as_csv(test)
