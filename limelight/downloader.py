"""Provide downloader for the original 20 newsgroups dataset.

Refrences
---------
http://qwone.com/~jason/20Newsgroups/

"""
import os.path
import shutil
from logging import getLogger
import tempfile
import tarfile
import requests
import tqdm


class Downloader:
    """A downloader to fetch the original 20 Newsgroups."""
    _LOGGER = getLogger(__name__)

    def __init__(self, location: str, chunk_size=100):
        """Take the path to a location to save the dataset.

        Parameters
        ----------
        location : str
            The path.

        """
        self.location = location
        self.chunk_size = chunk_size

    def download(self):
        """Download the dataset.

        The file should be a gz file.

        """
        response = requests.get(self.url())
        size = int(response.headers['content-length'])
        self._LOGGER.debug(f'Downloading the {size} bytes file.')
        with open(self.location, 'wb') as f, tqdm.tqdm(total=size) as bar:
            self._write(response, f, bar)

    def _write(self, response, writer, bar):
        written_byte_size = 0
        for chunk in response.iter_content(self.chunk_size):
            writer.write(chunk)
            written_byte_size += self.chunk_size
            bar.update(written_byte_size)

    def url(self) -> str:
        """Get the url of the datase."""
        return 'http://qwone.com/~jason/20Newsgroups/20news-19997.tar.gz'

class Extractor:
    """
    """

    def __init__(self,
                 compressed_file,
                 destination_directory):
        """
        """
        self.compressed_file = compressed_file
        self.destination_directory = destination_directory

    def extract(self):
        """
        """
        with tarfile.open(self.compressed_file, 'r:gz') as tar, \
                tempfile.TemporaryDirectory() as tmpdir:
            tar.extractall(tmpdir)
            shutil.move(os.path.join(tmpdir, '20_newsgroups'),
                        self.destination_directory)
