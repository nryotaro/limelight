"""Illustrate usage of lime."""
from setuptools import setup, find_packages


setup(name='limelight',
      version="0.0.1",
      packages=find_packages(),
      install_requires=[
          'torch',
          'greentea==3.8.0',
          'click',
          'requests',
          'tqdm',
          'torchvision',
          'scikit-learn'
      ],
      entry_points={'console_scripts': ['limelight=limelight:main']},
      extras_require={
          'test': [
              'pytest'
          ],
          'dev': [
              'ipython',
              'python-language-server[all]'
          ]})
