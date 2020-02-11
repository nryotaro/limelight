"""Illustrate usage of lime."""
from setuptools import setup, find_packages


setup(name='limelight',
      version="0.0.1",
      packages=find_packages(),
      install_requires=[
          'torch',
          'greentea==3.5.2',
          'click',
          'requests',
          'tqdm'
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
