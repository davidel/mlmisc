#!/usr/bin/env python3

from setuptools import setup, find_packages


setup(name='mlmisc',
      version='0.38',
      description='Miscellaneous ML Utilities',
      author='Davide Libenzi',
      packages=find_packages(),
      install_requires=[
          'datasets',
          'einops',
          'msgpack',
          'numpy',
          'Pillow',
          'py_misc_utils @ git+https://github.com/davidel/py_misc_utils',
          'torch',
          'torchvision',
      ],
      )
