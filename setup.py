#!/usr/bin/env python3

from setuptools import setup, find_packages


setup(name='mlmisc',
      version='0.39',
      description='Miscellaneous ML Utilities',
      author='Davide Libenzi',
      packages=find_packages(),
      install_requires=[
          'datasets',
          'einops',
          'msgpack',
          'numpy',
          'Pillow',
          'python_misc_utils',
          'torch',
          'torchvision',
      ],
      )
