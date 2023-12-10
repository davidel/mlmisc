#!/usr/bin/env python3

from setuptools import setup, find_packages


setup(name='mlmisc',
      version='0.1',
      description='Miscellaneous ML Utilities',
      author='Davide Libenzi',
      packages=find_packages(),
      install_requires=[
          'torch',
          'numpy',
          'Pillow',
          'py_misc_utils @ git+https://github.com/davidel/py_misc_utils',
      ],
      )
