#!usr/bin/env python
# -*- coding: utf-8 -*-
# Filename: setup.py
"""
KM3BUU setup script.

"""
import os
import tempfile
from setuptools import setup, find_packages

PACKAGE_NAME = 'km3buu'
URL = 'https://git.km3net.de/simulation/km3buu'
DESCRIPTION = 'GiBUU tools for KM3NeT'
__author__ = 'Johannes Schumann'
__email__ = 'jschumann@km3net.de'

with open('requirements.txt') as fobj:
    REQUIREMENTS = [l.strip() for l in fobj.readlines()]

with open('requirements-dev.txt') as fobj:
    DEV_REQUIREMENTS = [l.strip() for l in fobj.readlines()]

setup(
    name=PACKAGE_NAME,
    url=URL,
    description=DESCRIPTION,
    author=__author__,
    author_email=__email__,
    packages=find_packages(),
    include_package_data=True,
    platforms='any',
    setup_requires=['setuptools_scm'],
    use_scm_version={
        'write_to': '{}/version.txt'.format(PACKAGE_NAME),
        'tag_regex': r'^(?P<prefix>v)?(?P<version>[^\+]+)(?P<suffix>.*)?$',
    },
    install_requires=REQUIREMENTS,
    extras_require={
        'dev': DEV_REQUIREMENTS
        },
    python_requires='>=3.0',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python',
    ],
)
