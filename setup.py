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


def read_requirements(kind):
    with open(os.path.join('requirements', kind + '.txt')) as fobj:
        requirements = [l.strip() for l in fobj.readlines()]
    return requirements

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
    install_requires=read_requirements("install"),
    extras_require={
        kind: read_requirements(kind)
        for kind in ["dev", "extras"]
    },
    python_requires='>=3.0',
    entry_points={'console_scripts': ['km3buu=km3buu.cmd:main']},
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python',
    ],
)
