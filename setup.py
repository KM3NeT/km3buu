#!usr/bin/env python
# -*- coding: utf-8 -*-
# Filename: setup.py
"""
KM3BUU setup script.

"""

from setuptools import setup, find_packages
from setuptools.command.install import install
from setuptools.command.develop import develop
from setuptools.command.egg_info import egg_info

PACKAGE_NAME = 'km3buu'
URL = 'https://git.km3net.de/simulation/km3buu'
DESCRIPTION = 'GiBUU tools for KM3NeT'
__author__ = 'Johannes Schumann'
__email__ = 'jschumann@km3net.de'

import os

with open('requirements.txt') as fobj:
    REQUIREMENTS = [l.strip() for l in fobj.readlines()]


def _build_image():
    os.system('make buildremote')


class CustomInstallCmd(install):
    def run(self):
        install.run(self)
        _build_image()


class CustomDevelopCmd(develop):
    def run(self):
        develop.run(self)
        _build_image()


class CustomEggInfoCmd(egg_info):
    def run(self):
        egg_info.run(self)
        _build_image()


setup(
    name=PACKAGE_NAME,
    url=URL,
    description=DESCRIPTION,
    author=__author__,
    author_email=__email__,
    packages=find_packages(),
    include_package_data=True,
    platforms='any',
    cmdclass={'install': CustomInstallCmd, 
              'develop': CustomDevelopCmd,
              'egg_info': CustomEggInfoCmd},
    setup_requires=['setuptools_scm'],
    use_scm_version={
        'write_to': '{}/version.txt'.format(PACKAGE_NAME),
        'tag_regex': r'^(?P<prefix>v)?(?P<version>[^\+]+)(?P<suffix>.*)?$',
    },
    install_requires=REQUIREMENTS,
    python_requires='>=3.0',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python',
    ],
)
