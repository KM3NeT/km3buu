#!/usr/bin/env python
# coding=utf-8
# Filename: config.py
# Author: Johannes Schumann <jschumann@km3net.de>
"""


"""
import os
import click
from os.path import isfile, isdir, join, dirname, abspath
from configparser import ConfigParser, Error, NoOptionError, NoSectionError
from thepipe.logger import get_logger
from . import IMAGE_NAME
from .environment import build_image
import mendeleev
import xml.etree.ElementTree as ElementTree

__author__ = "Johannes Schumann"
__copyright__ = "Copyright 2020, Johannes Schumann and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Johannes Schumann"
__email__ = "jschumann@km3net.de"
__status__ = "Development"

CONFIG_PATH = os.path.expanduser("~/.km3buu/config")

log = get_logger(__name__)

GIBUU_SECTION = "GiBUU"
PROPOSAL_SECTION = "PROPOSAL"


class Config(object):
    def __init__(self, config_path=CONFIG_PATH):
        self.config = ConfigParser()
        self._config_path = config_path
        if isfile(self._config_path):
            self.config.read(self._config_path)
        else:
            os.makedirs(dirname(CONFIG_PATH), exist_ok=True)

    def set(self, section, key, value):
        if section not in self.config.sections():
            self.config.add_section(section)
        self.config.set(section, key, value)
        with open(self._config_path, "w") as f:
            self.config.write(f)

    def get(self, section, key, default=None):
        try:
            value = self.config.get(section, key)
            try:
                return float(value)
            except ValueError:
                return value
        except (NoOptionError, NoSectionError):
            return default

    @property
    def gibuu_image_path(self):
        key = "image_path"
        image_path = self.get(GIBUU_SECTION, key)
        if image_path is None or not isfile(image_path):
            dev_path = abspath(join(dirname(__file__), os.pardir, IMAGE_NAME))
            if isfile(dev_path):
                image_path = dev_path
            elif click.confirm("Is the GiBUU image already available?",
                               default=False):
                image_path = click.prompt("GiBUU image path?",
                                          type=click.Path(exists=True,
                                                          dir_okay=False))
            elif click.confirm("Install image from remote?", default=True):
                default_dir = join(os.environ["HOME"], ".km3buu")
                image_dir = click.prompt(
                    "GiBUU image path (default: ~/.km3buu) ?",
                    type=click.Path(exists=True, file_okay=False),
                    default=default_dir,
                )
                image_path = build_image(image_dir)
            self.set(GIBUU_SECTION, key, image_path)
        return image_path

    @gibuu_image_path.setter
    def gibuu_image_path(self, value):
        section = "GiBUU"
        key = "image_path"
        self.set(section, key, value)

    @property
    def proposal_itp_tables(self):
        return self.get(PROPOSAL_SECTION, "itp_table_path")

    @proposal_itp_tables.setter
    def proposal_itp_tables(self, value):
        self.set(PROPOSAL_SECTION, "itp_table_path")


def read_media_compositions(filename):
    """
    Read gSeagen media composition xml formated file

    Parameters
    ----------
    filename: str
        Input file
    """
    xmlroot = ElementTree.parse(filename).getroot()
    if xmlroot == "media_comp":
        raise KeyError()

    compositions = dict()
    for media in xmlroot:
        if media.tag != "param_set":
            continue
        elements = dict()
        for element in media:
            name = element.attrib["name"]
            fraction = float(element.text)
            elements[name] = (mendeleev.element(name), fraction)
        attr = dict()
        if "density" in media.attrib:
            density = media.attrib["density"]
            attr["density"] = density
        attr["elements"] = elements
        key = media.attrib["media"]
        compositions[key] = attr
    return compositions
