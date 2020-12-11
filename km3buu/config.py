#!/usr/bin/env python
# coding=utf-8
# Filename: config.py
# Author: Johannes Schumann <jschumann@km3net.de>
"""
Configuration for km3buu

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

GENERAL_SECTION = "General"
KM3NET_LIB_PATH_KEY = "km3net_lib_path"

GIBUU_SECTION = "GiBUU"
GIBUU_IMAGE_PATH_KEY = "image_path"

PROPOSAL_SECTION = "PROPOSAL"
PROPOSAL_ITP_PATH_KEY = "itp_table_path"

GSEAGEN_SECTION = "gSeagen"
GSEAGEN_PATH_KEY = "path"

GSEAGEN_MEDIA_COMPOSITION_FILE = "MediaComposition.xml"


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
        image_path = self.get(GIBUU_SECTION, GIBUU_IMAGE_PATH_KEY)
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
        section = GIBUU_SECTION
        key = GIBUU_IMAGE_PATH_KEY
        self.set(section, key, value)

    @property
    def proposal_itp_tables(self):
        default_path = abspath(join(dirname(__file__), "../.tables"))
        return self.get(PROPOSAL_SECTION, PROPOSAL_ITP_PATH_KEY, default_path)

    @proposal_itp_tables.setter
    def proposal_itp_tables(self, value):
        self.set(PROPOSAL_SECTION, PROPOSAL_ITP_PATH_KEY, value)

    @property
    def gseagen_path(self):
        return self.get(GSEAGEN_SECTION, GSEAGEN_PATH_KEY)

    @gseagen_path.setter
    def gseagen_path(self, value):
        self.set(GIBUU_SECTION, GSEAGEN_PATH_KEY, value)

    @property
    def km3net_lib_path(self):
        return self.set(GENERAL_SECTION, KM3NET_LIB_PATH_KEY, None)

    @km3net_lib_path.setter
    def km3net_lib_path(self, value):
        return self.get(GENERAL_SECTION, KM3NET_LIB_PATH_KEY, value)


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
            density = float(media.attrib["density"])
            attr["density"] = density
        attr["elements"] = elements
        key = media.attrib["media"]
        compositions[key] = attr
    return compositions


def read_default_media_compositions():
    cfg = Config()
    fpath = join(cfg.gseagen_path, "dat", GSEAGEN_MEDIA_COMPOSITION_FILE)
    return read_media_compositions(fpath)
