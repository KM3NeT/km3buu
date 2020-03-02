# Filename: core.py
"""
Core functions for the package environment

"""

__author__ = "Johannes Schumann"
__copyright__ = "Copyright 2020, Johannes Schumann and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Johannes Schumann"
__email__ = "jschumann@km3net.de"
__status__ = "Development"

import os
from spython.main import Client
from os.path import join, isdir, basename
from thepipe.logger import get_logger

from . import IMAGE_NAME, DOCKER_URL

log = get_logger(basename(__file__))


def build_image(output_dir):
    if not isdir(output_dir):
        raise OSError("Directory not found!")
    image_path = join(output_dir, IMAGE_NAME)
    return Client.build(DOCKER_URL, image=image_path, sudo=False, ext="simg")
