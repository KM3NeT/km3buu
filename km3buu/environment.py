# Filename: environment.py
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
from spython.utils import get_singularity_version
from os.path import join, isdir, basename
from thepipe.logger import get_logger
from distutils.version import LooseVersion

from . import IMAGE_NAME, DOCKER_URL

log = get_logger(basename(__file__))

MIN_SINGULARITY_VERSION = "3.3"


def is_singularity_version_greater(min_version):  # pragma: no cover
    singularity_version = LooseVersion(get_singularity_version().split()[-1])
    retval = singularity_version > LooseVersion(MIN_SINGULARITY_VERSION)
    return retval


def build_image(output_dir):
    if not isdir(output_dir):
        raise OSError("Directory not found!")
    image_path = join(output_dir, IMAGE_NAME)
    return Client.build(DOCKER_URL, image=image_path, sudo=False, ext="simg")
