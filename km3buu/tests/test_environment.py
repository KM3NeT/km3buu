#!/usr/bin/env python
# coding=utf-8
# Filename: test_environment.py

__author__ = "Johannes Schumann"
__copyright__ = "Copyright 2020, Johannes Schumann and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Johannes Schumann"
__email__ = "jschumann@km3net.de"
__status__ = "Development"

import unittest
from unittest.mock import patch
from km3buu.environment import *
from os.path import dirname, join
from spython.main import Client
from km3buu import DOCKER_URL, IMAGE_NAME


class TestBuild(unittest.TestCase):
    def test_wrong_dir_path(self):
        wrong_path = "foobar"
        try:
            build_image(wrong_path)
            assert False
        except OSError as e:
            assert str(e) == "Directory not found!"

    @patch.object(Client, 'build', return_value=123)
    def test_build_cmd(self, function):
        existing_path = dirname(__file__)
        assert build_image(existing_path) == 123
        expected_image_path = join(existing_path, IMAGE_NAME)
        function.assert_called_once_with(DOCKER_URL,
                                         image=expected_image_path,
                                         sudo=False,
                                         ext="simg")
