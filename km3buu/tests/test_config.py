#!/usr/bin/env python
# coding=utf-8
# Filename: test_ctrl.py

__author__ = "Johannes Schumann"
__copyright__ = "Copyright 2020, Johannes Schumann and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Johannes Schumann"
__email__ = "jschumann@km3net.de"
__status__ = "Development"

import os
import unittest
from unittest.mock import patch
import numpy as np
from km3buu.jobcard import *
from km3buu.config import Config, read_media_compositions

from tempfile import NamedTemporaryFile

TEST_CONFIG = """
[General]
km3net_lib_path=/tmp/km3net_lib_test
[GiBUU]
image_path=%s
[PROPOSAL]
itp_table_path=/tmp/.tables
[gSeaGen]
path=/tmp/gseagen
"""


class TestConfig(unittest.TestCase):

    def setUp(self):
        self.cfg_tmpfile = NamedTemporaryFile(delete=False)
        self.mock_image_file = NamedTemporaryFile(delete=False)
        with open(self.cfg_tmpfile.name, "w") as f:
            f.write(TEST_CONFIG % self.mock_image_file.name)
        self.cfg = Config(self.cfg_tmpfile.name)

    def tearDown(self):
        os.remove(self.cfg_tmpfile.name)
        os.remove(self.mock_image_file.name)

    def test_general_section(self):
        assert self.cfg.km3net_lib_path == "/tmp/km3net_lib_test"

    def test_gibuu_section(self):
        assert self.cfg.gibuu_image_path == self.mock_image_file.name

    def test_proposal_section(self):
        assert self.cfg.proposal_itp_tables == "/tmp/.tables"

    def test_gseagen_path(self):
        assert self.cfg.gseagen_path == "/tmp/gseagen"
