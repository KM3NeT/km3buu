#!/usr/bin/env python
# coding=utf-8
# Filename: test_jobcard.py

__author__ = "Johannes Schumann"
__copyright__ = "Copyright 2020, Johannes Schumann and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Johannes Schumann"
__email__ = "jschumann@km3net.de"
__status__ = "Development"

import unittest
import numpy as np
from km3buu.jobcard import Jobcard, INPUT_PATH


class TestJobcard(unittest.TestCase):
    def setUp(self):
        self.test_jobcard = Jobcard()
        # Insert some test elements
        self.test_jobcard.set_property("ABC", "def", 42)

    def test_input_path(self):
        expected_line = "path_to_input = '%s'" % INPUT_PATH
        ctnt = str(self.test_jobcard)
        group_start = ctnt.find("&input")
        group_end = ctnt.find("/\n\n", group_start)
        assert ctnt[group_start:group_end].find(expected_line) != -1

    def test_elements(self):
        ctnt = str(self.test_jobcard)
        expected_line = "def = 42"
        group_start = ctnt.find("&ABC")
        group_end = ctnt.find("/\n\n", group_start)
        assert ctnt[group_start:group_end].find(expected_line) != -1

    def test_remove_elements(self):
        self.test_jobcard.remove_property("ABC", "def")
        ctnt = str(self.test_jobcard)
        expected_line = "def = 42"
        assert ctnt.find("&ABC") == -1
        assert ctnt.find(expected_line) == -1
