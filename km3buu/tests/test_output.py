#!/usr/bin/env python
# coding=utf-8
# Filename: test_output.py

__author__ = "Johannes Schumann"
__copyright__ = "Copyright 2020, Johannes Schumann and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Johannes Schumann"
__email__ = "jschumann@km3net.de"
__status__ = "Development"

import unittest
import numpy as np
from km3buu.output import *
from os import listdir
from os.path import abspath, join, dirname

TESTDATA_DIR = abspath(join(dirname(__file__), "../data/test-data/"))


class TestFinalEvents(unittest.TestCase):
    def setUp(self):
        self.filename = join(TESTDATA_DIR, "FinalEvents.dat")
        self.final_events = FinalEvents(self.filename)

    def test_values(self):
        assert self.final_events[0]["id"] == 901
        assert self.final_events[0]["charge"] == -1
        self.assertAlmostEqual(self.final_events[0]["perweight"], 6.154773e-1)
        self.assertAlmostEqual(self.final_events[0]["p_t"], 5.050394e-1)
        self.assertAlmostEqual(self.final_events[0]["p_x"], 2.619802e-2)
        self.assertAlmostEqual(self.final_events[0]["p_y"], 3.290991e-1)
        self.assertAlmostEqual(self.final_events[0]["p_z"], 3.821936e-1)
        self.assertAlmostEqual(self.final_events[0]["energy"], 1.0)
        assert self.final_events[3]["id"] == 1
        assert self.final_events[3]["charge"] == 1
        self.assertAlmostEqual(self.final_events[3]["perweight"], 6.154773e-1)

    def test_index(self):
        assert self.final_events[0] is not None

    def test_slicing(self):
        assert self.final_events[0:2] is not None


class TestGiBUUOutput(unittest.TestCase):
    def setUp(self):
        pass
