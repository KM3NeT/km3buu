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


class TestXSection(unittest.TestCase):
    def test_xsection_all(self):
        filename = join(TESTDATA_DIR, XSECTION_FILENAMES["all"])
        xsection = NeutrinoAbsorptionXSection(filename)
        self.assertAlmostEqual(xsection['var'], 1.0)
        self.assertAlmostEqual(xsection['sum'], 0.61548)
        self.assertAlmostEqual(xsection['Delta'], 0.61537)
        self.assertAlmostEqual(xsection['highRES'], 1.0661e-4)
        self.assertAlmostEqual(xsection['Delta'], 0.61537)


class TestFinalEvents(unittest.TestCase):
    def setUp(self):
        self.filename = join(TESTDATA_DIR, EVENT_FILENAME)
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

    def test_length(self):
        assert len(self.final_events) == 5


class TestGiBUUOutput(unittest.TestCase):
    def setUp(self):
        self.output = GiBUUOutput(TESTDATA_DIR)

    def test_attr(self):
        assert hasattr(self.output, "events")
