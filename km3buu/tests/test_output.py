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
from km3net_testdata import data_path

TESTDATA_DIR = data_path("gibuu")


class TestXSection(unittest.TestCase):
    def test_xsection_all(self):
        filename = join(TESTDATA_DIR, XSECTION_FILENAMES["all"])
        xsection = read_nu_abs_xsection(filename)
        self.assertAlmostEqual(xsection['var'], 58.631)
        self.assertAlmostEqual(xsection['sum'], 8.0929)
        self.assertAlmostEqual(xsection['Delta'], 0.26805)
        self.assertAlmostEqual(xsection['highRES'], 0.14248)


class TestGiBUUOutput(unittest.TestCase):
    def setUp(self):
        self.output = GiBUUOutput(TESTDATA_DIR)

    def test_attr(self):
        assert hasattr(self.output, "df")

    def test_mean_xsec(self):
        df = self.output.df
        df = df.groupby(level=0).head(1)
        df = df[(df.lepIn_E > 0.7) & (df.lepIn_E < 1.0)]
        xsec = np.sum(df.xsec / df.lepIn_E)
        n_evts = self.output.flux_interpolation.integral(0.7, 1.0) / 0.02
        self.assertAlmostEqual(xsec / n_evts, 0.8, places=2)
