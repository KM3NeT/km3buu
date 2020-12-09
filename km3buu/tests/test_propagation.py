#!/usr/bin/env python
# coding=utf-8
# Filename: test_propagation.py

__author__ = "Johannes Schumann"
__copyright__ = "Copyright 2020, Johannes Schumann and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Johannes Schumann"
__email__ = "jschumann@km3net.de"
__status__ = "Development"

import csv
import unittest
import numpy as np
import uproot
from os.path import abspath, join, dirname
from thepipe.logger import get_logger
from km3net_testdata import data_path

import proposal as pp

from km3buu.output import GiBUUOutput
from km3buu.propagation import propagate_lepton

TESTDATA_DIR = data_path("gibuu")

pp.RandomGenerator.get().set_seed(1234)


class TestTauPropagation(unittest.TestCase):
    def setUp(self):
        log = get_logger("ctrl.py")
        log.setLevel("INFO")
        self.gibuu_output = GiBUUOutput(TESTDATA_DIR)
        fname = join(TESTDATA_DIR, self.gibuu_output.root_pert_files[0])
        fobj = uproot.open(fname)
        data = fobj["RootTuple"].arrays()
        self.sec = propagate_lepton(data, 15)

    def test_secondary_momenta(self):
        np.testing.assert_array_almost_equal(np.array(self.sec[0].E),
                                             [0.535, 1.316, 0.331],
                                             decimal=3)
        np.testing.assert_array_almost_equal(np.array(self.sec[0].Px),
                                             [-0.467, 0.321, -0.246],
                                             decimal=3)
        np.testing.assert_array_almost_equal(np.array(self.sec[0].Py),
                                             [0.127, -0.822, 0.218],
                                             decimal=3)
        np.testing.assert_array_almost_equal(np.array(self.sec[0].Pz),
                                             [0.179, 0.967, -0.041],
                                             decimal=3)

    def test_secondary_types(self):
        np.testing.assert_array_equal(np.array(self.sec[0].barcode),
                                      [13, 16, -14])

    def test_secondary_positions(self):
        np.testing.assert_array_almost_equal(np.array(self.sec[0].x), [0, 0],
                                             decimal=1)
        np.testing.assert_array_almost_equal(np.array(self.sec[0].y), [0, 0],
                                             decimal=1)
        np.testing.assert_array_almost_equal(np.array(self.sec[0].z), [0, 0],
                                             decimal=1)
