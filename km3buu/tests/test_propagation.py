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
import pytest
from os.path import abspath, join, dirname
from thepipe.logger import get_logger

import awkward as ak

try:
    import proposal as pp
    pp.RandomGenerator.get().set_seed(1234)
    PROPOSAL_AVAILABLE = True
except ModuleNotFoundError as e:
    print(e)
    PROPOSAL_AVAILABLE = False

from km3buu.geometry import *
if PROPOSAL_AVAILABLE:
    from km3buu.propagation import *

np.random.seed(1234)


@pytest.mark.skipif(not PROPOSAL_AVAILABLE,
                    reason="PROPOSAL installation required")
class TestTauPropagation(unittest.TestCase):

    def setUp(self):
        prop = Propagator([15, -15], CANVolume().make_proposal_geometries())
        self.position = np.array([10.0, 10.0, 100.0])
        self.direction = np.array([1., 1., 0])
        self.energy = 10.0
        self.sec = prop.propagate(15, self.energy, self.position,
                                  self.direction)

    def test_secondary_momenta(self):
        np.testing.assert_array_almost_equal(np.array(self.sec.E),
                                             [4.141, 5.761, 0.098],
                                             decimal=3)
        np.testing.assert_array_almost_equal(np.array(self.sec.Dx),
                                             [0.79, 0.642, -0.089],
                                             decimal=3)
        np.testing.assert_array_almost_equal(np.array(self.sec.Dy),
                                             [0.599, 0.761, 0.984],
                                             decimal=3)
        np.testing.assert_array_almost_equal(np.array(self.sec.Dz),
                                             [-0.133, 0.098, -0.152],
                                             decimal=3)

    def test_secondary_types(self):
        np.testing.assert_array_equal(np.array(self.sec.barcode),
                                      [11, 16, -12])

    def test_secondary_positions(self):
        np.testing.assert_array_almost_equal(np.array(self.sec.x),
                                             [10.0, 10.0, 10.0],
                                             decimal=1)
        np.testing.assert_array_almost_equal(np.array(self.sec.y),
                                             [10.0, 10.0, 10.0],
                                             decimal=1)
        np.testing.assert_array_almost_equal(np.array(self.sec.z),
                                             [100., 100., 100.],
                                             decimal=1)


@pytest.mark.skipif(not PROPOSAL_AVAILABLE,
                    reason="PROPOSAL installation required")
class TestMuonPropagation(unittest.TestCase):

    def setUp(self):
        prop = Propagator([13, -13],
                          CylindricalVolume().make_proposal_geometries())
        self.position = np.array([200.0, 200.0, 100.0])
        self.direction = np.array([-1., -1., 0])
        self.energy = 100.0
        self.sec = prop.propagate(13, self.energy, self.position,
                                  self.direction)

    def test_secondary_momenta(self):
        np.testing.assert_array_almost_equal(np.array(self.sec.E), [77.102],
                                             decimal=3)
        np.testing.assert_array_almost_equal(np.array(self.sec.Dx), [-0.707],
                                             decimal=3)
        np.testing.assert_array_almost_equal(np.array(self.sec.Dy), [-0.707],
                                             decimal=3)
        np.testing.assert_array_almost_equal(np.array(self.sec.Dz), [-0.],
                                             decimal=3)

    def test_secondary_types(self):
        np.testing.assert_array_equal(np.array(self.sec.barcode), [13])

    def test_secondary_positions(self):
        np.testing.assert_array_almost_equal(np.array(self.sec.x), [141.7],
                                             decimal=1)
        np.testing.assert_array_almost_equal(np.array(self.sec.y), [141.7],
                                             decimal=1)
        np.testing.assert_array_almost_equal(np.array(self.sec.z), [100.],
                                             decimal=1)
