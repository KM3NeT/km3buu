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

import proposal as pp

import awkward as ak

from km3buu.geometry import *
from km3buu.propagation import *

pp.RandomGenerator.get().set_seed(1234)
np.random.seed(1234)


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
        np.testing.assert_array_almost_equal(np.array(self.sec.Px),
                                             [3.271, 3.696, -0.009],
                                             decimal=3)
        np.testing.assert_array_almost_equal(np.array(self.sec.Py),
                                             [2.48, 4.382, 0.097],
                                             decimal=3)
        np.testing.assert_array_almost_equal(np.array(self.sec.Pz),
                                             [-0.549, 0.564, -0.015],
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
        np.testing.assert_array_almost_equal(np.array(self.sec.Px), [-54.519],
                                             decimal=3)
        np.testing.assert_array_almost_equal(np.array(self.sec.Py), [-54.519],
                                             decimal=3)
        np.testing.assert_array_almost_equal(np.array(self.sec.Pz), [-0.],
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
