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

from km3buu.geometry import *
import numpy as np


class TestGeneralGeometry(unittest.TestCase):

    def test_abstract_init(self):
        with self.assertRaises(TypeError) as ctx:
            d = DetectorVolume()


class TestSphere(unittest.TestCase):

    def setUp(self):
        self.detector_geometry = SphericalVolume(20, (2, 2, 2))

    def test_volume(self):
        volume = self.detector_geometry.volume
        self.assertAlmostEqual(volume, 33510.32, 2)

    def test_random_pos(self):
        for i in range(50):
            pos = self.detector_geometry.random_pos()
            assert pos[0] > -18.0
            assert pos[1] > -18.0
            assert pos[2] > -18.0
            assert pos[0] < 22.0
            assert pos[1] < 22.0
            assert pos[2] < 22.0
            radius = np.sqrt(np.sum(np.power((np.array(pos) - 2), 2)))
            assert radius <= 20


class TestCan(unittest.TestCase):

    def setUp(self):
        self.detector_geometry = CanVolume()

    def test_volume(self):
        volume = self.detector_geometry.volume
        self.assertAlmostEqual(volume, 243604084.28, 2)
