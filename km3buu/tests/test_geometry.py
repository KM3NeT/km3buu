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

    def test_position(self):
        np.random.seed(1234)
        pos = self.detector_geometry.random_pos()
        self.assertAlmostEqual(pos[0], -127.07940486491587)
        self.assertAlmostEqual(pos[1], -122.54421157149173)
        self.assertAlmostEqual(pos[2], 208.57726763689004)

    def test_position(self):
        np.random.seed(1234)
        detector_geometry = CanVolume(detector_center=(100, 100))
        pos = detector_geometry.random_pos()
        self.assertAlmostEqual(pos[0], -27.07940486491587)
        self.assertAlmostEqual(pos[1], -22.54421157149173)
        self.assertAlmostEqual(pos[2], 208.57726763689004)

    def test_limited_zenith(self):
        np.random.seed(1234)
        geometry = CanVolume(zenith=(-0.4, 0.5))
        direction = geometry.random_dir()
        self.assertAlmostEqual(direction[1], 0.15989789393584863)
        geometry = CanVolume(zenith=(0.1, 0.3))
        direction = geometry.random_dir()
        self.assertAlmostEqual(direction[1], 0.25707171674275386)
        geometry = CanVolume(zenith=(-0.3, -0.2))
        direction = geometry.random_dir()
        self.assertAlmostEqual(direction[1], -0.2727407394717358)
