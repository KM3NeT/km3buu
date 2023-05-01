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


class TestNoVolume(unittest.TestCase):

    def setUp(self):
        self.detector_geometry = NoVolume()

    def test_volume(self):
        volume = self.detector_geometry.volume
        self.assertAlmostEqual(volume, 1.0)

    def test_random_pos(self):
        pos = self.detector_geometry.random_pos()
        np.testing.assert_array_almost_equal(pos, 0.0)

    def test_random_dir(self):
        direction = self.detector_geometry.random_dir()
        assert direction[0] == 0.0
        assert direction[1] == 1.0


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

    def test_limited_zenith(self):
        np.random.seed(1234)
        geometry = CANVolume(zenith=(-0.4, 0.5))
        self.assertAlmostEqual(geometry.solid_angle, 5.654866776461628)
        direction = geometry.random_dir()
        self.assertAlmostEqual(direction[1], 0.15989789393584863)
        geometry = CANVolume(zenith=(0.1, 0.3))
        direction = geometry.random_dir()
        self.assertAlmostEqual(direction[1], 0.25707171674275386)
        geometry = CANVolume(zenith=(-0.3, -0.2))
        direction = geometry.random_dir()
        self.assertAlmostEqual(direction[1], -0.2727407394717358)


class TestCan(unittest.TestCase):

    def setUp(self):
        self.detector_geometry = CANVolume()

    def test_volume(self):
        volume = self.detector_geometry.volume
        self.assertAlmostEqual(volume, 243604084.28, 2)

    def test_in_can(self):
        assert ~self.detector_geometry.in_can((300, 300, 100))
        assert self.detector_geometry.in_can((100, 100, 100))

    def test_header(self):
        header = self.detector_geometry.header_entries(1234)
        assert header["genvol"] == "0.000 476.500 403.400 243604084.277 1234"
        assert header["fixedcan"] == "0.000 0.000 0.000 476.500 403.400"

    def test_position(self):
        np.random.seed(1234)
        pos = self.detector_geometry.random_pos()
        self.assertAlmostEqual(pos[0], -127.07940486491587)
        self.assertAlmostEqual(pos[1], -122.54421157149173)
        self.assertAlmostEqual(pos[2], 208.57726763689004)

    def test_position(self):
        np.random.seed(1234)
        detector_geometry = CANVolume(detector_center=(100, 100))
        pos = detector_geometry.random_pos()
        self.assertAlmostEqual(pos[0], -27.07940486491587)
        self.assertAlmostEqual(pos[1], -22.54421157149173)
        self.assertAlmostEqual(pos[2], 208.57726763689004)

    def test_limited_zenith(self):
        np.random.seed(1234)
        geometry = CANVolume(zenith=(-0.4, 0.5))
        self.assertAlmostEqual(geometry.solid_angle, 5.654866776461628)
        direction = geometry.random_dir()
        self.assertAlmostEqual(direction[1], 0.15989789393584863)
        geometry = CANVolume(zenith=(0.1, 0.3))
        direction = geometry.random_dir()
        self.assertAlmostEqual(direction[1], 0.25707171674275386)
        geometry = CANVolume(zenith=(-0.3, -0.2))
        direction = geometry.random_dir()
        self.assertAlmostEqual(direction[1], -0.2727407394717358)


class TestCylindricalVolume(unittest.TestCase):

    def setUp(self):
        np.random.seed(1234)
        self.detector_geometry = CylindricalVolume()

    def test_volume(self):
        volume = self.detector_geometry.volume
        self.assertAlmostEqual(volume, 589048622.55, 2)

    def test_in_can(self):
        assert ~self.detector_geometry.in_can((300, 300, 100))
        assert self.detector_geometry.in_can((100, 100, 100))

    def test_header(self):
        header = self.detector_geometry.header_entries(1234)
        assert header[
            "genvol"] == "-100.000 650.000 500.000 589048622.548 1234"
        assert header["fixedcan"] == "0.000 0.000 0.000 350.000 200.400"

    def test_single_position(self):
        pos = self.detector_geometry.random_pos()
        np.testing.assert_array_almost_equal(
            pos, [-157.510418, -151.889206, 228.295804])

    def test_multiple_positions(self):
        pos = self.detector_geometry.random_pos(10)
        np.testing.assert_array_almost_equal(
            pos[0, :], [-137.152421, 170.496557, 173.664488])
        np.testing.assert_array_almost_equal(
            pos[9, :], [346.394334, -314.633317, 326.073989])

    def test_single_direction(self):
        direction = self.detector_geometry.random_dir()
        np.testing.assert_array_almost_equal(direction, [1.203352, 0.244218])

    def test_multiple_directions(self):
        directions = self.detector_geometry.random_dir(10)
        np.testing.assert_array_almost_equal(directions[0, :],
                                             [1.203352, -0.284365])
        np.testing.assert_array_almost_equal(directions[9, :],
                                             [5.503647, 0.765282])
