#!/usr/bin/env python
# coding=utf-8
# Filename: test_physics.py

__author__ = "Johannes Schumann"
__copyright__ = "Copyright 2021, Johannes Schumann and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Johannes Schumann"
__email__ = "jschumann@km3net.de"
__status__ = "Development"

import unittest
from unittest.mock import patch
import numpy as np
import pytest
from os.path import abspath, join, dirname

from km3buu.physics import *

TESTFILE = join(dirname(__file__), "data/visible_energy_tables.txt")


class TestVisibleEnergyWeights(unittest.TestCase):
    def setUp(self):
        self.ref_values = np.loadtxt(TESTFILE).T

    def test_ngamma_elec(self):
        vfunc = np.vectorize(number_photons_per_electron)
        val = vfunc(self.ref_values[0, :])
        assert np.allclose(self.ref_values[1, :], val, rtol=0.05)

    def test_pion_weight(self):
        vfunc = np.vectorize(pion_weight)
        val = vfunc(self.ref_values[0, :])
        assert np.allclose(self.ref_values[2, :], val, rtol=0.05)

    def test_kaon_weight(self):
        vfunc = np.vectorize(kaon_weight)
        val = vfunc(self.ref_values[0, :])
        assert np.allclose(self.ref_values[3, :], val, rtol=0.05)

    def test_kshort_weight(self):
        vfunc = np.vectorize(kshort_weight)
        val = vfunc(self.ref_values[0, :])
        assert np.allclose(self.ref_values[4, :], val, rtol=0.05)

    def test_klong_weight(self):
        vfunc = np.vectorize(klong_weight)
        val = vfunc(self.ref_values[0, :])
        assert np.allclose(self.ref_values[5, :], val, rtol=0.05)

    def test_proton_weight(self):
        vfunc = np.vectorize(proton_weight)
        val = vfunc(self.ref_values[0, :])
        assert np.allclose(self.ref_values[6, :], val, atol=0.05)

    def test_neutron_weight(self):
        vfunc = np.vectorize(neutron_weight)
        val = vfunc(self.ref_values[0, :])
        assert np.allclose(self.ref_values[7, :], val, rtol=0.05)

    def test_high_ene_weights(self):
        vfunc = np.vectorize(high_energy_weight)
        val = vfunc(self.ref_values[0, :])
        assert np.allclose(self.ref_values[8, :], val, rtol=0.05)
