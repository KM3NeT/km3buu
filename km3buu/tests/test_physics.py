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
import re
import numpy as np
import pytest
from os.path import abspath, join, dirname
from particle import Particle

from km3buu.physics import *

FUNCTIONS_TESTFILE = join(dirname(__file__),
                          "data/visible_energy_weight_functions.txt")
PARTICLE_TESTFILE = join(dirname(__file__),
                         "data/visible_energy_particle_frac.txt")
MUON_TESTFILE = join(dirname(__file__), "data/muon_range_seawater.txt")


class TestKineticEnergy(unittest.TestCase):

    def test_electron_mass(self):
        val = get_kinetic_energy(0.51099895e-3, 11)[0]
        self.assertAlmostEqual(val, 0.0, 6)

    def test_negative_value(self):
        with self.assertWarnsRegex(RuntimeWarning,
                                   "invalid value encountered in sqrt"):
            get_kinetic_energy(0.0, 11)

    def test_suppress_warning(self):
        import warnings
        with warnings.catch_warnings(record=True) as w:
            get_kinetic_energy(0.0, 11, False)
            assert len(w) == 0


class TestMuonRangeSeaWater(unittest.TestCase):

    def setUp(self):
        self.ref_values = np.loadtxt(MUON_TESTFILE).T

    def test_particles(self):
        assert np.allclose(muon_range_seawater(self.ref_values[0, :],
                                               self.ref_values[1, :]),
                           self.ref_values[2, :],
                           rtol=0.01)


class TestMuonVisibleEnergy(unittest.TestCase):

    def test_muon_energies(self):
        np.testing.assert_almost_equal(visible_energy_fraction(10.0, 13),
                                       0.8516107)
        np.testing.assert_almost_equal(visible_energy_fraction(100.0, 13),
                                       0.7379085)
        np.testing.assert_almost_equal(visible_energy_fraction(1000.0, 13),
                                       0.4959454)


class TestVisEnergyParticle(unittest.TestCase):

    def setUp(self):
        with open(PARTICLE_TESTFILE, "r") as f:
            tmp = f.readline()
            self.particles = [
                int(p[2:-1]) for p in re.findall(r'\s\(-?\d+\)', tmp)
            ]
        self.ref_values = np.loadtxt(PARTICLE_TESTFILE).T

    def test_particles(self):
        for i, pdgid in enumerate(self.particles):
            val = km3_opa_fraction(self.ref_values[0, :], pdgid)
            assert np.allclose(self.ref_values[i + 1, :],
                               val,
                               rtol=0.05,
                               atol=0.01)


class TestVisEnergyWeightFunctions(unittest.TestCase):

    def setUp(self):
        self.ref_values = np.loadtxt(FUNCTIONS_TESTFILE).T

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
