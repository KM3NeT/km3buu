#!/usr/bin/env python
# coding=utf-8
# Filename: test_jobcard.py

__author__ = "Johannes Schumann"
__copyright__ = "Copyright 2020, Johannes Schumann and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Johannes Schumann"
__email__ = "jschumann@km3net.de"
__status__ = "Development"

import unittest
import numpy as np
import pytest
from os import environ
from itertools import combinations
from km3buu.jobcard import *
from km3buu.output import GiBUUOutput
from tempfile import TemporaryFile, TemporaryDirectory

GIBUU_INSTALL_AVAILABLE = environ.get("CONTAINER_GIBUU_EXEC") is not None

if GIBUU_INSTALL_AVAILABLE:
    from km3buu.ctrl import run_jobcard


class TestUtils(unittest.TestCase):

    def test_event_numbers(self):
        e, r = estimate_number_of_ensembles(1000, (1, 1))
        assert e == 2000
        assert r == 1
        e, r = estimate_number_of_ensembles(1000000, (1, 1))
        assert e == 100000
        assert r == 20
        e, r = estimate_number_of_ensembles(100000, (16, 8))
        assert e == 6250
        assert r == 2

    def test_minimal_ensembles(self):
        e, r = estimate_number_of_ensembles(1000, (56, 26))
        assert e == 100
        assert r == 1


class TestJobcard(unittest.TestCase):

    def setUp(self):
        self.test_jobcard = Jobcard()
        # Insert some test elements
        self.test_jobcard["ABC"]["def"] = 42

    def test_input_path(self):
        expected_line = "path_to_input = '%s'" % INPUT_PATH
        ctnt = str(self.test_jobcard)
        group_start = ctnt.find("&input")
        group_end = ctnt.find("/\n", group_start)
        assert ctnt[group_start:group_end].find(expected_line) != -1

    def test_elements(self):
        ctnt = str(self.test_jobcard)
        expected_line = "def = 42"
        group_start = ctnt.find("&abc")
        group_end = ctnt.find("/", group_start)
        print(ctnt)
        assert ctnt[group_start:group_end].find(expected_line) != -1

    def test_remove_elements(self):
        del self.test_jobcard["ABC"]["def"]
        ctnt = str(self.test_jobcard)
        expected_line = "def = 42"
        assert ctnt.find("&ABC") == -1
        assert ctnt.find(expected_line) == -1


class TestNeutrinoEnergyRangeJobcard(unittest.TestCase):

    def setUp(self):
        self.test_fluxfile = TemporaryFile()
        self.test_Z = np.random.randint(1, 100)
        self.test_A = np.random.randint(self.test_Z, 100)
        self.test_energy_min = np.random.uniform(0.0, 100.0)
        self.test_energy_max = np.random.uniform(self.test_energy_min, 100.0)
        self.photon_propagation_flag = np.random.choice([True, False])
        self.do_decay = np.random.choice([True, False])
        self.test_jobcard = generate_neutrino_jobcard(
            1000,
            1,
            "CC",
            "electron", (self.test_energy_min, self.test_energy_max),
            (self.test_A, self.test_Z),
            do_decay=self.do_decay,
            photon_propagation=self.photon_propagation_flag,
            fluxfile=self.test_fluxfile.name,
            input_path="/test")

    def test_input_path(self):
        self.assertEqual("/test", self.test_jobcard["input"]["path_to_input"])

    def test_target(self):
        self.assertEqual(self.test_Z, self.test_jobcard["target"]["Z"])
        self.assertEqual(self.test_A, self.test_jobcard["target"]["A"])

    def test_energy(self):
        self.assertAlmostEqual(
            self.test_energy_min,
            self.test_jobcard["nl_neutrino_energyflux"]["eflux_min"])
        self.assertAlmostEqual(
            self.test_energy_max,
            self.test_jobcard["nl_neutrino_energyflux"]["eflux_max"])

    def test_flux(self):
        self.assertEqual(self.test_fluxfile.name,
                         self.test_jobcard["neutrino_induced"]["FileNameflux"])
        self.assertEqual(self.test_jobcard["neutrino_induced"]["nuexp"], 99)
        # Test flat flux
        flat_flux_jobcard = generate_neutrino_jobcard(
            1000, 1, "CC", "electron",
            (self.test_energy_min, self.test_energy_max),
            (self.test_Z, self.test_A))
        self.assertEqual(flat_flux_jobcard["neutrino_induced"]["nuexp"], 10)

    def test_photon_propagation_flag(self):
        self.assertEqual(self.test_jobcard["insertion"]["propagateNoPhoton"],
                         not self.photon_propagation_flag)


class TestNeutrinoSingleEnergyJobcard(unittest.TestCase):

    def setUp(self):
        self.test_fluxfile = TemporaryFile()
        self.test_Z = np.random.randint(1, 100)
        self.test_A = np.random.randint(self.test_Z, 100)
        self.test_energy = np.random.uniform(0.0, 100.0)
        self.photon_propagation_flag = np.random.choice([True, False])
        self.do_decay = np.random.choice([True, False])
        self.test_jobcard = generate_neutrino_jobcard(
            1000,
            1,
            "CC",
            "electron",
            self.test_energy, (self.test_A, self.test_Z),
            do_decay=self.do_decay,
            photon_propagation=self.photon_propagation_flag,
            fluxfile=self.test_fluxfile.name,
            input_path="/test")

    def test_input_path(self):
        self.assertEqual("/test", self.test_jobcard["input"]["path_to_input"])

    def test_target(self):
        self.assertEqual(self.test_Z, self.test_jobcard["target"]["z"])
        self.assertEqual(self.test_A, self.test_jobcard["target"]["a"])

    def test_energy(self):
        self.assertAlmostEqual(self.test_energy,
                               self.test_jobcard["nl_sigmamc"]["enu"])

    def test_photon_propagation_flag(self):
        self.assertEqual(self.test_jobcard["insertion"]["propagateNoPhoton"],
                         not self.photon_propagation_flag)


@pytest.mark.skipif(not GIBUU_INSTALL_AVAILABLE, reason="GiBUU not installed")
class TestJobcardSeed(unittest.TestCase):

    def setUp(self):
        jc = generate_neutrino_jobcard(100,
                                       "CC",
                                       "electron", (1.0, 2.0), (1, 1),
                                       do_decay=False,
                                       photon_propagation=False,
                                       seed=1234)
        self.dfs = []
        for i in range(2):
            output_dir = TemporaryDirectory()
            run_jobcard(jc, output_dir.name, container=True)
            self.dfs.append(GiBUUOutput(output_dir.name).df)

    def test_output(self):
        for a, b in combinations(self.dfs, 2):
            assert all((a == b).all(1))
