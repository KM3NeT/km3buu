#!/usr/bin/env python
# coding=utf-8
# Filename: test_ctrl.py

__author__ = "Johannes Schumann"
__copyright__ = "Copyright 2020, Johannes Schumann and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Johannes Schumann"
__email__ = "jschumann@km3net.de"
__status__ = "Development"

import csv
import pytest
import unittest
import numpy as np
from km3buu.jobcard import *
from km3buu.ctrl import run_jobcard
from tempfile import TemporaryDirectory, NamedTemporaryFile
from os import listdir, environ
from os.path import abspath, join, dirname
from thepipe.logger import get_logger
from km3net_testdata import data_path

TESTDATA_DIR = data_path("gibuu")

GIBUU_INSTALL_AVAILABLE = environ.get("CONTAINER_GIBUU_EXEC") is not None


@pytest.mark.skipif(not GIBUU_INSTALL_AVAILABLE, reason="GiBUU not installed")
class TestCTRLbyJobcardFile(unittest.TestCase):

    def setUp(self):
        self.filename = join(TESTDATA_DIR, "km3net_testdata.job")
        self.output_dir = TemporaryDirectory()
        self.flux_file = NamedTemporaryFile(suffix='.dat')
        with open(self.flux_file.name, 'w') as f:
            ene = np.linspace(0.1, 20, 100)
            writer = csv.writer(f, delimiter=' ')
            writer.writerows(zip(ene, np.power(ene, -1)))
        jc = read_jobcard(self.filename)
        jc["neutrino_induced"]["FileNameFlux"] = self.flux_file.name
        self.retval = run_jobcard(jc, self.output_dir.name, container=True)
        log = get_logger("ctrl.py")
        log.setLevel("INFO")

    def test_output(self):
        assert self.retval == 0

    def test_output_files_existing(self):
        files = listdir(self.output_dir.name)
        assert "EventOutput.Pert.00000001.root" in files


@pytest.mark.skipif(not GIBUU_INSTALL_AVAILABLE, reason="GiBUU not installed")
class TestCTRLbyJobcardObject(unittest.TestCase):

    def setUp(self):
        log = get_logger("ctrl.py")
        log.setLevel("INFO")
        self.test_jobcard = Jobcard()
        # NEUTRINO
        self.test_jobcard["neutrino_induced"]["process_ID"] = PROCESS_LOOKUP[
            "cc"]
        self.test_jobcard["neutrino_induced"]["flavor_ID"] = FLAVOR_LOOKUP[
            "electron"]
        self.test_jobcard["neutrino_induced"][
            "nuXsectionMode"] = XSECTIONMODE_LOOKUP["dSigmaMC"]
        self.test_jobcard["neutrino_induced"]["includeDIS"] = True
        self.test_jobcard["neutrino_induced"]["printAbsorptionXS"] = True
        self.test_jobcard["nl_SigmaMC"]["enu"] = 1
        # INPUT
        self.test_jobcard["input"]["numTimeSteps"] = 0
        self.test_jobcard["input"]["eventtype"] = 5
        self.test_jobcard["input"]["numEnsembles"] = 1
        self.test_jobcard["input"]["delta_T"] = 0.2
        self.test_jobcard["input"]["localEnsemble"] = True
        self.test_jobcard["input"]["num_runs_SameEnergy"] = 1
        self.test_jobcard["input"]["LRF_equals_CALC_frame"] = True
        # TARGET
        self.test_jobcard["target"]["z"] = 1
        self.test_jobcard["target"]["a"] = 1
        # MISC
        # self.test_jobcard["nl_neutrinoxsection"]["DISmassless"] =  True
        self.test_jobcard["neutrinoAnalysis"]["outputEvents"] = True
        self.test_jobcard["pythia"]["PARP(91)"] = 0.44
        self.output_dir = TemporaryDirectory()
        self.retval = run_jobcard(self.test_jobcard,
                                  self.output_dir.name,
                                  container=True)
        # raise Exception(self.test_jobcard)

    def test_output(self):
        assert self.retval == 0

    def test_output_files_existing(self):
        files = listdir(self.output_dir.name)
        assert "FinalEvents.dat" in files
