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

import unittest
import numpy as np
from km3buu.jobcard import *
from km3buu.ctrl import run_jobcard
from tempfile import TemporaryDirectory
from os import listdir
from thepipe.logger import get_logger


class TestCTRL(unittest.TestCase):
    def setUp(self):
        log = get_logger("ctrl.py")
        log.setLevel("INFO")
        self.test_jobcard = Jobcard()
        # NEUTRINO
        self.test_jobcard.set_property("neutrino_induced", "process_ID",
                                       PROCESS_LOOKUP["cc"])
        self.test_jobcard.set_property("neutrino_induced", "flavor_ID",
                                       FLAVOR_LOOKUP["electron"])
        self.test_jobcard.set_property("neutrino_induced", "nuXsectionMode",
                                       XSECTIONMODE_LOOKUP["dSigmaMC"])
        self.test_jobcard.set_property("neutrino_induced", "includeDIS", True)
        self.test_jobcard.set_property("neutrino_induced", "printAbsorptionXS",
                                       "T")
        self.test_jobcard.set_property("nl_SigmaMC", "enu", 1)
        # INPUT
        self.test_jobcard.set_property("input", "numTimeSteps", 0)
        self.test_jobcard.set_property("input", "eventtype", 5)
        self.test_jobcard.set_property("input", "numEnsembles", 1)
        self.test_jobcard.set_property("input", "delta_T", 0.2)
        self.test_jobcard.set_property("input", "localEnsemble", True)
        self.test_jobcard.set_property("input", "num_runs_SameEnergy", 1)
        self.test_jobcard.set_property("input", "LRF_equals_CALC_frame", True)
        # TARGET
        self.test_jobcard.set_property("target", "target_Z", 1)
        self.test_jobcard.set_property("target", "target_A", 1)
        # MISC
        # self.test_jobcard.set_property("nl_neutrinoxsection", "DISmassless", True)
        self.test_jobcard.set_property("neutrinoAnalysis", "outputEvents",
                                       True)
        self.test_jobcard.set_property("pythia", "PARP(91)", 0.44)
        self.output_dir = TemporaryDirectory()
        self.retval = run_jobcard(self.test_jobcard, self.output_dir.name)

    def test_output(self):
        assert self.retval == 0

    def test_output_files_existing(self):
        files = listdir(self.output_dir.name)
        assert "FinalEvents.dat" in files
