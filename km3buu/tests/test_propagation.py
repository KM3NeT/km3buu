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
from km3buu.propagation import propagate_lepton

pp.RandomGenerator.get().set_seed(1234)


@pytest.mark.skip(reason="CI boost lib problem")
class TestTauPropagation(unittest.TestCase):
    def setUp(self):
        data = ak.Array({
            "lepOut_E": [
                17.45830624434573, 3.1180989952362594, 21.270059768902005,
                5.262659790136034, 23.52185741888274
            ],
            "lepOut_Px": [
                -0.42224402086330426, -1.0232258668453014, -0.5801431899058521,
                -0.9038349288874724, 0.9022573877437422
            ],
            "lepOut_Py": [
                0.3644190693190108, -0.24542303987320932, 0.24499631087268617,
                -1.1060562370375715, -3.982173292871768
            ],
            "lepOut_Pz": [
                17.35867612031871, 2.336148261778657, 21.186342871282157,
                4.743161507744377, 23.096499191566885
            ]
        })
        self.sec = propagate_lepton(data, 15)

    def test_secondary_momenta(self):
        np.testing.assert_array_almost_equal(np.array(self.sec[0].E),
                                             [2.182, 13.348, 1.928],
                                             decimal=3)
        np.testing.assert_array_almost_equal(np.array(self.sec[0].Px),
                                             [0.295, -0.48, -0.237],
                                             decimal=3)
        np.testing.assert_array_almost_equal(np.array(self.sec[0].Py),
                                             [-0.375, 0.784, -0.044],
                                             decimal=3)
        np.testing.assert_array_almost_equal(np.array(self.sec[0].Pz),
                                             [2.129, 13.316, 1.913],
                                             decimal=3)

    def test_secondary_types(self):
        np.testing.assert_array_equal(np.array(self.sec[0].barcode),
                                      [13, 16, -14])

    def test_secondary_positions(self):
        np.testing.assert_array_almost_equal(np.array(self.sec[0].x),
                                             [-1.4e-05, -1.4e-05, -1.4e-05],
                                             decimal=1)
        np.testing.assert_array_almost_equal(np.array(self.sec[0].y),
                                             [1.2e-05, 1.2e-05, 1.2e-05],
                                             decimal=1)
        np.testing.assert_array_almost_equal(np.array(self.sec[0].z),
                                             [0., 0., 0.],
                                             decimal=1)
