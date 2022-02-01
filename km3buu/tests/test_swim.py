#!/usr/bin/env python
# coding=utf-8
# Filename: test_swim.py

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
import uproot
from tempfile import TemporaryFile
from km3buu.swim import *

try:
    import ROOT
    ROOT_AVAILABLE = True
except ModuleNotFoundError:
    ROOT_AVAILABLE = False


class TestSWIMLabels(unittest.TestCase):
    def test_xsection_labels(self):
        assert "gnum_CC_E" == build_swim_xsec_label("muon", "CC", False)
        assert "gnut_CC_logE" == build_swim_xsec_label("tau", "CC", False,
                                                       True)
        assert "gnbe_CC_E" == build_swim_xsec_label("electron", "CC", True)

    def test_bjorkeny_labels(self):
        assert "h2d_bydist_E_numu_cc" == build_swim_by_label(
            "muon", "CC", False)
        assert "h2d_bydist_E_nutau_cc" == build_swim_by_label(
            "tau", "CC", False)
        assert "h2d_bydist_E_anue_cc" == build_swim_by_label(
            "electron", "CC", True)


@pytest.mark.skipif(not ROOT_AVAILABLE, reason="PyROOT required")
class TestSWIMFiles(unittest.TestCase):
    def setUp(self):
        self.filename = TemporaryFile(suffix=".root")
        lbl = build_swim_xsec_label("electron", "CC", True)
        write_swim_xsec_file(np.linspace(0, 1, 10),
                             np.linspace(0, 1, 10),
                             lbl,
                             filename=str(self.filename.name))

    def test_xsec_keys(self):
        fobj = uproot.open(str(self.filename.name))
        assert type(fobj["single_graphs/gnue_CC_E"]
                    ) == uproot.models.TGraph.Model_TGraph_v4
