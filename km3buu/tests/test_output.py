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
import numpy as np
import pytest
import km3io
from km3buu.output import *
from os import listdir
from os.path import abspath, join, dirname
from km3net_testdata import data_path
from tempfile import NamedTemporaryFile

TESTDATA_DIR = data_path("gibuu")

try:
    import aa, ROOT
    AANET_AVAILABLE = True
except ImportError:
    AANET_AVAILABLE = False


class TestXSection(unittest.TestCase):
    def test_xsection_all(self):
        filename = join(TESTDATA_DIR, XSECTION_FILENAMES["all"])
        xsection = read_nu_abs_xsection(filename)
        self.assertAlmostEqual(xsection['var'], 58.631)
        self.assertAlmostEqual(xsection['sum'], 8.0929)
        self.assertAlmostEqual(xsection['Delta'], 0.26805)
        self.assertAlmostEqual(xsection['highRES'], 0.14248)


class TestGiBUUOutput(unittest.TestCase):
    def setUp(self):
        self.output = GiBUUOutput(TESTDATA_DIR)

    def test_attr(self):
        assert hasattr(self.output, "df")

    def test_mean_xsec(self):
        df = self.output.df
        df = df.groupby(level=0).head(1)
        df = df[(df.lepIn_E > 0.7) & (df.lepIn_E < 1.0)]
        xsec = np.sum(df.xsec / df.lepIn_E)
        n_evts = self.output.flux_interpolation.integral(0.7, 1.0) / 0.02
        self.assertAlmostEqual(xsec / n_evts, 0.8, places=2)


@pytest.mark.skipif(not AANET_AVAILABLE, reason="aanet required")
class TestAANET(unittest.TestCase):
    def setUp(self):
        output = GiBUUOutput(TESTDATA_DIR)
        datafile = NamedTemporaryFile(suffix=".root")
        np.random.seed(1234)
        write_detector_file(output, datafile.name)
        self.fobj = km3io.OfflineReader(datafile.name)

    def test_numbering(self):
        evts = self.fobj.events
        np.testing.assert_array_equal(evts.id, range(4005))

    def test_firstevent(self):
        evt = self.fobj.events[0]
        np.testing.assert_array_equal(evt.mc_tracks.type,
                                      [12, 2212, 111, 211, -211])
        np.testing.assert_array_almost_equal(
            evt.mc_tracks.E,
            [11.90433897, 1.45689677, 0.49284856, 8.33975778, 0.28362369])
        np.testing.assert_array_almost_equal(
            evt.mc_tracks.dir_x,
            [0.18255849, 0.48623089, 0.23767571, 0.24971059, 0.11284916])
        np.testing.assert_array_almost_equal(
            evt.mc_tracks.dir_y,
            [-0.80816248, -0.49241334, -0.84679953, -0.83055629, -0.82624071])
        np.testing.assert_array_almost_equal(
            evt.mc_tracks.dir_z,
            [0.55995162, 0.72187854, 0.47585798, 0.4978161, -0.55189796])
        # Test dataset is elec CC -> outgoing particles are placed at vertex pos
        np.testing.assert_allclose(evt.mc_tracks.t, 8603022.62272017)
        np.testing.assert_allclose(evt.mc_tracks.pos_x, -127.07940486)
        np.testing.assert_allclose(evt.mc_tracks.pos_y, -122.54421157)
        np.testing.assert_allclose(evt.mc_tracks.pos_z, 208.57726764)
        usr = evt.mc_tracks.usr[0]
        # Bx
        np.testing.assert_almost_equal(usr[0], 0.35479262672400624)
        # By
        np.testing.assert_almost_equal(usr[1], 0.8167222969153614)
        # iChannel
        np.testing.assert_equal(usr[2], 2)
        # CC/NC
        np.testing.assert_equal(usr[3], 1)
