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
from unittest.mock import patch
import numpy as np
import pytest
import km3io
from os import listdir, environ
from os.path import abspath, join, dirname
from km3net_testdata import data_path
from tempfile import NamedTemporaryFile, TemporaryDirectory

from km3buu.output import *
from km3buu.geometry import NoVolume
from km3buu.config import Config

TESTDATA_DIR = data_path("gibuu")

try:
    import ROOT
    libpath = environ.get("KM3NET_LIB")
    if libpath is None:
        libpath = Config().km3net_lib_path
    KM3NET_LIB_AVAILABLE = (ROOT.gSystem.Load(join(libpath,
                                                   "libKM3NeTROOT.so")) >= 0)
except ModuleNotFoundError:
    KM3NET_LIB_AVAILABLE = False


class TestXSection(unittest.TestCase):

    def test_xsection_all(self):
        filename = join(TESTDATA_DIR, XSECTION_FILENAMES["all"])
        xsection = read_nu_abs_xsection(filename)
        self.assertAlmostEqual(xsection['var'], 58.631)
        self.assertAlmostEqual(xsection['sum'], 8.0929)
        self.assertAlmostEqual(xsection['Delta'], 0.26805)
        self.assertAlmostEqual(xsection['highRES'], 0.14248)


class TestGiBUUOutput(unittest.TestCase):

    def setup_class(self):
        self.output = GiBUUOutput(TESTDATA_DIR)

    def test_tmp_dir_init(self):
        with patch('tempfile.TemporaryDirectory',
                   spec=TemporaryDirectory) as mock:
            instance = mock.return_value
            instance.name = abspath(TESTDATA_DIR)
            output = GiBUUOutput(instance)
            assert output.data_path == abspath(TESTDATA_DIR)

    def test_attr(self):
        assert hasattr(self.output, "df")

    def test_mean_xsec(self):
        df = self.output.df
        df = df.groupby(level=0).head(1)
        df = df[(df.lepIn_E > 0.7) & (df.lepIn_E < 1.0)]
        xsec = np.sum(df.xsec / df.lepIn_E)
        n_evts = self.output.flux_interpolation.integral(0.7, 1.0) / 0.02
        self.assertAlmostEqual(xsec / n_evts, 0.8, places=2)

    def test_nucleus_properties(self):
        assert self.output.Z == 8
        assert self.output.A == 16

    def test_flux_index(self):
        assert np.isclose(self.output.flux_index, -0.904, rtol=1e-3)

    def test_w2weights(self):
        w2 = self.output.w2weights(123.0, 2.6e28, 4 * np.pi)
        np.testing.assert_array_almost_equal(
            w2[:3], [2.42100575e-06, 1.14490671e-08, 3.59246902e-05],
            decimal=5)

    def test_global_generation_weight(self):
        self.assertAlmostEqual(self.output.global_generation_weight(4 * np.pi),
                               2511.13458,
                               places=2)


@pytest.mark.skipif(not KM3NET_LIB_AVAILABLE,
                    reason="KM3NeT dataformat required")
class TestOfflineFile(unittest.TestCase):

    def setUp(self):
        output = GiBUUOutput(TESTDATA_DIR)
        datafile = NamedTemporaryFile(suffix=".root")
        np.random.seed(1234)
        write_detector_file(output, datafile.name)
        self.fobj = km3io.OfflineReader(datafile.name)

    def test_header_event_numbers(self):
        np.testing.assert_equal(self.fobj.header.genvol.numberOfEvents, 4005)
        np.testing.assert_equal(self.fobj.header.gibuu_Nevents, 10000)

    def test_numbering(self):
        evts = self.fobj.events
        np.testing.assert_array_equal(evts.id, range(4005))

    def test_firstevent(self):
        evt = self.fobj.events[0]
        np.testing.assert_array_equal(evt.mc_tracks.pdgid,
                                      [12, 11, 2212, 111, 211, -211])
        np.testing.assert_array_equal(evt.mc_tracks.status,
                                      [100, 1, 1, 1, 1, 1])
        np.testing.assert_array_almost_equal(evt.mc_tracks.E, [
            11.90433897, 2.1818, 1.45689677, 0.49284856, 8.33975778, 0.28362369
        ])
        np.testing.assert_array_almost_equal(evt.mc_tracks.dir_x, [
            0.18255849, -0.2469, 0.48623089, 0.23767571, 0.24971059, 0.11284916
        ])
        np.testing.assert_array_almost_equal(evt.mc_tracks.dir_y, [
            -0.80816248, -0.619212, -0.49241334, -0.84679953, -0.83055629,
            -0.82624071
        ])
        np.testing.assert_array_almost_equal(evt.mc_tracks.dir_z, [
            0.55995162, 0.745398, 0.72187854, 0.47585798, 0.4978161,
            -0.55189796
        ])
        # Test dataset is elec CC -> outgoing particles are placed at vertex pos
        np.testing.assert_allclose(evt.mc_tracks.t, 8603022.62272017)
        np.testing.assert_allclose(evt.mc_tracks.pos_x, -127.07940486)
        np.testing.assert_allclose(evt.mc_tracks.pos_y, -122.54421157)
        np.testing.assert_allclose(evt.mc_tracks.pos_z, 208.57726764)
        usr = evt.mc_tracks.usr[0]
        # XSEC
        np.testing.assert_almost_equal(evt.w2list[13], 40.62418521597373)
        # Bx
        np.testing.assert_almost_equal(evt.w2list[7], 0.35479262672400624)
        # By
        np.testing.assert_almost_equal(evt.w2list[8], 0.8203215908456797)
        # iChannel
        np.testing.assert_equal(evt.w2list[9], 3)
        # CC/NC
        np.testing.assert_equal(evt.w2list[10], 2)
        # GiBUU weight
        np.testing.assert_almost_equal(evt.w2list[23], 0.004062418521597373)


@pytest.mark.skipif(not KM3NET_LIB_AVAILABLE,
                    reason="KM3NeT dataformat required")
class TestNoGeometryWriteout(unittest.TestCase):

    def setUp(self):
        output = GiBUUOutput(TESTDATA_DIR)
        datafile = NamedTemporaryFile(suffix=".root")
        geometry = NoVolume()
        np.random.seed(1234)
        write_detector_file(output, datafile.name, geometry=geometry)
        self.fobj = km3io.OfflineReader(datafile.name)

    def test_firstevent(self):
        evt = self.fobj.events[0]
        np.testing.assert_array_almost_equal(evt.mc_tracks.dir_x[0], 0)
        np.testing.assert_array_almost_equal(evt.mc_tracks.dir_y[0], 0)
        np.testing.assert_array_almost_equal(evt.mc_tracks.dir_z[0], 1)
        np.testing.assert_allclose(evt.mc_tracks.pos_x, 0.0)
        np.testing.assert_allclose(evt.mc_tracks.pos_y, 0.0)
        np.testing.assert_allclose(evt.mc_tracks.pos_z, 0.0)


@pytest.mark.skipif(not KM3NET_LIB_AVAILABLE,
                    reason="KM3NeT dataformat required")
class TestMultiFileOutput(unittest.TestCase):

    def setUp(self):
        output = GiBUUOutput(TESTDATA_DIR)
        datafile = NamedTemporaryFile(suffix=".root")
        np.random.seed(1234)
        write_detector_file(output, datafile.name, no_files=2)
        self.fobj1 = km3io.OfflineReader(
            datafile.name.replace(".root", ".1.root"))
        self.fobj2 = km3io.OfflineReader(
            datafile.name.replace(".root", ".2.root"))

    def test_header_event_numbers(self):
        np.testing.assert_equal(self.fobj1.header.genvol.numberOfEvents, 2002)
        np.testing.assert_equal(self.fobj2.header.genvol.numberOfEvents, 2003)
        np.testing.assert_equal(self.fobj1.header.gibuu_Nevents, 10000)
        np.testing.assert_equal(self.fobj2.header.gibuu_Nevents, 10000)
        np.testing.assert_equal(self.fobj1.header.n_split_files, 2)
        np.testing.assert_equal(self.fobj2.header.n_split_files, 2)

    def test_numbering(self):
        np.testing.assert_array_equal(self.fobj1.events.id, range(2002))
        np.testing.assert_array_equal(self.fobj2.events.id, range(2003))

    def test_firstevent_first_file(self):
        evt = self.fobj1.events[0]
        np.testing.assert_array_equal(evt.mc_tracks.pdgid,
                                      [12, 11, 2212, 111, 211, -211])
        np.testing.assert_array_equal(evt.mc_tracks.status,
                                      [100, 1, 1, 1, 1, 1])
        np.testing.assert_array_almost_equal(evt.mc_tracks.E, [
            11.90433897, 2.1818, 1.45689677, 0.49284856, 8.33975778, 0.28362369
        ])
        np.testing.assert_array_almost_equal(evt.mc_tracks.dir_x, [
            0.18255849, -0.2469, 0.48623089, 0.23767571, 0.24971059, 0.11284916
        ])
        np.testing.assert_array_almost_equal(evt.mc_tracks.dir_y, [
            -0.80816248, -0.619212, -0.49241334, -0.84679953, -0.83055629,
            -0.82624071
        ])
        np.testing.assert_array_almost_equal(evt.mc_tracks.dir_z, [
            0.55995162, 0.745398, 0.72187854, 0.47585798, 0.4978161,
            -0.55189796
        ])
        # Test dataset is elec CC -> outgoing particles are placed at vertex pos
        np.testing.assert_allclose(evt.mc_tracks.t, 8603022.62272017)
        np.testing.assert_allclose(evt.mc_tracks.pos_x, -127.07940486)
        np.testing.assert_allclose(evt.mc_tracks.pos_y, -122.54421157)
        np.testing.assert_allclose(evt.mc_tracks.pos_z, 208.57726764)
        usr = evt.mc_tracks.usr[0]
        # XSEC
        np.testing.assert_almost_equal(evt.w2list[13], 40.62418521597373)
        # Bx
        np.testing.assert_almost_equal(evt.w2list[7], 0.35479262672400624)
        # By
        np.testing.assert_almost_equal(evt.w2list[8], 0.8203215908456797)
        # iChannel
        np.testing.assert_equal(evt.w2list[9], 3)
        # CC/NC
        np.testing.assert_equal(evt.w2list[10], 2)
        # GiBUU weight
        np.testing.assert_almost_equal(evt.w2list[23], 0.004062418521597373)

    def test_firstevent_second_file(self):
        evt = self.fobj2.events[0]
        np.testing.assert_array_equal(evt.mc_tracks.pdgid, [12, 11, 2212, 111])
        np.testing.assert_array_equal(evt.mc_tracks.status, [100, 1, 1, 1])
        np.testing.assert_array_almost_equal(
            evt.mc_tracks.E, [7.043544, 3.274632, 4.429621, 0.21289])
        np.testing.assert_array_almost_equal(
            evt.mc_tracks.dir_x, [0.997604, 0.824817, 0.941969, 0.00302])
        np.testing.assert_array_almost_equal(
            evt.mc_tracks.dir_y, [-0.058292, -0.553647, 0.327013, -0.097914])
        np.testing.assert_array_almost_equal(
            evt.mc_tracks.dir_z, [0.037271, 0.114683, -0.075871, 0.99519])
        # Test dataset is elec CC -> outgoing particles are placed at vertex pos
        np.testing.assert_allclose(evt.mc_tracks.t, 1951721.26185)
        np.testing.assert_allclose(evt.mc_tracks.pos_x, -171.8025)
        np.testing.assert_allclose(evt.mc_tracks.pos_y, -55.656482)
        np.testing.assert_allclose(evt.mc_tracks.pos_z, 363.950535)
        usr = evt.mc_tracks.usr[0]
        # XSEC
        np.testing.assert_almost_equal(evt.w2list[13], 4.218262109165907)
        # Bx
        np.testing.assert_almost_equal(evt.w2list[7], 0.35479262672400624)
        # By
        np.testing.assert_almost_equal(evt.w2list[8], 0.8203215908456797)
        # iChannel
        np.testing.assert_equal(evt.w2list[9], 3)
        # CC/NC
        np.testing.assert_equal(evt.w2list[10], 2)
        # GiBUU weight
        np.testing.assert_almost_equal(evt.w2list[23], 0.00042182621091659065)
