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
from km3buu.geometry import NoVolume, CANVolume
from km3buu.config import Config

TESTDATA_DIR = data_path("gibuu")

try:
    import ROOT
    libpath = environ.get("KM3NET_LIB")
    if libpath is None:
        libpath = Config().km3net_lib_path
    KM3NET_LIB_AVAILABLE = libpath is not None and (ROOT.gSystem.Load(
        join(libpath, "libKM3NeTROOT.so")) >= 0)
except (ImportError, ModuleNotFoundError):
    KM3NET_LIB_AVAILABLE = False


class TestXSection(unittest.TestCase):

    def test_xsection_all(self):
        filename = join(TESTDATA_DIR, XSECTION_FILENAMES["all"])
        xsection = read_nu_abs_xsection(filename)
        self.assertAlmostEqual(xsection['var'], 12.435)
        self.assertAlmostEqual(xsection['sum'], 16.424)
        self.assertAlmostEqual(xsection['Delta'], 0.3777)
        self.assertAlmostEqual(xsection['highRES'], 0.43642)


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
        self.assertAlmostEqual(xsec / n_evts, 0.6552, places=2)

    def test_nucleus_properties(self):
        assert self.output.Z == 8
        assert self.output.A == 16

    def test_flux_index(self):
        assert np.isclose(self.output.flux_index, -0.00576924, rtol=1e-3)

    def test_flux_norm(self):
        assert np.isclose(self.output.flux_norm, 4.08791, rtol=1e-3)

    def test_fixed_flux_index(self):
        output = GiBUUOutput(TESTDATA_DIR)
        output.flux_index = -1.0
        assert np.isclose(output.flux_index, -1.0, rtol=1e-3)

    def test_w2weights(self):
        w2 = self.output.w2weights(123.0, 2.6e28, 4 * np.pi)
        np.testing.assert_array_almost_equal(
            w2[:3], [2.73669e-07, 4.04777e-09, 7.26780e-11], decimal=5)

    def test_global_generation_weight(self):
        self.assertAlmostEqual(self.output.global_generation_weight(4 * np.pi),
                               2513.2765713094464,
                               places=2)

    def test_event_values(self):
        arr = self.output.arrays
        np.testing.assert_array_almost_equal(
            arr["M2"][0], [0.019044, 0.832647, 0.019044, 0.019044, 0.019044])

    def test_free_particle_mask(self):
        mask = self.output.free_particle_mask
        np.testing.assert_array_equal(mask[0], [True, False, True, True, True])

    def test_right_handed_helicity(self):
        arr = self.output.arrays
        np.testing.assert_array_almost_equal(
            arr.rh_prob[:3], [4.518065e-05, 9.428467e-06, 3.488926e-05])

    def test_right_handed_helicity_nc(self):
        arr = self.output._get_raw_arrays()
        arr["process_ID"] = arr.process_ID + 1
        rh_prob = GiBUUOutput.right_helicity_probability(arr)
        np.testing.assert_array_almost_equal(rh_prob[:3], [0.0, 0.0, 0.0])


@pytest.mark.skipif(not KM3NET_LIB_AVAILABLE,
                    reason="KM3NeT dataformat required")
class TestOfflineFile(unittest.TestCase):

    def setUp(self):
        output = GiBUUOutput(TESTDATA_DIR)
        datafile = NamedTemporaryFile(suffix=".root")
        np.random.seed(1234)
        geometry = CANVolume()
        write_detector_file(output,
                            datafile.name,
                            run_number=1234,
                            geometry=geometry,
                            free_particle_cuts=False)
        self.fobj = km3io.OfflineReader(datafile.name)

    def test_header_event_numbers(self):
        np.testing.assert_equal(self.fobj.header.genvol.numberOfEvents, 10000)
        np.testing.assert_equal(self.fobj.header.gibuu_Nevents, 10000)
        np.testing.assert_equal(self.fobj.header.start_run.run_id, 1234)

    def test_numbering(self):
        evts = self.fobj.events
        np.testing.assert_array_equal(evts.id, range(4755))

    def test_firstevent(self):
        evt = self.fobj.events[0]
        np.testing.assert_array_equal(
            evt.mc_tracks.pdgid,
            [12, 1000080160, 11, 111, 2212, -211, 111, 211])
        np.testing.assert_array_equal(evt.mc_tracks.status,
                                      [100, 5, 1, 1, 1, 1, 1, 1])
        np.testing.assert_array_almost_equal(evt.mc_tracks.E, [
            24.777316, 14.899169, 5.654808, 1.808523, 11.802908, 2.866352,
            0.222472, 3.124241
        ])
        np.testing.assert_array_almost_equal(evt.mc_tracks.dir_x, [
            0.93038, 0.0, 0.761401, 0.818513, 0.964518, 0.984183, 0.39,
            0.925828
        ])
        np.testing.assert_array_almost_equal(evt.mc_tracks.dir_y, [
            -0.124785, 0.0, 0.088805, -0.326036, -0.15996, -0.104239,
            -0.868099, -0.200785
        ])
        np.testing.assert_array_almost_equal(evt.mc_tracks.dir_z, [
            -0.344705, 0.0, -0.64217, -0.473008, -0.210043, -0.143242,
            -0.307089, -0.320199
        ])
        # Test dataset is elec CC -> outgoing particles are placed at vertex pos
        np.testing.assert_allclose(evt.mc_tracks.t, 0.0)
        np.testing.assert_allclose(evt.mc_tracks.pos_x, -105.056045)
        np.testing.assert_allclose(evt.mc_tracks.pos_y, -359.273506)
        np.testing.assert_allclose(evt.mc_tracks.pos_z, 326.074154)
        usr = evt.mc_tracks.usr[0]
        # XSEC
        np.testing.assert_almost_equal(evt.w2list[13], 8.055736278936948)
        # Bx
        np.testing.assert_almost_equal(evt.w2list[7], 0.6810899274375058)
        # By
        np.testing.assert_almost_equal(evt.w2list[8], 0.7719161976356189)
        # iChannel
        np.testing.assert_equal(evt.w2list[9], 3)
        # CC/NC
        np.testing.assert_equal(evt.w2list[10], 2)
        # GiBUU weight
        np.testing.assert_almost_equal(evt.w2list[23], 0.0008055736278936948)


@pytest.mark.skipif(not KM3NET_LIB_AVAILABLE,
                    reason="KM3NeT dataformat required")
class TestFreeParticleCuts(unittest.TestCase):

    def setUp(self):
        output = GiBUUOutput(TESTDATA_DIR)
        datafile = NamedTemporaryFile(suffix=".root")
        np.random.seed(1234)
        geometry = CANVolume()
        write_detector_file(output,
                            datafile.name,
                            run_number=1235,
                            free_particle_cuts=True,
                            geometry=geometry,
                            timeinterval=(10.0, 100.0))
        self.fobj = km3io.OfflineReader(datafile.name)

    def test_header_event_numbers(self):
        np.testing.assert_equal(self.fobj.header.genvol.numberOfEvents, 10000)
        np.testing.assert_equal(self.fobj.header.gibuu_Nevents, 10000)
        np.testing.assert_equal(self.fobj.header.start_run.run_id, 1235)
        np.testing.assert_equal(self.fobj.header.tgen, 90.0)

    def test_numbering(self):
        evts = self.fobj.events
        np.testing.assert_array_equal(evts.id, range(4755))

    def test_timestamp(self):
        assert self.fobj.mc_event_time_sec[0] == 10
        assert self.fobj.mc_event_time_ns[0] == 2331931

    def test_firstevent(self):
        evt = self.fobj.events[0]
        np.testing.assert_array_equal(
            evt.mc_tracks.pdgid, [12, 1000080160, 11, 111, -211, 111, 211])
        np.testing.assert_array_equal(evt.mc_tracks.status,
                                      [100, 5, 1, 1, 1, 1, 1])
        np.testing.assert_array_almost_equal(evt.mc_tracks.E, [
            24.777316, 14.899169, 5.654808, 1.808523, 2.866352, 0.222472,
            3.124241
        ])
        np.testing.assert_array_almost_equal(
            evt.mc_tracks.dir_x,
            [0.93038, 0.0, 0.761401, 0.818513, 0.984183, 0.39, 0.925828])
        np.testing.assert_array_almost_equal(evt.mc_tracks.dir_y, [
            -0.124785, 0.0, 0.088805, -0.326036, -0.104239, -0.868099,
            -0.200785
        ])
        np.testing.assert_array_almost_equal(evt.mc_tracks.dir_z, [
            -0.344705, 0.0, -0.64217, -0.473008, -0.143242, -0.307089,
            -0.320199
        ])
        # Test dataset is elec CC -> outgoing particles are placed at vertex pos
        np.testing.assert_allclose(evt.mc_tracks.t, 0.0)
        np.testing.assert_allclose(evt.mc_tracks.pos_x, -105.056045)
        np.testing.assert_allclose(evt.mc_tracks.pos_y, -359.273506)
        np.testing.assert_allclose(evt.mc_tracks.pos_z, 326.074154)
        usr = evt.mc_tracks.usr[0]
        # XSEC
        np.testing.assert_almost_equal(evt.w2list[13], 8.055736278936948)
        # Bx
        np.testing.assert_almost_equal(evt.w2list[7], 0.6810899274375058)
        # By
        np.testing.assert_almost_equal(evt.w2list[8], 0.7719161976356189)
        # iChannel
        np.testing.assert_equal(evt.w2list[9], 3)
        # CC/NC
        np.testing.assert_equal(evt.w2list[10], 2)
        # GiBUU weight
        np.testing.assert_almost_equal(evt.w2list[23], 0.0008055736278936948)


@pytest.mark.skipif(not KM3NET_LIB_AVAILABLE,
                    reason="KM3NeT dataformat required")
class TestNoGeometryWriteout(unittest.TestCase):

    def setUp(self):
        output = GiBUUOutput(TESTDATA_DIR)
        datafile = NamedTemporaryFile(suffix=".root")
        geometry = NoVolume()
        np.random.seed(1234)
        write_detector_file(output,
                            datafile.name,
                            geometry=geometry,
                            free_particle_cuts=False)
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
        geometry = CANVolume()
        write_detector_file(output,
                            datafile.name,
                            no_files=2,
                            geometry=geometry,
                            free_particle_cuts=False)
        self.fobj1 = km3io.OfflineReader(
            datafile.name.replace(".root", ".1.root"))
        self.fobj2 = km3io.OfflineReader(
            datafile.name.replace(".root", ".2.root"))

    def test_header_event_numbers(self):
        np.testing.assert_equal(self.fobj1.header.genvol.numberOfEvents, 10000)
        np.testing.assert_equal(self.fobj2.header.genvol.numberOfEvents, 10000)
        np.testing.assert_equal(self.fobj1.header.gibuu_Nevents, 10000)
        np.testing.assert_equal(self.fobj2.header.gibuu_Nevents, 10000)
        np.testing.assert_equal(self.fobj1.header.n_split_files, 2)
        np.testing.assert_equal(self.fobj2.header.n_split_files, 2)

    def test_numbering(self):
        np.testing.assert_array_equal(self.fobj1.events.id, range(2377))
        np.testing.assert_array_equal(self.fobj2.events.id, range(2378))

    def test_firstevent_first_file(self):
        evt = self.fobj1.events[0]
        np.testing.assert_array_equal(
            evt.mc_tracks.pdgid,
            [12, 1000080160, 11, 111, 2212, -211, 111, 211])
        np.testing.assert_array_equal(evt.mc_tracks.status,
                                      [100, 5, 1, 1, 1, 1, 1, 1])
        np.testing.assert_array_almost_equal(evt.mc_tracks.E, [
            24.777316, 14.899169, 5.654808, 1.808523, 11.802908, 2.866352,
            0.222472, 3.124241
        ])
        np.testing.assert_array_almost_equal(evt.mc_tracks.dir_x, [
            0.93038, 0.0, 0.761401, 0.818513, 0.964518, 0.984183, 0.39,
            0.925828
        ])
        np.testing.assert_array_almost_equal(evt.mc_tracks.dir_y, [
            -0.124785, 0.0, 0.088805, -0.326036, -0.15996, -0.104239,
            -0.868099, -0.200785
        ])
        np.testing.assert_array_almost_equal(evt.mc_tracks.dir_z, [
            -0.344705, 0.0, -0.64217, -0.473008, -0.210043, -0.143242,
            -0.307089, -0.320199
        ])
        # Test dataset is elec CC -> outgoing particles are placed at vertex pos
        np.testing.assert_allclose(evt.mc_tracks.t, 0.0)
        np.testing.assert_allclose(evt.mc_tracks.pos_x, -105.056045)
        np.testing.assert_allclose(evt.mc_tracks.pos_y, -359.273506)
        np.testing.assert_allclose(evt.mc_tracks.pos_z, 326.074154)
        usr = evt.mc_tracks.usr[0]
        # XSEC
        np.testing.assert_almost_equal(evt.w2list[13], 8.055736278936948)
        # Bx
        np.testing.assert_almost_equal(evt.w2list[7], 0.6810899274375058)
        # By
        np.testing.assert_almost_equal(evt.w2list[8], 0.7719161976356189)
        # iChannel
        np.testing.assert_equal(evt.w2list[9], 3)
        # CC/NC
        np.testing.assert_equal(evt.w2list[10], 2)
        # GiBUU weight
        np.testing.assert_almost_equal(evt.w2list[23], 0.0008055736278936948)

    def test_firstevent_second_file(self):
        evt = self.fobj2.events[0]
        np.testing.assert_array_equal(evt.mc_tracks.pdgid,
                                      [12, 1000080160, 11, 2212])
        np.testing.assert_array_equal(evt.mc_tracks.status, [100, 5, 1, 1])
        np.testing.assert_array_almost_equal(
            evt.mc_tracks.E, [5.010154, 14.899169, 1.041807, 4.87277])
        np.testing.assert_array_almost_equal(
            evt.mc_tracks.dir_x, [0.659808, 0.0, -0.546414, 0.79964])
        np.testing.assert_array_almost_equal(
            evt.mc_tracks.dir_y, [-0.705338, 0.0, -0.570482, -0.587667])
        np.testing.assert_array_almost_equal(
            evt.mc_tracks.dir_z, [-0.259137, 0.0, -0.613174, -0.123386])
        # Test dataset is elec CC -> outgoing particles are placed at vertex pos
        np.testing.assert_allclose(evt.mc_tracks.t, 0.0)
        np.testing.assert_allclose(evt.mc_tracks.pos_x, -282.369594)
        np.testing.assert_allclose(evt.mc_tracks.pos_y, 26.85389)
        np.testing.assert_allclose(evt.mc_tracks.pos_z, 36.872621)
        usr = evt.mc_tracks.usr[0]
        # XSEC
        np.testing.assert_almost_equal(evt.w2list[13], 0.0021272535302635045)
        # Bx
        np.testing.assert_almost_equal(evt.w2list[7], 0.6810899274375058)
        # By
        np.testing.assert_almost_equal(evt.w2list[8], 0.7719161976356189)
        # iChannel
        np.testing.assert_equal(evt.w2list[9], 1)
        # CC/NC
        np.testing.assert_equal(evt.w2list[10], 2)
        # GiBUU weight
        np.testing.assert_almost_equal(evt.w2list[23], 2.1272535302635046e-07)
