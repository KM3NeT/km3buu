# Filename: output.py
"""
IO for km3buu

"""

__author__ = "Johannes Schumann"
__copyright__ = "Copyright 2020, Johannes Schumann and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Johannes Schumann"
__email__ = "jschumann@km3net.de"
__status__ = "Development"

import re
import numpy as np
from io import StringIO
from os import listdir
from os.path import isfile, join, abspath
from tempfile import TemporaryDirectory
import awkward1
import uproot4
from scipy.interpolate import UnivariateSpline
from scipy.spatial.transform import Rotation

from .jobcard import Jobcard, read_jobcard, PDGID_LOOKUP

EVENT_FILENAME = "FinalEvents.dat"
ROOT_PERT_FILENAME = "EventOutput.Pert.[0-9]{8}.root"
ROOT_REAL_FILENAME = "EventOutput.Real.[0-9]{8}.root"
FLUXDESCR_FILENAME = "neutrino_initialized_energyFlux.dat"
XSECTION_FILENAMES = {"all": "neutrino_absorption_cross_section_ALL.dat"}

PARTICLE_COLUMNS = ["E", "Px", "Py", "Pz", "barcode"]
EVENTINFO_COLUMNS = [
    "weight", "evType", "lepIn_E", "lepIn_Px", "lepIn_Py", "lepIn_Pz",
    "lepOut_E", "lepOut_Px", "lepOut_Py", "lepOut_Pz", "nuc_E", "nuc_Px",
    "nuc_Py", "nuc_Pz"
]

LHE_NU_INFO_DTYPE = np.dtype([
    ("type", np.int),
    ("weight", np.float64),
    ("mom_lepton_in_E", np.float64),
    ("mom_lepton_in_x", np.float64),
    ("mom_lepton_in_y", np.float64),
    ("mom_lepton_in_z", np.float64),
    ("mom_lepton_out_E", np.float64),
    ("mom_lepton_out_x", np.float64),
    ("mom_lepton_out_y", np.float64),
    ("mom_lepton_out_z", np.float64),
    ("mom_nucleon_in_E", np.float64),
    ("mom_nucleon_in_x", np.float64),
    ("mom_nucleon_in_y", np.float64),
    ("mom_nucleon_in_z", np.float64),
])

FLUX_INFORMATION_DTYPE = np.dtype([("energy", np.float64),
                                   ("flux", np.float64),
                                   ("events", np.float64)])

EVENT_TYPE = {
    1: "QE",
    32: "pi neutron-background",
    33: "pi proton-background",
    34: "DIS",
    35: "2p2h QE",
    36: "2p2h Delta",
    37: "2pi background",
}

ROOTTUPLE_KEY = "RootTuple"


def read_nu_abs_xsection(filepath):
    """
    Read the crosssections calculated by GiBUU

    Parameters
    ----------
    filepath: str
        Filepath to the GiBUU output file with neutrino absorption cross-section
        (neutrino_absorption_cross_section_*.dat)

    """
    with open(filepath, "r") as f:
        lines = f.readlines()
    header = re.sub(r"\d+:|#", "", lines[0]).split()
    dt = np.dtype([(field, np.float64) for field in header])
    values = np.genfromtxt(StringIO(lines[-1]), dtype=dt)
    return values


class GiBUUOutput:
    def __init__(self, data_dir):
        """
        Class for parsing GiBUU output files

        Parameters
        ----------
        data_dir: str
        """
        if isinstance(data_dir, TemporaryDirectory):
            self._tmp_dir = data_dir
            self._data_path = abspath(data_dir.name)
        else:
            self._data_path = abspath(data_dir)
        self.output_files = [
            f for f in listdir(self._data_path)
            if isfile(join(self._data_path, f))
        ]
        self._read_xsection_file()
        self._read_root_output()
        self._read_flux_file()
        self._read_jobcard()

    def _read_root_output(self):
        root_pert_regex = re.compile(ROOT_PERT_FILENAME)
        self.root_pert_files = list(
            filter(root_pert_regex.match, self.output_files))

        root_real_regex = re.compile(ROOT_REAL_FILENAME)
        self.root_real_files = list(
            filter(root_real_regex.match, self.output_files))

    def _read_xsection_file(self):
        if XSECTION_FILENAMES["all"] in self.output_files:
            setattr(
                self,
                "xsection",
                read_nu_abs_xsection(
                    join(self._data_path, XSECTION_FILENAMES["all"])),
            )

    def _read_jobcard(self):
        jobcard_regex = re.compile(".*.job")
        jobcard_files = list(filter(jobcard_regex.match, self.output_files))
        if len(jobcard_files) == 1:
            self._jobcard_fname = jobcard_files[0]
            self.jobcard = read_jobcard(
                join(self._data_path, self._jobcard_fname))
        else:
            self.jobcard = None

    def _read_flux_file(self):
        fpath = join(self._data_path, FLUXDESCR_FILENAME)
        self.flux_data = np.loadtxt(fpath, dtype=FLUX_INFORMATION_DTYPE)
        self.flux_interpolation = UnivariateSpline(self.flux_data["energy"],
                                                   self.flux_data["events"])

    def _event_xsec(self, root_tupledata):
        weights = np.array(root_tupledata.weight)
        deltaE = np.mean(self.flux_data['energy'][1:] -
                         self.flux_data['energy'][:-1])
        energy_min = np.min(self.flux_data["energy"])
        energy_max = np.max(self.flux_data["energy"])
        total_flux_events = self.flux_interpolation.integral(
            energy_min, energy_max)
        n_files = len(self.root_pert_files)
        xsec = np.divide(total_flux_events * weights, deltaE * n_files)
        return xsec

    @staticmethod
    def bjorken_x(roottuple_data):
        d = roottuple_data
        k_in = np.vstack([
            np.array(d.lepIn_E),
            np.array(d.lepIn_Px),
            np.array(d.lepIn_Py),
            np.array(d.lepIn_Pz)
        ])
        k_out = np.vstack([
            np.array(d.lepOut_E),
            np.array(d.lepOut_Px),
            np.array(d.lepOut_Py),
            np.array(d.lepOut_Pz)
        ])
        q = k_in - k_out
        qtmp = np.power(q, 2)
        q2 = qtmp[0, :] - np.sum(qtmp[1:4, :], axis=0)
        pq = q[0, :] * d.nuc_E - q[1, :] * d.nuc_Px - q[2, :] * d.nuc_Py - q[
            3, :] * d.nuc_Pz
        x = np.divide(-q2, 2 * pq)
        return np.array(x)

    @staticmethod
    def bjorken_y(roottuple_data):
        d = roottuple_data
        y = 1 - np.divide(np.array(d.lepOut_E), np.array(d.lepIn_E))
        return y

    @property
    def A(self):
        return self.jobcard["target"]["target_a"]

    @property
    def Z(self):
        return self.jobcard["target"]["target_z"]

    @property
    def data_path(self):
        return self._data_path

    @property
    def df(self):
        import pandas as pd
        df = None
        for fname in self.root_pert_files:
            with uproot4.open(join(self._data_path, fname)) as fobj:
                event_data = fobj[ROOTTUPLE_KEY].arrays()
            tmp_df = awkward1.to_pandas(event_data)
            idx_mask = tmp_df.groupby(level=[0]).ngroup()
            tmp_df["xsec"] = self._event_xsec(event_data)[idx_mask]
            tmp_df["Bx"] = GiBUUOutput.bjorken_x(event_data)[idx_mask]
            tmp_df["By"] = GiBUUOutput.bjorken_y(event_data)[idx_mask]
            if df is None:
                df = tmp_df
            else:
                new_indices = tmp_df.index.levels[0] + df.index.levels[0].max(
                ) + 1
                tmp_df.index = tmp_df.index.set_levels(new_indices, level=0)
                df = df.append(tmp_df)
        df.columns = [col[0] for col in df.columns]
        sec_df = df[df.index.get_level_values(1) == 0]
        sec_df.loc[:, "E"] = sec_df.lepOut_E
        sec_df.loc[:, "Px"] = sec_df.lepOut_Px
        sec_df.loc[:, "Py"] = sec_df.lepOut_Py
        sec_df.loc[:, "Pz"] = sec_df.lepOut_Pz
        sec_pdgid = (
            PDGID_LOOKUP[self.jobcard["neutrino_induced"]["flavor_id"]] -
            1) * np.sign(self.jobcard["neutrino_induced"]["process_id"])
        sec_df.loc[:, "barcode"] = sec_pdgid
        sec_df.index = pd.MultiIndex.from_tuples(
            zip(*np.unique(df.index.get_level_values(0), return_counts=True)))
        df = df.append(sec_df)
        return df


def write_detector_file(gibuu_output,
                        ofile="gibuu.aanet.root",
                        can=(0, 476.5, 403.4),
                        livetime=3.156e7):
    """
    Convert the GiBUU output to a KM3NeT MC (AANET) file

    Parameters
    ----------
    gibuu_output: GiBUUOutput
        Output object which wraps the information from the GiBUU output files
    ofile: str
        Output filename
    can: tuple
        The can dimensions which are used to distribute the events
    livetime: float
        The data livetime
    """
    import aa, ROOT

    aafile = ROOT.EventFile()
    aafile.set_output(ofile)
    mc_event_id = 0

    is_cc = False

    if gibuu_output.jobcard is None:
        raise EnvironmentError("No jobcard provided within the GiBUU output!")

    nu_type = PDGID_LOOKUP[gibuu_output.jobcard["neutrino_induced"]
                           ["flavor_id"]]
    sec_lep_type = nu_type
    ichan = abs(gibuu_output.jobcard["neutrino_induced"]["process_id"])
    if ichan == 2:
        is_cc = True
        sec_lep_type -= 1
    if gibuu_output.jobcard["neutrino_induced"]["process_id"] < 0:
        nu_type *= -1
        sec_lep_type *= -1

    for ifile in gibuu_output.root_pert_files:
        fobj = uproot4.open(join(gibuu_output.data_path, ifile))
        event_data = fobj["RootTuple"].arrays()
        bjorkenx = GiBUUOutput.bjorken_x(event_data)
        bjorkeny = GiBUUOutput.bjorken_y(event_data)
        for i, event in enumerate(event_data):
            aafile.evt.clear()
            aafile.evt.id = mc_event_id
            aafile.evt.mc_run_id = mc_event_id
            mc_event_id += 1
            # Vertex Position
            r = can[2] * np.sqrt(np.random.uniform(0, 1))
            phi = np.random.uniform(0, 2 * np.pi)
            pos_x = r * np.cos(phi)
            pos_y = r * np.sin(phi)
            pos_z = np.random.uniform(can[0], can[1])
            vtx_pos = np.array([pos_x, pos_y, pos_z])
            # Direction
            phi = np.random.uniform(0, 2 * np.pi)
            cos_theta = np.random.uniform(-1, 1)
            sin_theta = np.sqrt(1 - cos_theta**2)

            dir_x = np.cos(phi) * sin_theta
            dir_y = np.sin(phi) * sin_theta
            dir_z = cos_theta

            direction = np.array([dir_x, dir_y, dir_z])
            theta = np.arccos(cos_theta)
            R = Rotation.from_euler("yz", [theta, phi])

            timestamp = np.random.uniform(0, livetime)

            nu_in_trk = ROOT.Trk()
            nu_in_trk.id = 0
            nu_in_trk.mother_id = -1
            nu_in_trk.type = nu_type
            nu_in_trk.pos.set(*vtx_pos)
            nu_in_trk.dir.set(*direction)
            nu_in_trk.E = event.lepIn_E
            nu_in_trk.t = timestamp

            lep_out_trk = ROOT.Trk()
            lep_out_trk.id = 1
            lep_out_trk.mother_id = 0
            lep_out_trk.type = sec_lep_type
            lep_out_trk.pos.set(*vtx_pos)
            mom = np.array([event.lepOut_Px, event.lepOut_Py, event.lepOut_Pz])
            p_dir = R.apply(mom / np.linalg.norm(mom))
            lep_out_trk.dir.set(*p_dir)
            lep_out_trk.E = event.lepOut_E
            lep_out_trk.t = timestamp

            # bjorken_y = 1.0 - float(event.lepOut_E / event.lepIn_E)
            nu_in_trk.setusr('bx', bjorkenx[i])
            nu_in_trk.setusr('by', bjorkeny[i])
            nu_in_trk.setusr('ichan', ichan)
            nu_in_trk.setusr("cc", is_cc)

            aafile.evt.mc_trks.push_back(nu_in_trk)
            aafile.evt.mc_trks.push_back(lep_out_trk)

            for i in range(len(event.E)):
                trk = ROOT.Trk()
                trk.id = i + 2
                mom = np.array([event.Px[i], event.Py[i], event.Pz[i]])
                p_dir = R.apply(mom / np.linalg.norm(mom))
                trk.pos.set(*vtx_pos)
                trk.dir.set(*p_dir)
                trk.mother_id = 0
                trk.type = int(event.barcode[i])
                trk.E = event.E[i]
                trk.t = timestamp
                aafile.evt.mc_trks.push_back(trk)
            aafile.write()

    del aafile
