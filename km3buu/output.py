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
import mmap
import numpy as np
import pylhe
import pandas as pd
from io import StringIO
from os import listdir
from os.path import isfile, join, abspath
from tempfile import TemporaryDirectory
from collections import defaultdict
import awkward
import itertools
import uproot
from scipy.spatial.transform import Rotation

from .jobcard import Jobcard, read_jobcard, PDGID_LOOKUP

EVENT_FILENAME = "FinalEvents.dat"
LHE_PERT_FILENAME = "EventOutput.Pert.[0-9]{8}.lhe"
LHE_REAL_FILENAME = "EventOutput.Real.[0-9]{8}.lhe"
ROOT_PERT_FILENAME = "EventOutput.Pert.[0-9]{8}.root"
ROOT_REAL_FILENAME = "EventOutput.Real.[0-9]{8}.root"
FLUXDESCR_FILENAME = "neutrino_initialized_energyFlux.dat"
XSECTION_FILENAMES = {"all": "neutrino_absorption_cross_section_ALL.dat"}

PARTICLE_COLUMNS = ["E", "Px", "Py", "Pz", "barcode"]

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


def parse_gibuu_event_info(line):
    fields = line.split()[1:]
    if int(fields[0]) != 5:
        raise NotImplementedError(
            "Event information type %s cannot be parsed yet!" % fields[0])
    else:
        return np.genfromtxt(StringIO(line[3:]), dtype=LHE_NU_INFO_DTYPE)


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
            self.jobcard = read_jobcard(self._jobcard_fname)
        else:
            self.jobcard = None

    def _read_flux_file(self):
        fpath = join(self._data_path, FLUXDESCR_FILENAME)
        if isfile(fpath):
            self.flux_flat = False
            self.flux_data = np.loadtxt(fpath, dtype=FLUX_INFORMATION_DTYPE)
        else:
            self.flux_flat = True

    @property
    def particle_df(self):
        df = None
        for fname in self.root_pert_files:
            fobj = uproot.open(fname)
            file_df = None
            for col in PARTICLE_COLUMNS:
                tmp = awkward.topandas(fobj["RootTuple"][col].array(),
                                       flatten=True)
                if file_df is None:
                    file_df = tmp
                else:
                    file_df = pd.concat([file_df, tmp], axis=1)
            if df is None:
                df = file_df
            else:
                new_indices = file_df.index.levels[0] + df.index.levels[0].max(
                ) + 1
                file_df.index = file_df.index.set_levels(new_indices, level=0)
                df = df.append(file_df)
            fobj.close()

        df = df.rename(columns=dict(enumerate(PARTICLE_COLUMNS)))
        return df


def write_detector_file(gibuu_output,
                        ofile="gibuu.aanet.root",
                        can=(0, 476.5, 403.4),
                        livetime=3.156e7):
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
    if abs(gibuu_output.jobcard["neutrino_induced"]["process_id"]) == 2:
        is_cc = True
        sec_lep_type -= 1
    if gibuu_output.jobcard["neutrino_induced"]["process_id"] < 0:
        nu_type *= -1
        sec_lep_type *= -1

    for ifile in gibuu_output.lhe_pert_files:
        lhe_data = pylhe.readLHEWithAttributes(ifile)
        for event in lhe_data:
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
            rotation = np.array([dir_y, -dir_x, 0])
            sin_rot = np.linalg.norm(rotation)
            R = Rotation.from_rotvec(rotation * np.arcsin(sin_rot) / sin_rot)

            timestamp = np.random.uniform(0, livetime)

            event_info = parse_gibuu_event_info(event.optional[0])
            nu_in_trk = ROOT.Trk()
            nu_in_trk.id = 0
            nu_in_trk.mother_id = -1
            nu_in_trk.type = nu_type
            nu_in_trk.pos.set(*vtx_pos)
            nu_in_trk.dir.set(*direction)
            nu_in_trk.E = event_info["mom_lepton_in_E"].item()
            nu_in_trk.t = timestamp
            nu_in_trk.setusr("cc", is_cc)
            aafile.evt.mc_trks.push_back(nu_in_trk)

            lep_out_trk = ROOT.Trk()
            lep_out_trk.id = 1
            lep_out_trk.mother_id = 0
            lep_out_trk.type = sec_lep_type
            lep_out_trk.pos.set(*vtx_pos)
            mom = np.array(event_info[[
                "mom_lepton_out_x", "mom_lepton_out_y", "mom_lepton_out_z"
            ]].item())
            p_dir = R.apply(mom / np.linalg.norm(mom))
            lep_out_trk.dir.set(*p_dir)
            lep_out_trk.E = event_info["mom_lepton_out_E"].item()
            lep_out_trk.t = timestamp
            aafile.evt.mc_trks.push_back(lep_out_trk)

            for i, p in enumerate(event.particles):
                trk = ROOT.Trk()
                trk.id = i + 2
                mom = np.array([p.px, p.py, p.pz])
                p_dir = R.apply(mom / np.linalg.norm(mom))
                trk.pos.set(*vtx_pos)
                trk.dir.set(*p_dir)
                trk.mother_id = 0
                trk.type = int(p.id)
                trk.E = np.linalg.norm(mom)
                trk.t = timestamp
                aafile.evt.mc_trks.push_back(trk)
            aafile.write()
            if mc_event_id > 100:
                break

    del aafile
