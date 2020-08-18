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
from io import StringIO
from os import listdir
from os.path import isfile, join, abspath
from tempfile import TemporaryDirectory
from collections import defaultdict
import awkward1
import itertools
from scipy.spatial.transform import Rotation

from .jobcard import Jobcard

EVENT_FILENAME = "FinalEvents.dat"
LHE_PERT_FILENAME = "EventOutput.Pert.[0-9]{8}.lhe"
LHE_REAL_FILENAME = "EventOutput.Real.[0-9]{8}.lhe"
XSECTION_FILENAMES = {"all": "neutrino_absorption_cross_section_ALL.dat"}

EVENT_FILE_DTYPE = np.dtype([
    ("run", np.uint32),
    ("event", np.uint32),
    ("id", np.int32),
    ("charge", np.float64),
    ("perweight", np.float64),
    ("x", np.float64),
    ("y", np.float64),
    ("z", np.float64),
    ("p_t", np.float64),
    ("p_x", np.float64),
    ("p_y", np.float64),
    ("p_z", np.float64),
    ("history", np.int32),
    ("pID", np.int32),
    ("nu_energy", np.float64),
    ("Bx", np.float64),
    ("By", np.float64),
])

LHE_NU_INFO_DTYPE = np.dtype([('type', np.int), ('weight', np.float64),
                              ('mom_lepton_in_E', np.float64),
                              ('mom_lepton_in_x', np.float64),
                              ('mom_lepton_in_y', np.float64),
                              ('mom_lepton_in_z', np.float64),
                              ('mom_lepton_out_E', np.float64),
                              ('mom_lepton_out_x', np.float64),
                              ('mom_lepton_out_y', np.float64),
                              ('mom_lepton_out_z', np.float64),
                              ('mom_nucleon_in_E', np.float64),
                              ('mom_nucleon_in_x', np.float64),
                              ('mom_nucleon_in_y', np.float64),
                              ('mom_nucleon_in_z', np.float64)])


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


class FinalEvents:
    """ 
    Reader for FinalEvents.dat

    Parameters
    ----------
    filepath: str
        Filepath pointing to the FinalEvents file
    """
    def __init__(self, filepath):
        """ Constructor """
        self._filepath = filepath
        with open(self._filepath, "r") as f:
            self._final_events_map = mmap.mmap(f.fileno(),
                                               0,
                                               prot=mmap.PROT_READ)
        self._line_pos = self._line_mapper()

    def _line_mapper(self):
        self._final_events_map.seek(0)
        line_pos = []
        while True:
            next_pos = self._final_events_map.find(b"\n") + 1
            if next_pos == 0:
                break
            line_pos.append(next_pos)
            self._final_events_map.seek(line_pos[-1])
        return line_pos[:-1]

    def __getitem__(self, i):
        if isinstance(i, slice):
            pos = self._line_pos[i]
        else:
            pos = [self._line_pos[i]]
        s = []
        for p in pos:
            self._final_events_map.seek(p)
            line = self._final_events_map.readline().decode("utf-8").strip(
                "\n")
            s.append("%s %.3f %3f\n" % (line, 0.0, 0.0))
        values = np.genfromtxt(StringIO("\n".join(s)), dtype=EVENT_FILE_DTYPE)
        # values['Bx'] = / 2. /
        values["By"] = 1 - values["p_t"] / values["nu_energy"]
        return values

    def __len__(self):
        return len(self._line_pos)


OUTPUT_FILE_WRAPPERS = {"FinalEvents.dat": FinalEvents}


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
        if EVENT_FILENAME in self.output_files:
            setattr(self, "final_events",
                    FinalEvents(join(self._data_path, EVENT_FILENAME)))

        if XSECTION_FILENAMES["all"] in self.output_files:
            setattr(
                self,
                "xsection",
                read_nu_abs_xsection(
                    join(self._data_path, XSECTION_FILENAMES["all"])),
            )
        lhe_pert_regex = re.compile(LHE_PERT_FILENAME)
        self.lhe_pert_files = list(
            filter(lhe_pert_regex.match, self.output_files))

        lhe_real_regex = re.compile(LHE_REAL_FILENAME)
        self.lhe_real_files = list(
            filter(lhe_real_regex.match, self.output_files))

        jobcard_regex = re.compile('.job')
        jobcard_files = list(filter(jobcard_regex.match, self.output_files))
        if len(jobcard_files) == 1:
            self._jobcard = jobcard_files[0]
        else:
            self._jobcard = None

    def associate_jobcard(self, jobcard):
        """
        Append a jobcard to the wrapped output directory

        Parameters
        ----------
        jobcard: Jobcard
            Jobcard object instance
        """
        self._jobcard = jobcard


def write_detector_file(gibuu_output,
                        ofile="gibuu.aanet.root",
                        can=(0, 476.5, 403.4),
                        livetime=3.156e7):
    import aa, ROOT
    fobj = ROOT.EventFile()
    fobj.set_output(ofile)
    mc_event_id = 0

    for ifile in gibuu_output.lhe_pert_files:
        lhe_data = pylhe.readLHEWithAttributes(ifile)
        for event in lhe_data:
            fobj.evt.clear()
            fobj.evt.id = mc_event_id
            fobj.evt.mc_run_id = mc_event_id
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

            # event_info = parse_gibuu_event_info(event.optional[0])
            # p_lepton_in = event_info
            for i, p in enumerate(event.particles):
                trk = ROOT.Trk()
                trk.id = i
                mom = np.array([p.px, p.py, p.pz])
                p_dir = mom / np.linalg.norm(mom)
                trk.pos.set(*vtx_pos)
                trk.dir.set(*p_dir)
                trk.mother_id = 0
                trk.type = int(p.id)
                trk.E = np.linalg.norm(mom)
                trk.t = timestamp
                fobj.evt.mc_trks.push_back(trk)
            fobj.write()

    del fobj
