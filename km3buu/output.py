# Filename: io.py
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
from io import StringIO
from os import listdir
from os.path import isfile, join, abspath
from tempfile import TemporaryDirectory

from .jobcard import Jobcard

EVENT_FILENAME = "FinalEvents.dat"
XSECTION_FILENAMES = {"all": "neutrino_absorption_cross_section_ALL.dat"}


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
    header = re.sub(r'\d+:|#', '', lines[0]).split()
    dt = np.dtype([(field, np.float64) for field in header])
    values = np.genfromtxt(StringIO(lines[-1]), dtype=dt)
    return values


class FinalEvents:
    def __init__(self, filepath):
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
            next_pos = self._final_events_map.find(b'\n') + 1
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
            line = self._final_events_map.readline().decode('utf-8').strip(
                "\n")
            s.append("%s %.3f %3f\n" % (line, 0., 0.))
        dt = np.dtype([('run', np.uint32), ('event', np.uint32),
                       ('id', np.int32), ('charge', np.float64),
                       ('perweight', np.float64), ('x', np.float64),
                       ('y', np.float64), ('z', np.float64),
                       ('p_t', np.float64), ('p_x', np.float64),
                       ('p_y', np.float64), ('p_z', np.float64),
                       ('history', np.int32), ('pID', np.int32),
                       ('nu_energy', np.float64), ('Bx', np.float64),
                       ('By', np.float64)])
        values = np.genfromtxt(StringIO('\n'.join(s)), dtype=dt)
        # values['Bx'] = / 2. /
        values['By'] = 1 - values["p_t"] / values["nu_energy"]
        return values

    def __len__(self):
        return len(self._line_pos)


OUTPUT_FILE_WRAPPERS = {'FinalEvents.dat': FinalEvents}


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
            setattr(self, "events",
                    FinalEvents(join(self._data_path, EVENT_FILENAME)))
        if XSECTION_FILENAMES["all"] in self.output_files:
            setattr(
                self, "xsection",
                read_nu_abs_xsection(
                    join(self._data_path, XSECTION_FILENAMES["all"])))
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
