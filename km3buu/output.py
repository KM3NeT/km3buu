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

import mmap
import numpy as np
from io import StringIO
from os import listdir
from os.path import isfile, join, abspath


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
        return line_pos

    def __getitem__(self, i):
        if isinstance(i, slice):
            pos = self._line_pos[i]
        else:
            pos = [self._line_pos[i]]
        s = []
        for p in pos:
            self._final_events_map.seek(p)
            s.append(self._final_events_map.readline().decode('utf-8'))
        dt = np.dtype([('run', np.uint32), ('event', np.uint32),
                       ('id', np.int32), ('charge', np.float64),
                       ('perweight', np.float64), ('x', np.float64),
                       ('y', np.float64), ('z', np.float64),
                       ('p_x', np.float64), ('p_y', np.float64),
                       ('p_z', np.float64), ('history', np.int32),
                       ('pID', np.int32), ('energy', np.float64)])
        return np.genfromtxt(StringIO('\n'.join(s)), dtype=dt)[::1]

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
        self._data_dir = abspath(data_dir)
        self.output_files = [
            f for f in listdir(self._data_dir)
            if isfile(join(self._data_dir, f))
        ]
        if "FinalEvents.dat" in self.output_files:
            setattr(self, "events",
                    FinalEvents(join(self._data_dir, "FinalEvents.dat")))
