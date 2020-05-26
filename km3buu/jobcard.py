# Filename: jobcard.py
"""
Tools for creation of GiBUU jobcards

"""

__author__ = "Johannes Schumann"
__copyright__ = "Copyright 2020, Johannes Schumann and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Johannes Schumann"
__email__ = "jschumann@km3net.de"
__status__ = "Development"

import f90nml
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

INPUT_PATH = "/opt/buuinput2019/"

PROCESS_LOOKUP = {"cc": 2, "nc": 3, "anticc": -2, "antinc": -3}
FLAVOR_LOOKUP = {"electron": 1, "muon": 2, "tau": 3}
XSECTIONMODE_LOOKUP = {
    "integratedSigma": 0,
    "dSigmadCosThetadElepton": 1,
    "dSigmadQsdElepton": 2,
    "dSigmadQs": 3,
    "dSigmadCosTheta": 4,
    "dSigmadElepton": 5,
    "dSigmaMC": 6,
    "dSigmadW": 7,
    "EXP_dSigmadEnu": 10,
    "EXP_dSigmadCosThetadElepton": 11,
    "EXP_dSigmadQsdElepton": 12,
    "EXP_dSigmadQs": 13,
    "EXP_dSigmadCosTheta": 14,
    "EXP_dSigmadElepton": 15,
    "EXP_dSigmaMC": 16,
    "EXP_dSigmadW": 17
}


class Jobcard(object):
    """
    A object to manage GiBUU jobcard properties and format them

    Parameters
    ----------
    input_path: str
        The input path pointing to the GiBUU lookup data which should be used
    """
    def __init__(self, input_path=INPUT_PATH):
        self.input_path = "%s" % input_path
        self._nml = f90nml.Namelist()
        self.set_property("input", "path_to_input", self.input_path)

    def set_property(self, group, name, value):
        """ Set a property to the jobcard

        Parameters
        ----------
        group: str
            The parameter group where the property is incorporated
        name: str
            The property name
        value:
            The property value
        """
        if group not in self._nml.keys():
            self._nml[group] = {}
        self._nml[group][name] = value

    def remove_property(self, group, name):
        del self._nml[group][name]
        if len(self._nml[group]) == 0:
            del self._nml[group]

    def __str__(self):
        stream = StringIO()
        self._nml.write(stream)
        return stream.getvalue()

    def __getitem__(self, key):
        if isinstance(key, str):
            if '/' in key:
                k = key.split('/')
                return self._nml[k[0]][k[1]]
            else:
                return self._nml[key]
        elif isinstance(key, tuple) and len(key) == 2:
            return self._nml[key[0]][key[1]]
        else:
            raise IndexError("Invalid access to Jobcard")


def read_jobcard(fpath):
    """ Read a jobcard from file

    Parameters
    ----------
    fpath: str
        Filepath of the jobcard
    """
    jc = Jobcard()
    jc._nml = f90nml.read(fpath)
    return jc


def generate_neutrino_jobcard_template(
    process,
    flavour,
    energy,
    target,
    write_events=False,
    input_path=INPUT_PATH):  # pragma: no cover
    """
    Generate a jobcard for neutrino interaction

    Parameters
    ----------
    process: str
        Interaction channel ["CC", "NC", "antiCC", "antiNC"]
    flavour: str
        Flavour ["electron", "muon", "tau"]
    energy: float
        Initial energy of the neutrino in GeV
    target: (int, int)
        (Z, A) describing the target nukleon
    input_path: str
        The input path pointing to the GiBUU lookup data which should be used
    """
    jc = Jobcard(input_path)
    # NEUTRINO
    jc.set_property("neutrino_induced", "process_ID",
                    PROCESS_LOOKUP[process.lower()])
    jc.set_property("neutrino_induced", "flavour_ID",
                    FLAVOR_LOOKUP[flavour.lower()])
    jc.set_property("neutrino_induced", "nuXsectionMode", 6)
    jc.set_property("neutrino_induced", "includeDIS", True)
    jc.set_property("neutrino_induced", "includeDELTA", True)
    jc.set_property("neutrino_induced", "includeRES", True)
    jc.set_property("neutrino_induced", "includeQE", True)
    jc.set_property("neutrino_induced", "include1pi", True)
    jc.set_property("neutrino_induced", "include2p2hQE", True)
    jc.set_property("neutrino_induced", "include2pi", False)
    jc.set_property("neutrino_induced", "include2p2hDelta", False)
    jc.set_property("neutrino_inducted", "printAbsorptionXS", "T")

    # INPUT
    jc.set_property("input", "numTimeSteps", 0)
    jc.set_property("input", "eventtype", 5)
    jc.set_property("input", "numEnsembles", 100000)
    jc.set_property("input", "delta_T", 0.2)
    jc.set_property("input", "localEnsemble", True)
    jc.set_property("input", "num_runs_SameEnergy", 1)
    # TARGET
    jc.set_property("target", "Z", target[0])
    jc.set_property("target", "A", target[1])
    # MISC
    jc.set_property("neutrinoAnalysis", "outputEvents", write_events)
    return jc
