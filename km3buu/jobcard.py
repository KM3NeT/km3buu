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
    "EXP_dSigmadW": 17,
}


class Jobcard(f90nml.Namelist):
    """
    A object to manage GiBUU jobcard properties and format them

    Parameters
    ----------
    input_path: str
        The input path pointing to the GiBUU lookup data which should be used
    """
    def __init__(self, *args, **kwargs):
        if "input_path" in kwargs:
            self.input_path = "%s" % input_path
            del kwargs["input_path"]
        else:
            self.input_path = INPUT_PATH
        self.__getitem__("input")["path_to_input"] = self.input_path
        super(Jobcard, self).__init__(*args, **kwargs)

    def __getitem__(self, key):
        if not self.__contains__(key):
            self.__setitem__(key, f90nml.Namelist())
        return super(Jobcard, self).__getitem__(key)

    def _clean_namelist(self):
        for k, v in self.items():
            if isinstance(v, f90nml.Namelist) and len(v) == 0:
                self.__delitem__(k)

    def __str__(self):
        self._clean_namelist()
        stream = StringIO()
        self.write(stream)
        return stream.getvalue()


def read_jobcard(filepath):
    return Jobcard(f90nml.read(filepath))


def generate_neutrino_jobcard_template(
    process,
    flavour,
    energy_limits,
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
    energy_range: (float, float)
        Energy range limits of the incoming neutrino flux in GeV
    target: (int, int)
        (Z, A) describing the target nukleon
    input_path: str
        The input path pointing to the GiBUU lookup data which should be used
    """
    jc = Jobcard(input_path)
    # NEUTRINO
    jc["neutrino_induced"]["process_ID"] = PROCESS_LOOKUP[process.lower()]
    jc["neutrino_induced"]["flavour_ID"] = FLAVOR_LOOKUP[flavour.lower()]
    jc["neutrino_induced"]["nuXsectionMode"] = XSECTIONMODE_LOOKUP[
        "EXP_dSigmaMC"]
    jc["neutrino_induced"]["includeDIS"] = True
    jc["neutrino_induced"]["includeDELTA"] = True
    jc["neutrino_induced"]["includeRES"] = True
    jc["neutrino_induced"]["includeQE"] = True
    jc["neutrino_induced"]["include1pi"] = True
    jc["neutrino_induced"]["include2p2hQE"] = True
    jc["neutrino_induced"]["include2pi"] = False
    jc["neutrino_induced"]["include2p2hDelta"] = False
    jc["neutrino_induced"]["printAbsorptionXS"] = True
    jc["neutrino_induced"]["nuExp"] = 99
    # INPUT
    jc["input"]["numTimeSteps"] = 0
    jc["input"]["eventtype"] = 5
    jc["input"]["numEnsembles"] = 10000
    jc["input"]["delta_T"] = 0.2
    jc["input"]["localEnsemble"] = True
    jc["input"]["num_runs_SameEnergy"] = 1
    # FLUX
    jc["nl_fluxcuts"]["energylimit_for_Qsrec"] = True
    jc["nl_neutrino_energyFlux"]["Eflux_min"] = energy_limits[0]
    jc["nl_neutrino_energyFlux"]["Eflux_max"] = energy_limits[1]
    # TARGET
    jc["target"]["Z"] = target[0]
    jc["target"]["A"] = target[1]
    # MISC
    jc["neutrinoAnalysis"]["outputEvents"] = False
    jc["EventOutput"]["EventFormat"] = 1
    jc["EventOutput"]["WritePerturbativeParticles"] = True
    jc["EventOutput"]["WriteRealParticles"] = False
    return jc
