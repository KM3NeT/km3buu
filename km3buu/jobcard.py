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
        self.input_path = "'%s'" % input_path
        self._groups = {}
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
        if group not in self._groups.keys():
            self._groups[group] = {}
        self._groups[group][name] = value

    def remove_property(self, group, name):
        del self._groups[group][name]
        if len(self._groups[group]) == 0:
            del self._groups[group]

    def __str__(self):
        retval = ""
        for group, attrs in self._groups.items():
            if len(self._groups[group]) == 0:
                continue
            retval += "&%s\n" % group
            for attr, value in attrs.items():
                if type(value) is bool:
                    retval += "\t%s = .%s.\n" % (attr, str(value).lower())
                else:
                    retval += "\t%s = %s\n" % (attr, value)
            retval += "/\n\n"
        return retval


def generate_neutrino_jobcard(process,
                              flavour,
                              energy,
                              target,
                              input_path=INPUT_PATH):
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
    jc.set_property("neutrino_inducted", "process_ID",
                    PROCESS_LOOKUP[process.lower()])
    jc.set_property("neutrino_inducted", "flavour_ID",
                    FLAVOUR_LOOKUP[flavour.lower()])
    jc.set_property("neutrino_inducted", "nuXsectionMode", 6)
    jc.set_property("neutrino_inducted", "includeDIS", True)
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
    return jc
