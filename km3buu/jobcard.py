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
from os.path import basename, dirname, abspath, join, isfile

try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

INPUT_PATH = "/opt/buuinput2021/"

DEFAULT_JOBCARD_FILENAME = "jobcard.job"

PROCESS_LOOKUP = {"cc": 2, "nc": 3, "anticc": -2, "antinc": -3}
FLAVOR_LOOKUP = {"electron": 1, "muon": 2, "tau": 3}
PDGID_LOOKUP = {1: 12, 2: 14, 3: 16}
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


4132: 59 
-3322: 53
4232: 59
-311: 111
-3222: 53
-3312: 53
3312: 53
113: 103
-3212: 33
311: 110, 111
-3112: 33
3322: 53

33,53,59, 103, 110,111


DECAYED_HADRONS = [33, 53, 56, 57, 59, 103, 110, 111, 114, 115, 118, 119]


class Jobcard(f90nml.Namelist):
    """
    A object to manage GiBUU jobcard properties and format them

    Parameters
    ----------
    input_path: str
        The input path pointing to the GiBUU lookup data which should be used
    """
    def __init__(self,
                 *args,
                 filename=DEFAULT_JOBCARD_FILENAME,
                 input_path=INPUT_PATH,
                 **kwargs):
        super(Jobcard, self).__init__(*args, **kwargs)
        self.filename = filename
        self.input_path = INPUT_PATH
        self.__getitem__("input")["path_to_input"] = self.input_path

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
    return Jobcard(f90nml.read(filepath), filename=basename(filepath))


def generate_neutrino_jobcard(events,
                              process,
                              flavour,
                              energy,
                              target,
                              write_pert=True,
                              write_real=False,
                              do_decay=False,
                              photon_propagation=True,
                              fluxfile=None,
                              input_path=INPUT_PATH):  # pragma: no cover
    """
    Generate a jobcard for neutrino interaction

    Parameters
    ----------
    events: int
        Simulated number of neutrino events
    process: str
        Interaction channel ["CC", "NC", "antiCC", "antiNC"]
    flavour: str
        Flavour ["electron", "muon", "tau"]
    energy: tuple
        Initial energy range of the neutrino in GeV
    target: (int, int)
        (Z, A) describing the target nucleon
    write_pert: boolean (default: True)
        Write perturbative particles
    write_real: boolean (default: False)
        Write real particles
    do_decay: boolean (default: False)
        Decay final state particles using PYTHIA
    photon_propagation: boolean (default: True)
        Propagate photons and write it out
    fluxfile: str (default: None)
        Fluxfile, 1st col energy [GeV] and 2nd col flux [A.U.]
    input_path: str
        The input path pointing to the GiBUU lookup data which should be used
    """
    jc = read_jobcard(join(dirname(abspath(__file__)), "data/template.job"))
    jc["input"]["path_to_input"] = input_path
    # NEUTRINO
    jc["neutrino_induced"]["process_ID"] = PROCESS_LOOKUP[process.lower()]
    jc["neutrino_induced"]["flavor_ID"] = FLAVOR_LOOKUP[flavour.lower()]
    # TARGET
    jc["target"]["target_Z"] = target[0]
    jc["target"]["target_A"] = target[1]
    # EVENTS
    run_events = int(100000 / target[1])
    if events < run_events:
        run_events = events
        runs = 1
    else:
        runs = events // run_events
    jc["input"]["numEnsembles"] = run_events
    jc["input"]["num_runs_SameEnergy"] = runs
    # ENERGY
    jc["nl_neutrino_energyflux"]["eflux_min"] = energy[0]
    jc["nl_neutrino_energyflux"]["eflux_max"] = energy[1]
    # DECAY
    if do_decay:
        for i in DECAYED_HADRONS:
            key = "stabilityFlag({:d})".format(i)
            jc["ModifyParticles"][key] = 4
        jc["pythia"]["MDCY(102,1)"] = 1
    # FLUX
    if fluxfile is not None:
        if not isfile(fluxfile):
            raise IOError("Fluxfile {} not found!")
        jc["neutrino_induced"]["nuexp"] = 99
        jc["neutrino_induced"]["FileNameflux"] = fluxfile
    # OUTPUT
    jc["eventoutput"]["writeperturbativeparticles"] = write_pert
    jc["eventoutput"]["writerealparticles"] = write_real
    # MISC
    jc["insertion"]["propagateNoPhoton"] = not photon_propagation

    return jc
