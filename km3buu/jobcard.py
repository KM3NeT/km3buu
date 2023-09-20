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
import numpy as np
from os.path import basename, dirname, abspath, join, isfile
from os import environ

try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

INPUT_PATH = environ.get("CONTAINER_GIBUU_INPUT")
if INPUT_PATH is None:
    INPUT_PATH = "/opt/buuinput2021/"

BLOCKING_FACTOR = 2

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

DECAYED_HADRONS = [56, 57, 114, 115, 118, 119]
MAX_OVERALL_ENSEMBLES = 100000
MIN_RUN_ENSEMBLES = 100


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


def write_jobcard(jobcard, filepath):
    with open(filepath, 'w') as nml_file:
        f90nml.write(jobcard, nml_file)


def estimate_number_of_ensembles(events, target):
    """
    Estimate a suiting ensemble configuration for GiBUU based on a 
    the desired number of events
    Parameters
    ----------
    events: int
        Number of events which should be generated
    target: (int, int)
        (A, Z) properties of the target nucleus
    Return
    ------
    tuple (int, int) [ensembles, runs] 
    """

    run_ensembles = int(np.ceil(MAX_OVERALL_ENSEMBLES / target[0]))
    if run_ensembles < MIN_RUN_ENSEMBLES:
        run_ensembles = MIN_RUN_ENSEMBLES

    required_total_ensembles = int(np.ceil(events / target[0])) * BLOCKING_FACTOR
    if required_total_ensembles < MIN_RUN_ENSEMBLES:
        required_total_ensembles = MIN_RUN_ENSEMBLES

    if required_total_ensembles <= run_ensembles:
        ensembles = required_total_ensembles
        runs = 1
    else:
        ensembles = run_total_ensembles
        runs = int(np.ceil(required_total_ensembles / run_ensembles))
    return (ensembles, runs)


def generate_neutrino_jobcard(ensembles,
                              runs,
                              process,
                              flavour,
                              energy,
                              target,
                              write_pert=True,
                              write_real=True,
                              do_decay=False,
                              photon_propagation=True,
                              fluxfile=None,
                              seed=None,
                              timesteps=-1,
                              input_path=INPUT_PATH):  # pragma: no cover
    """
    Generate a jobcard for neutrino interaction

    Parameters
    ----------
    ensembles: int
        Simulated number of ensembles per nucleon & runs, which will result in #events < ensembles * runs
    runs: int
        Number of runs which should be executed by   
    process: str
        Interaction channel ["CC", "NC", "antiCC", "antiNC"]
    flavour: str
        Flavour ["electron", "muon", "tau"]
    energy: float, tuple
        Initial energy or energy range (emin, emax) of the primary neutrino in GeV
    target: (int, int)
        (A, Z) properties of the target nucleus
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
    seed: int (default: 0)
        Input seed for the random number generator in GiBUU 
        (0: some seed will be drawn based on system time)
    timesteps: int (default: -1)
        Number of timesteps done by GiBUU
        (-1: Default value from GiBUU is used)
    input_path: str
        The input path pointing to the GiBUU lookup data which should be used
    """
    jc = read_jobcard(join(dirname(abspath(__file__)), "data/template.job"))
    jc["input"]["path_to_input"] = input_path
    # NEUTRINO
    jc["neutrino_induced"]["process_ID"] = PROCESS_LOOKUP[process.lower()]
    jc["neutrino_induced"]["flavor_ID"] = FLAVOR_LOOKUP[flavour.lower()]
    # TARGET
    jc["target"]["z"] = target[1]
    jc["target"]["a"] = target[0]
    # FSI
    if timesteps >= 0:
        jc["input"]["numTimeSteps"] = timesteps
    # EVENTS
    jc["input"]["numEnsembles"] = ensembles
    jc["input"]["num_runs_SameEnergy"] = runs
    # ENERGY
    if isinstance(energy, tuple):
        jc["nl_neutrino_energyflux"]["eflux_min"] = energy[0]
        jc["nl_neutrino_energyflux"]["eflux_max"] = energy[1]
    else:
        jc["nl_sigmamc"]["enu"] = energy
        jc["neutrino_induced"]["nuXsectionMode"] = XSECTIONMODE_LOOKUP[
            "dSigmaMC"]
        jc["neutrino_induced"]["nuExp"] = 0
    # DECAY
    if do_decay:
        for i in DECAYED_HADRONS:
            key = "stabilityFlag({:d})".format(i)
            jc["ModifyParticles"][key] = 4
        jc["pythia"]["MDCY(102,1)"] = 1
    # FLUX
    if fluxfile is not None and isinstance(energy, tuple):
        if not isfile(fluxfile):
            raise IOError("Fluxfile {} not found!")
        jc["neutrino_induced"]["nuexp"] = 99
        jc["neutrino_induced"]["FileNameflux"] = fluxfile
    # OUTPUT
    jc["eventoutput"]["writeperturbativeparticles"] = write_pert
    jc["eventoutput"]["writerealparticles"] = write_real
    # MISC
    if seed:
        jc["initRandom"]["Seed"] = seed
    jc["insertion"]["propagateNoPhoton"] = not photon_propagation

    return jc
