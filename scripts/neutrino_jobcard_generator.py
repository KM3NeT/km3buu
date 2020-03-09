#!/usr/bin/env python
# coding=utf-8
# Filename: neutrino_jobcard_generator.py
# Author: Johannes Schumann <jschumann@km3net.de>
"""
Convert ROOT and EVT files to HDF5.

Usage:
    neutrino_jobcard_generator.py
    neutrino_jobcard_generator.py (-h | --help)

Options:
    -h --help   Show this screen.

"""
from km3buu.ctrl import run_jobcard
from km3buu.jobcard import Jobcard
from km3buu.jobcard import XSECTIONMODE_LOOKUP, PROCESS_LOOKUP, FLAVOR_LOOKUP
from km3buu.output import GiBUUOutput
from thepipe.logger import get_logger
from tempfile import TemporaryDirectory
from os.path import dirname


def generate_neutrino_jobcard(process, flavor, energy, target):
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
    jc = Jobcard()
    # NEUTRINO
    jc.set_property("neutrino_induced", "process_ID",
                    PROCESS_LOOKUP[process.lower()])
    jc.set_property("neutrino_induced", "flavor_ID",
                    FLAVOR_LOOKUP[flavor.lower()])
    jc.set_property("neutrino_induced", "nuXsectionMode",
                    XSECTIONMODE_LOOKUP["dSigmaMC"])
    jc.set_property("neutrino_induced", "includeDIS", True)
    jc.set_property("neutrino_induced", "printAbsorptionXS", "T")
    jc.set_property("nl_SigmaMC", "enu", energy)
    # INPUT
    jc.set_property("input", "numTimeSteps", 0)
    jc.set_property("input", "eventtype", 5)
    jc.set_property("input", "numEnsembles", 100)
    jc.set_property("input", "delta_T", 0.2)
    jc.set_property("input", "localEnsemble", True)
    jc.set_property("input", "num_runs_SameEnergy", 1)
    jc.set_property("input", "LRF_equals_CALC_frame", True)
    # TARGET
    jc.set_property("target", "target_Z", target[0])
    jc.set_property("target", "target_A", target[1])
    # MISC
    # jc.set_property("nl_neutrinoxsection", "DISmassless", True)
    jc.set_property("neutrinoAnalysis", "outputEvents", True)
    jc.set_property("pythia", "PARP(91)", 0.44)
    return jc


def main():
    from docopt import docopt
    args = docopt(__doc__)


if __name__ == '__main__':
    main()
    log = get_logger("ctrl.py")
    log.setLevel("INFO")
    tmpjc = generate_neutrino_jobcard("cc", "electron", 1.0, (1, 1))
    datadir = TemporaryDirectory(dir=dirname(__file__))
    run_jobcard(tmpjc, datadir.name)
    data = GiBUUOutput(datadir.name)
    print(data.events[1:10])
    



