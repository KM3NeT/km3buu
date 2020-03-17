#!/usr/bin/env python
# coding=utf-8
# Filename: plot_cc_xsection.py
# Author: Johannes Schumann <jschumann@km3net.de>
"""
Generate 

Usage:
    neutrino_jobcard_generator.py OUTFILE [--threads=<n>] [--verbose]
    neutrino_jobcard_generator.py (-h | --help)

Options:
    -h --help       Show this screen.
    OUTFILE         Filename for the output pdf containing the plot
    --threads=<n>   Number of worker threads which should be used to process data
                    parallel [default: 1]
    --verbose       Display GiBUU output
"""
import os
import time
import click
import numpy as np
import matplotlib.pyplot as plt
from thepipe.logger import get_logger
from tempfile import TemporaryDirectory
from os.path import dirname, abspath
from collections import defaultdict

from concurrent.futures import ThreadPoolExecutor

from km3buu.ctrl import run_jobcard
from km3buu.jobcard import Jobcard
from km3buu.jobcard import XSECTIONMODE_LOOKUP, PROCESS_LOOKUP, FLAVOR_LOOKUP
from km3buu.output import GiBUUOutput


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
    jc.set_property("neutrino_induced", "includeDELTA", True)
    jc.set_property("neutrino_induced", "includeRES", True)
    jc.set_property("neutrino_induced", "includeQE", True)
    jc.set_property("neutrino_induced", "include1pi", True)
    jc.set_property("neutrino_induced", "include2p2hQE", True)
    jc.set_property("neutrino_induced", "include2pi", False)
    jc.set_property("neutrino_induced", "include2p2hDelta", False)
    jc.set_property("neutrino_induced", "printAbsorptionXS", "T")
    jc.set_property("nl_SigmaMC", "enu", energy)
    # INPUT
    jc.set_property("input", "numTimeSteps", 0)
    jc.set_property("input", "eventtype", 5)
    jc.set_property("input", "numEnsembles", 1000)
    jc.set_property("input", "delta_T", 0.2)
    jc.set_property("input", "localEnsemble", True)
    jc.set_property("input", "num_runs_SameEnergy", 1)
    jc.set_property("input", "LRF_equals_CALC_frame", True)
    # TARGET
    jc.set_property("target", "target_Z", target[0])
    jc.set_property("target", "target_A", target[1])
    # MISC
    # jc.set_property("nl_neutrinoxsection", "DISmassless", True)
    jc.set_property("neutrinoAnalysis", "outputEvents", False)
    jc.set_property("pythia", "PARP(91)", 0.44)
    return jc


def worker(energy):
    # create a neutrino jobcard for oxygen
    tmpjc = generate_neutrino_jobcard("cc", "electron", energy, (8, 16))
    datadir = TemporaryDirectory(dir=dirname(__file__),
                                 prefix="%sGeV" % energy)
    run_jobcard(tmpjc, datadir.name)
    data = GiBUUOutput(datadir.name)
    return data


def main():
    from docopt import docopt
    args = docopt(__doc__)
    if args["--verbose"]:
        log.setLevel("INFO")
        log = get_logger("ctrl.py")
    workers = int(args["--threads"])
    xsections = defaultdict(list)
    energies = np.logspace(-1, 1, 50)
    tasks = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for i, energy in enumerate(energies):
            tasks[i] = executor.submit(worker, energy)
        while True:
            time.sleep(1)
            status = {k: v.done() for k, v in tasks.items()}
            click.clear()
            for thr, st in status.items():
                click.echo("Thread %s (%s): %s" % (thr, energies[thr], st))
            if all(status.values()):
                break
    for i, res in tasks.items():
        data = res.result()
        for k in ['sum', 'QE', 'highRES', 'DIS', 'Delta']:
            xsections[k].append(data.xsection[k])
    for k in ['sum', 'QE', 'highRES', 'DIS', 'Delta']:
        plt.plot(energies, np.divide(xsections[k], energies), label=k)
    plt.xlabel("Energy [GeV]")
    plt.ylabel("Crosssection")
    plt.legend()
    plt.xscale("log")
    plt.grid()
    plt.savefig(args["OUTFILE"])


if __name__ == '__main__':
    main()
