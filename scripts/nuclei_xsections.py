#!/usr/bin/env python
# coding=utf-8
# Filename: nuclei_xsections.py
# Author: Johannes Schumann <jschumann@km3net.de>
"""
Generate 

Usage:
    nuclei_xsections.py DATAFILE [--Zmin=<Zmin>] [--Zmax=<Zmax>] [--threads=<n>] [--verbose] [--anticc]
    nuclei_xsections.py DATAFILE PLOTFILE
    nuclei_xsections.py (-h | --help)

Options:
    -h --help       Show this screen.
    OUTFILE         Filename for the hdf5 output containing the plot data
    PLOTFILE        Filename for the pdf output containing the plots
    --threads=<n>   Number of worker threads which should be used to process data
                    parallel [default: 1]
    --verbose       Display GiBUU output
"""
import os
import time
import numpy as np
import km3pipe as kp
import matplotlib.pyplot as plt
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


def worker(energy, Z, anti_flag=False):
    process = "cc"
    if anti_flag:
        process = "anticc"
    # create a neutrino jobcard for oxygen
    tmpjc = generate_neutrino_jobcard(process, "electron", energy, (Z, 2 * Z))
    datadir = TemporaryDirectory(dir=dirname(__file__),
                                 prefix="%sGeV" % energy)
    run_jobcard(tmpjc, datadir.name)
    data = GiBUUOutput(datadir)
    return data


def plot(datafile, plotfile):
    plt.xlabel("Energy [GeV]")
    plt.ylabel(r"$\nu$ Crosssection / E [$10^-38cm^{cm^2}/GeV$]")
    plt.legend()
    plt.xscale("log")
    plt.grid()
    plt.savefig(plotfile)
    pass


class XSectionPump(kp.Module):
    def configure(self):
        self.tasks = self.require('tasks')
        self.energies = self.require('energies')
        self.Zrange = self.require('zrange')

    def process(self, blob):
        blob = kp.Blob()
        if len(self.tasks) == 0:
            raise StopIteration
        key = list(self.tasks.keys())[0]
        task = self.tasks.pop(key)
        res = task.result()

        dct = dict(res.xsection._xsections)
        dct.update({'energy': self.energies[key[0]], 'Z': self.Zrange[key[1]]})
        blob['Xsection'] = kp.Table(dct, h5loc='xsec')
        return blob


def main():
    from docopt import docopt
    args = docopt(__doc__)
    datafile = args['DATAFILE']
    if args['PLOTFILE']:
        plotfile = args['PLOTFILE']
        plot(datafile, plotfile)
        return

    if args["--verbose"]:
        log = kp.logger.get_logger("ctrl.py")
        log.setLevel("INFO")
    workers = int(args["--threads"])
    xsections = defaultdict(list)

    Zmin = 1
    if args['--Zmin']:
        Zmin = int(args['--Zmin'])

    Zmax = 16
    if args['--Zmax']:
        Zmax = int(args['--Zmax'])

    energies = np.logspace(-1, 1, 2)
    Zrange = range(Zmin, Zmax + 1)
    tasks = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for i, energy in enumerate(energies):
            for j, Z in enumerate(Zrange):
                tasks[i, j] = executor.submit(worker, energy, Z,
                                              args["--anticc"])
        if args["--verbose"]:
            import click
            while True:
                time.sleep(1)
                status = {k: v.done() for k, v in tasks.items()}
                click.clear()
                for idx, st in status.items():
                    click.echo("Energy %.3f - Z %d: %s" %
                               (energies[idx[0]], Zrange[idx[1]], st))
                if all(status.values()):
                    break

    pipe = kp.Pipeline()
    pipe.attach(XSectionPump, tasks=tasks, energies=energies, zrange=Zrange)
    pipe.attach(kp.io.HDF5Sink, filename=datafile)
    pipe.drain()

    # for i, res in tasks.items():
    #     data = res.result()
    #     for k in ['sum', 'QE', 'highRES', 'DIS', 'Delta']:
    #         xsections[k].append(data.xsection[k])
    # for k in ['sum', 'QE', 'highRES', 'DIS', 'Delta']:
    #     plt.plot(energies, np.divide(xsections[k], energies), label=k)


if __name__ == '__main__':
    main()
