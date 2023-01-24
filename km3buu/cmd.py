#!/usr/bin/env python
# coding=utf-8
# Filename: cmd.py
# Author: Johannes Schumann <jschumann@km3net.de>

import numpy as np
import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
from os.path import join

from km3buu.jobcard import generate_neutrino_jobcard
from km3buu.ctrl import run_jobcard
from km3buu.geometry import CanVolume, SphericalVolume, NoVolume
from km3buu.output import GiBUUOutput, write_detector_file

ARGPARSE_DESC = {
    "prog": "km3buu",
    "description": "Runscript for GIBUU/KM3BUU",
    "epilog": "https://git.km3net.de/simulation/km3buu"
}

ARGPARSE_GENERAL_PARAMS = [{
    "option_strings": ["--events", "-n"],
    "dest": "events",
    "type": int,
    "help": "Number of events which are simulated",
    "required": True
}, {
    "option_strings": ["--multifile", "-m"],
    "dest": "multifile",
    "type": int,
    "help": "How many km3net files to write from the dataset",
    "required": False,
    "default": 1
}, {
    "option_strings": ["--seed", "-s"],
    "dest": "seed",
    "type": int,
    "help": "Seed which should be used for the (pseudo) random number gen.",
    "required": False,
    "default": 0
}, {
    "option_strings": ["--flavor", "-f"],
    "dest": "flavor",
    "choices": ["electron", "muon", "tau"],
    "help": "Simulated neutrino flavor",
    "required": True
}, {
    "option_strings": ["--interaction", "-i"],
    "dest": "interaction",
    "choices": ["nc", "cc", "antinc", "anticc"],
    "help": "The current type of the weak interaction",
    "required": True
}, {
    "option_strings": ["--target", "-t"],
    "dest": "target",
    "nargs": 2,
    "type": int,
    "help": "The number of nucleons/protons in the target nucleus",
    "metavar": ("A", "Z"),
    "required": True
}, {
    "option_strings": ["--geometry", "-g"],
    "dest":
    "geometry",
    "choices": ["no", "can", "sphere"],
    "default":
    "no",
    "help":
    "Type of detector enviroment geometry should be used"
}, {
    "option_strings": ["--center", "-c"],
    "dest":
    "center",
    "type":
    float,
    "nargs":
    3,
    "metavar": ("x", "y", "z"),
    "default": [0, 0, 0],
    "help":
    "Center (offset) if a geometry is used (otherwise ignored)"
}, {
    "option_strings": ["--dimensions", "-d"],
    "dest":
    "dimensions",
    "type":
    float,
    "nargs":
    "*",
    "help":
    "Dimensions of the geometry; sphere -> -d <radius> / can -> -d <radius> <zmin> <zmax>"
}, {
    "option_strings": ["--output-dir", "-o"],
    "dest": "output",
    "type": Path,
    "default": Path("."),
    "help": "Output directory"
}, {
    "option_strings": ["--run", "-r"],
    "dest": "runnumber",
    "type": int,
    "default": 1,
    "help": "Run number to use"
}, {
    "option_strings": ["--gibuuparams", "-p"],
    "dest":
    "gibuuparams",
    "type":
    Path,
    "help":
    "JSON file for modified GiBUU namelist params"
}, {
    "option_strings": ["--taupropagation"],
    "dest": "tauprop",
    "action": argparse.BooleanOptionalAction,
    "help": "Do tau propagation",
    "default": False
}, {
    "option_strings": ["--zenith", "-z"],
    "dest": "zenith",
    "type": float,
    "nargs": 2,
    "help": "Zenith range of the direction if a geometry is used",
    "metavar": ("cosZmin", "cosZmax"),
    "required": False,
    "default": [-1, 1]
}]


def main():
    #
    # ARG PARSE
    #
    parser = argparse.ArgumentParser(**ARGPARSE_DESC)
    for kwargs in ARGPARSE_GENERAL_PARAMS:
        args = kwargs.pop("option_strings")
        parser.add_argument(*args, **kwargs)
    subparsers = parser.add_subparsers(title="modes", help="Modes")
    # Single Energy
    single_energy_parser = subparsers.add_parser(
        "single", help="Run in single energy mode")
    single_energy_parser.add_argument("-e",
                                      "--energy",
                                      dest="energy",
                                      type=float,
                                      help="Neutrino energy [GeV]",
                                      required=True)
    # Energy Range
    energy_range_parser = subparsers.add_parser(
        "range", help="Run in energy range mode")
    energy_range_parser.add_argument("-e",
                                     "--energy",
                                     dest="energy",
                                     type=float,
                                     nargs=2,
                                     help="Neutrino energy range [GeV]",
                                     metavar=("Emin", "Emax"),
                                     required=True)
    energy_range_parser.add_argument(
        "-y",
        "--gamma",
        dest="flux",
        type=float,
        help="The power law index of the simulated neutrino flux",
        default=0.0)
    args = parser.parse_args()

    single_energy_run = type(args.energy) is float
    energy = args.energy if single_energy_run else tuple(args.energy)

    if single_energy_run:
        descriptor = "{0}_{1}_{2}GeV_A{3}Z{4}".format(args.flavor,
                                                      args.interaction,
                                                      args.energy,
                                                      args.target[0],
                                                      args.target[1])
    else:
        descriptor = "{0}_{1}_{2}-{3}GeV_A{4}Z{5}_power_law_{6:.1f}".format(
            args.flavor, args.interaction, energy[0], energy[1],
            args.target[0], args.target[1], args.flux)

    gibuu_dir = Path(join(args.output, descriptor))
    gibuu_dir.mkdir(exist_ok=True)
    #
    # FLUX
    #
    fluxfile = None

    if not single_energy_run and not np.isclose(args.flux, 0.):
        energies = np.linspace(energy[0], energy[1], 1000)
        flux = 1e3 * energies**args.flux
        fluxfile = join(args.output, "flux.dat")
        np.savetxt(fluxfile, np.c_[energies, flux])

    jc = generate_neutrino_jobcard(args.events,
                                   args.interaction,
                                   args.flavor,
                                   energy,
                                   args.target,
                                   seed=args.seed,
                                   fluxfile=fluxfile,
                                   do_decay=False)

    jc["neutrinoanalysis"]["outputEvents"] = True
    jc["neutrinoanalysis"]["inclusiveAnalysis"] = False

    if args.gibuuparams:
        with open(args.gibuuparams) as f:
            additional_args = json.load(f)

        for k1, param_dct in additional_args.items():
            for k2, v in param_dct.items():
                jc[k1][k2] = v

    run_jobcard(jc, gibuu_dir)

    fobj = GiBUUOutput(gibuu_dir)

    if args.geometry == 'no':
        volume = NoVolume()
    elif args.geometry == 'sphere':
        volume = SphericalVolume(args.dimensions[0],
                                 tuple(args.center),
                                 zenith=args.zenith)
    elif args.geometry == 'can':
        kwargs = {"detector_center": tuple(args.center), "zenith": args.zenith}
        if args.dimensions:
            kwargs["radius"] = args.dimensions[0]
            kwargs["zmin"] = args.dimensions[1]
            kwargs["zmax"] = args.dimensions[2]
        volume = CanVolume(**kwargs)
    run_descriptor = ""
    if args.runnumber:
        run_descriptor = "run{:08d}_".format(args.runnumber)

    outfilename = join(args.output,
                       "km3buu_" + run_descriptor + descriptor + ".root")

    write_detector_file(fobj,
                        geometry=volume,
                        ofile=outfilename,
                        run_number=args.runnumber,
                        no_files=args.multifile,
                        propagate_tau=args.tauprop)


if __name__ == '__main__':
    main()
