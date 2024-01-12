#!/usr/bin/env python
# coding=utf-8
# Filename: cmd.py
# Author: Johannes Schumann <jschumann@km3net.de>
"""
Runscript for GIBUU/KM3BUU

options:
  -h, --help            show this help message and exit
  --events EVENTS, -n EVENTS
                        Number of events which are simulated
  --multifile MULTIFILE, -m MULTIFILE
                        How many km3net files to write from the dataset
  --seed SEED, -s SEED  Seed which should be used for the (pseudo) random number gen.
  --flavor {electron,muon,tau}, -f {electron,muon,tau}
                        Simulated neutrino flavor
  --interaction {nc,cc,antinc,anticc}, -i {nc,cc,antinc,anticc}
                        The current type of the weak interaction
  --target A Z, -a A Z  The number of nucleons/protons in the target nucleus
  --timesteps TIMESTEPS, -x TIMESTEPS
                        The number of timesteps performed by GiBUU
  --geometry {no,can,sphere,cylindrical}, -g {no,can,sphere,cylindrical}
                        Type of detector enviroment geometry should be used
  --center x y z, -c x y z
                        Center (offset) if a geometry is used (otherwise ignored)
  --dimensions [DIMENSIONS ...], -d [DIMENSIONS ...]
                        Dimensions of the geometry; sphere -> -d <radius> / can -> -d <radius> <zmin> <zmax> / cylindrical -> -d
                        <seawaterheight> <rockheight> <radius> <canradius> <canzmin> <canzmax>
  --output-dir OUTPUT, -o OUTPUT
                        Output directory
  --run RUNNUMBER, -r RUNNUMBER
                        Run number to use
  --timeinterval begin end, -t begin end
                        Unix time interval the events are distributed in [ms]
  --gibuuparams GIBUUPARAMS, -p GIBUUPARAMS
                        JSON file for modified GiBUU namelist params
  --decay, --no-decay   Decay final state particles (according to the decays done in gSeaGen) (default: False)
  --zenith cosZmin cosZmax, -z cosZmin cosZmax
                        Zenith range of the direction if a geometry is used
  --km3net, --no-km3net Write a km3net dataformat file (default: True)

modes:
  {single,range}        Modes
    single              Run in single energy mode
    range               Run in energy range mode

"""

import numpy as np
import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
from os.path import join

from km3buu.jobcard import *
from km3buu.ctrl import run_jobcard
from km3buu.geometry import *
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
    "option_strings": ["--target", "-a"],
    "dest": "target",
    "nargs": 2,
    "type": int,
    "help": "The number of nucleons/protons in the target nucleus",
    "metavar": ("A", "Z"),
    "required": True
}, {
    "option_strings": ["--timesteps", "-x"],
    "dest": "timesteps",
    "type": int,
    "help": "The number of timesteps performed by GiBUU",
    "required": False,
    "default": -1
}, {
    "option_strings": ["--geometry", "-g"],
    "dest":
    "geometry",
    "choices": ["no", "can", "sphere", "cylindrical"],
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
    "Dimensions of the geometry; sphere -> -d <radius> / can -> -d <radius> <zmin> <zmax> / cylindrical -> -d <seawaterheight> <rockheight> <radius> <canzmin> <canzmax> <canradius>"
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
    "option_strings": ["--timeinterval", "-t"],
    "dest": "timeinterval",
    "nargs": 2,
    "type": float,
    "help": "Unix time interval the events are distributed in [s]",
    "metavar": ("begin", "end"),
    "required": True
}, {
    "option_strings": ["--gibuuparams", "-p"],
    "dest":
    "gibuuparams",
    "type":
    Path,
    "help":
    "JSON file for modified GiBUU namelist params"
}, {
    "option_strings": ["--decay"],
    "dest": "decay",
    "action": argparse.BooleanOptionalAction,
    "help":
    "Decay final state particles (according to the decays done in gSeaGen)",
    "default": False
}, {
    "option_strings": ["--km3net"],
    "dest": "km3net",
    "action": argparse.BooleanOptionalAction,
    "help": "Write km3net dataformat file (default: true)",
    "default": True
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
        flux = 1e3 * energies**-args.flux
        fluxfile = join(args.output, "flux.dat")
        np.savetxt(fluxfile, np.c_[energies, flux])

    ensembles, runs = estimate_number_of_ensembles(args.events, args.target)

    jc = generate_neutrino_jobcard(ensembles,
                                   runs,
                                   args.interaction,
                                   args.flavor,
                                   energy,
                                   args.target,
                                   seed=args.seed,
                                   fluxfile=fluxfile,
                                   do_decay=args.decay,
                                   timesteps=args.timesteps)

    # FinalEvents.dat develop option
    jc["neutrinoanalysis"]["outputEvents"] = False
    jc["neutrinoanalysis"]["applyCuts"] = 2

    if args.gibuuparams:
        with open(args.gibuuparams) as f:
            additional_args = json.load(f)

        for k1, param_dct in additional_args.items():
            for k2, v in param_dct.items():
                jc[k1][k2] = v

    run_jobcard(jc, gibuu_dir)

    if not args.km3net:
        return

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
            kwargs["zmin"] = args.dimensions[0]
            kwargs["zmax"] = args.dimensions[1]
            kwargs["radius"] = args.dimensions[2]
        volume = CANVolume(**kwargs)
    elif args.geometry == 'cylindrical':
        kwargs = {"detector_center": tuple(args.center), "zenith": args.zenith}
        kwargs["sw_height"] = args.dimensions[0]
        kwargs["sr_height"] = args.dimensions[1]
        kwargs["radius"] = args.dimensions[2]
        kwargs["can_zmin"] = args.dimensions[3]
        kwargs["can_zmax"] = args.dimensions[4]
        kwargs["can_radius"] = args.dimensions[5]
        volume = CylindricalVolume(**kwargs)
    run_descriptor = ""
    if args.runnumber:
        run_descriptor = "run{:08d}_".format(args.runnumber)

    outfilename = join(args.output,
                       "km3buu_" + run_descriptor + descriptor + ".root")

    write_detector_file(fobj,
                        geometry=volume,
                        ofile=outfilename,
                        run_number=args.runnumber,
                        timeinterval=(args.timeinterval[0],
                                      args.timeinterval[1]),
                        no_files=args.multifile)


if __name__ == '__main__':
    main()
