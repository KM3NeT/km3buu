#!/usr/bin/env python
# coding=utf-8
# Filename: runner.py
# Author: Johannes Schumann <jschumann@km3net.de>
"""

Usage:
    runner.py fixed --energy=ENERGY --events=EVENTS (--CC|--NC) (--electron|--muon|--tau) --target-a=TARGETA --target-z=TARGETZ (--sphere=RADIUS | --can) [--outdir=OUTDIR] [--km3netfile=OUTFILE]
    runner.py range --energy-min=ENERGYMIN --energy-max=ENERGYMAX --events=EVENTS (--CC|--NC) (--electron|--muon|--tau) --target-a=TARGETA --target-z=TARGETZ (--sphere=RADIUS | --can) [--flux=FLUXFILE] [--outdir=OUTDIR] [--km3netfile=OUTFILE]
    runner.py (-h | --help)

Options:
    -h --help   Show this screen.
    --energy=ENERGY                 Neutrino energy [type: float]                   
    --energy-min=ENERGYMIN          Neutrino energy [type: float]                   
    --energy-max=ENERGYMAX          Neutrino energy [type: float]                   
    --events=EVENTS                 Number of simulated events [type: int]          
    --target-a=TARGETA              Target nucleons [type: int]                     
    --target-z=TARGETZ              Target protons [type: int]                      
    --sphere=RADIUS                 Radius of the sphere volume in metres [type: float]
    --can                           Use CAN with std. dimensions
    --flux=FLUXFILE                 Flux definition [type: path]
    --outdir=OUTDIR                 Output directory [type: path]
    (--CC | --NC)                   Interaction type
    (--electron | --muon | --tau)   Neutrino flavor
"""
from type_docopt import docopt
from pathlib import Path
from tempfile import TemporaryDirectory
from os.path import join

from km3buu.jobcard import generate_neutrino_jobcard
from km3buu.ctrl import run_jobcard
from km3buu.geometry import CanVolume, SphericalVolume
from km3buu.output import GiBUUOutput, write_detector_file


def main():
    args = docopt(__doc__, types={'path': Path})

    events = args["--events"]
    energy = args["--energy"] if args["fixed"] else (args["--energy-min"],
                                                     args["--energy-max"])
    interaction = "CC" if args["--CC"] else "NC"
    flavour = "electron" if args["--electron"] else (
        "muon" if args["--muon"] else "tau")
    target = (args["--target-z"], args["--target-a"])

    jc = generate_neutrino_jobcard(events,
                                   interaction,
                                   flavour,
                                   energy,
                                   target,
                                   fluxfile=args["--flux"])

    outdir = args["--outdir"] if args["--outdir"] else TemporaryDirectory()
    outdirname = outdir if args["--outdir"] else outdir.name

    run_jobcard(jc, outdirname)

    fobj = GiBUUOutput(outdir)

    volume = SphericalVolume(
        args["--sphere"]) if args["--sphere"] else CanVolume()

    if args["fixed"]:
        descriptor = "{0}_{1}_{2}GeV_A{3}Z{4}".format(flavour, interaction,
                                                      energy, target[0],
                                                      target[1])
    else:
        descriptor = "{0}_{1}_{2}-{3}GeV_A{4}Z{5}".format(
            flavour, interaction, energy[0], energy[1], target[0], target[1])

    if args["--km3netfile"]:
        outfilename = args["--km3netfile"]
    else:
        outfilename = "km3buu_" + descriptor + ".root"
        if args["--outdir"]:
            outfilename = join(args["--outdir"], outfilename)

    write_detector_file(fobj, geometry=volume, ofile=outfilename)


if __name__ == '__main__':
    main()
