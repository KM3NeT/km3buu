#!/usr/bin/env python
# coding=utf-8
# Filename: write_swim_xsec.py
# Author: Johannes Schumann <jschumann@km3net.de>
"""
Convert GiBUU output to 

Usage:
    write_swim_xsec.py INPUT_PATHS... [--output-file=OUTPUT]
    write_swim_xsec.py (-h | --help)

Options:
    -h --help           Show this screen.
    INPUT_PATHS         GiBUU data
    -o --output-file    File to write 

"""
import numpy as np
from km3buu.jobcard import FLAVOR_LOOKUP, PROCESS_LOOKUP
from km3buu.output import GiBUUOutput
from km3buu.swim import build_swim_xsec_label, write_swim_xsec_file

ENERGY_POINTS = np.logspace(-1, 2, 100)

FLAVOR_INV_LOOKUP = {v: k for k, v in FLAVOR_LOOKUP.items()}
PROCESS_INV_LOOKUP = {v: k for k, v in PROCESS_LOOKUP.items()}


def main():
    from docopt import docopt
    args = docopt(__doc__)
    ofpath = "crossSection.root"
    if args["--output-file"]:
        ofpath = args["--output-file"]
    for gibuu_path in args["INPUT_PATHS"]:
        fobj = GiBUUOutput(gibuu_path)
        xsec = fobj.mean_xsec(ENERGY_POINTS)
        flavor = FLAVOR_INV_LOOKUP[fobj.jobcard["neutrino_induced"]
                                   ["process_id"]]
        process = PROCESS_INV_LOOKUP[abs(
            fobj.jobcard["neutrino_induced"]["process_id"])]
        anti = np.sign(fobj.jobcard["neutrino_induced"]["process_id"]) > 0
        lbl = build_swim_xsec_label(flavor, process, anti)
        write_swim_xsec_file(ENERGY_POINTS, xsec, lbl, filename=ofpath)
        lbl = build_swim_xsec_label(flavor, process, anti, loglabel=True)
        write_swim_xsec_file(np.log10(ENERGY_POINTS),
                             xsec,
                             lbl,
                             filename=ofpath)

if __name__ == '__main__':
    main()
