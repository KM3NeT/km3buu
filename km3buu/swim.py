# Filename: swim.py
"""
Interfacing for SWIM framework

"""

__author__ = "Johannes Schumann"
__copyright__ = "Copyright 2021, Johannes Schumann and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Johannes Schumann"
__email__ = "jschumann@km3net.de"
__status__ = "Development"

import numpy as np
import ROOT


def build_swim_xsec_label(flavor, process, anti, loglabel=False):
    f = flavor[0].upper()
    p = process.upper()
    a = "u" if anti else "b"
    l = "log" if loglabel else ""
    return "gn{}{}_{}_{}E".format(a, f, p, l)


def write_swim_xsec_file(energy,
                         xsection,
                         label,
                         filename="crossSection.root",
                         subdir="single_graphs"):
    f = ROOT.TFile.Open(filename, "UPDATE")
    f.mkdir(subdir)
    f.Cd(subdir)
    gr = ROOT.TGraph(len(energy), energy, xsection)
    gr.SetName(label)
    gr.Write()
    f.Close()
