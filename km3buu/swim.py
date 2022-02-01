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
import warnings
try:
    import ROOT
except:
    warnings.warn("ROOT or PyROOT not available")

ENERGY_RANGE = (0.1, 100)
ENERGY_BINS = 40
BJORKENY_BINS = 20


def build_swim_xsec_label(flavor, process, anti, loglabel=False):
    f = flavor[0].lower()
    p = process.upper()
    a = "b" if anti else "u"

ENERGY_RANGE = (0.1, 100)
ENERGY_BINS = 40
BJORKENY_BINS = 20


def build_swim_xsec_label(flavor, process, anti, loglabel=False):
    f = flavor[0].lower()
    p = process.upper()
    a = "b" if anti else "u"
    l = "log" if loglabel else ""
    return "gn{}{}_{}_{}E".format(a, f, p, l)


def build_swim_by_label(flavor, process, anti):
    flavor_map = {"electron": "e", "muon": "mu", "tau": "tau"}
    f = flavor_map[flavor]
    p = process.lower()
    a = "a" if anti else ""
    return "h2d_bydist_E_{}nu{}_{}".format(a, f, p)


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
    gr.Write(label, ROOT.TObject.kOverwrite)
    f.Close()


def write_swim_bjorkeny_file(energy, by, label, filename="nu_bjorken_y.root"):
    f = ROOT.TFile.Open(filename, "UPDATE")
    f.mkdir("single_graphs")
    f.Cd("single_graphs")
    hist = ROOT.TH2D(label, label, ENERGY_BINS, ENERGY_RANGE[0],
                     ENERGY_RANGE[1], BJORKENY_BINS, 0, 1)
    z, x, y = np.histogram2d(energy,
                             by,
                             bins=(np.linspace(ENERGY_RANGE[0],
                                               ENERGY_RANGE[1], ENERGY_BINS),
                                   np.linspace(0, 1, BJORKENY_BINS)),
                             density=True)
    for idxE, idxBy in np.ndindex(z.shape):
        hist.SetBinContent(idxE, idxBy, z[idxE, idxBy])

    hist.Write(label, ROOT.TObject.kOverwrite)
    f.Close()
