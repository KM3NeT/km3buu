# Filename: physics.py
"""
Additional physics functionality

Visible energy:
* Implementation applied from JPP
* Theory by M. Dentler https://inspirehep.net/literature/1321036
"""

__author__ = "Johannes Schumann"
__copyright__ = "Copyright 2021, Johannes Schumann and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Johannes Schumann"
__email__ = "jschumann@km3net.de"
__status__ = "Development"

import numpy as np

ELEC_PARAMS = {
    "ELECa": 1.33356e5,
    "ELECb": 1.66113e2,
    "ELECc": 16.4949,
    "ELECd": 1.5385e5,
    "ELECe": 6.04871e5,
}

PION_PARAMS = {
    "PIa": 0.538346,
    "PIb": 1.32041,
    "PIc": 0.737415,
    "PId": -0.813861,
    "PIe": -2.22444,
    "PIf": -2.15795,
    "PIg": -6.47242,
    "PIh": -2.7567,
    "PIx": 8.83498,
}

KAON_PARAMS = {
    "Ka": 12.7537,
    "Kb": -1.052,
    "Kc": 3.49559,
    "Kd": 16.7914,
    "Ke": -0.355066,
    "Kf": 2.77116,
}

KSHORT_PARAMS = {
    "K0sa": 0.351242,
    "K0sb": 0.613076,
    "K0sc": 6.24682,
    "K0sd": 2.8858,
}

KLONG_PARAMS = {
    "K0la": 2.18152,
    "K0lb": -0.632798,
    "K0lc": 0.999514,
    "K0ld": 1.36052,
    "K0le": 4.22484,
    "K0lf": -0.307859,
}

PROTON_PARAMS = {
    "Pa": 12.1281,
    "Pb": -17.1528,
    "Pc": 0.573158,
    "Pd": 34.1436,
    "Pe": -0.28944,
    "Pf": -13.2619,
    "Pg": 24.1357,
}

NEUTRON_PARAMS = {
    "Na": 1.24605,
    "Nb": 0.63819,
    "Nc": -0.802822,
    "Nd": -0.935327,
    "Ne": -6.1126,
    "Nf": -1.96894,
    "Ng": 0.716954,
    "Nh": 2.68246,
    "Ni": -9.39464,
    "Nj": 15.0737,
}

HE_PARAMS = {
    "Ma": 72.425,
    "Mb": -49.417,
    "Mc": 5.858,
    "Md": 207.252,
    "Me": 132.784,
    "Mf": -10.277,
    "Mg": -19.441,
    "Mh": 58.598,
    "Mi": 53.161,
    "Mkref": 2.698,
}


def visible_energy_fraction(pdgid, energy):
    """
    Returns the visible energy fraction in the one particle approximation (OPA)

    Parameters
    ----------
    pdgid: int
        PDGID of the given particle
    energy: float
        Kinetic energy of the given particle
    """

    # Cover trivial cases, i.e. 'non-visible' particles and ultra-high energy
    if pdgid in [11, -11, 22, 111, 221]:
        return 1.0
    elif energy > 1e7:
        return high_energy_weight(energy)

    tmp = energy if energy < 40. else 40.
    weight = 0.0

    if abs(pdgid) == 211:
        weight = pion_weight(tmp)
    elif pdgid == 130:
        weight = klong_weight(tmp)
    elif pdgid == 310:
        weight = kshort_weight(tmp)
    elif abs(pdgid) == 321:
        weight = kaon_weight(tmp)
    elif pdgid == 2112:
        weight = neutron_weight(tmp)
    elif pdgid == 2122:
        weight = proton_weight(tmp)
    elif pdgid in [12,-12,14,-14,16,-16, -13, 13, 15, -15]:
        weight = 0.0
    else:
        weight = proton_weight(tmp)

    if energy < 40.:
        return weight
    else:
        he_weight = high_energy_weight(energy)
        he40GeV_weight = high_energy_weight(40.)
        return he_weight - (he40GeV_weight -
                            weight) * (7. - np.log10(energy)) / 5.398


def number_photons_per_electron(energy):
    """Expected number of photons for electrons as function of energy"""
    exp_coeff = np.exp(-energy / ELEC_PARAMS["ELECc"])
    n = (ELEC_PARAMS["ELECa"] * energy + ELEC_PARAMS["ELECb"]) * exp_coeff + (
        ELEC_PARAMS["ELECd"] * energy + ELEC_PARAMS["ELECe"]) * (1 - exp_coeff)
    return n


def pion_weight(energy):
    norm = number_photons_per_electron(energy)
    if energy < 6e-2:
        return 1e4 * PION_PARAMS["PIa"] / norm
    elif energy < 1.5e-1:
        return 1e4 * PION_PARAMS["PIx"] * energy / norm
    else:
        logE = np.log(energy)
        return (1e5 * PION_PARAMS["PIb"] * energy +
                (energy**(1. - 1. / PION_PARAMS["PIc"])) *
                (PION_PARAMS["PId"] * 1e4 + 1e4 * PION_PARAMS["PIe"] * logE +
                 1e4 * PION_PARAMS["PIf"] * logE**2 + 1e3 * PION_PARAMS["PIg"]
                 * logE**3 + 1e2 * PION_PARAMS["PIh"] * logE**4)) / norm


def kaon_weight(energy):
    norm = number_photons_per_electron(energy)
    if energy > 0.26:
        exp_coeff = np.exp(-energy / KAON_PARAMS["Kc"])
        return (1e4 * KAON_PARAMS["Ka"] * (energy + KAON_PARAMS["Kb"]) *
                (1. - exp_coeff) + 1e4 *
                (KAON_PARAMS["Kd"] * energy + KAON_PARAMS["Ke"]) *
                exp_coeff) / norm
    else:
        return KAON_PARAMS["Kf"] * 1e4 / norm


def kshort_weight(energy):
    norm = number_photons_per_electron(energy)
    return (KSHORT_PARAMS["K0sa"] * 1e5 + KSHORT_PARAMS["K0sb"] * 1e5 * energy
            + energy * KSHORT_PARAMS["K0sc"] * 1e4 *
            np.log(KSHORT_PARAMS["K0sd"] + 1. / energy)) / norm


def klong_weight(energy):
    norm = number_photons_per_electron(energy)
    if energy < 1.5:
        return (1e4 * KLONG_PARAMS["K0la"] +
                energy * 1e5 * KLONG_PARAMS["K0lb"] * np.log(energy) +
                1e5 * KLONG_PARAMS["K0lc"] * energy**1.5) / norm
    else:
        return (energy * KLONG_PARAMS["K0ld"] * 1e5 +
                energy**(1. - 1. / KLONG_PARAMS["K0le"]) *
                KLONG_PARAMS["K0lf"] * 1e5) / norm


def proton_weight(energy):
    exp_coeff = np.exp(-energy / PROTON_PARAMS["Pc"])
    norm = number_photons_per_electron(energy)
    weight = 1e4 * (PROTON_PARAMS["Pa"] * energy + PROTON_PARAMS["Pb"]) * (
        1 - exp_coeff) + 1e4 * (PROTON_PARAMS["Pd"] * energy +
                                PROTON_PARAMS["Pe"] +
                                PROTON_PARAMS["Pf"] * energy**2 +
                                PROTON_PARAMS["Pg"] * energy**3) * exp_coeff
    return weight / norm


def neutron_weight(energy):
    norm = number_photons_per_electron(energy)
    if energy > 0.5:
        logE = np.log(energy)
        return (NEUTRON_PARAMS["Na"] * 1e5 * energy +
                1e3 * energy**(1. - 1. / NEUTRON_PARAMS["Nb"]) *
                (100 * NEUTRON_PARAMS["Nc"] + 100 * NEUTRON_PARAMS["Nd"] * logE
                 + 10 * NEUTRON_PARAMS["Ne"] * logE**2 +
                 11 * NEUTRON_PARAMS["Nf"] * logE**3)) / norm
    else:
        return (1e3 * NEUTRON_PARAMS["Ng"] + 1e4 * NEUTRON_PARAMS["Nh"] *
                energy + 1e4 * NEUTRON_PARAMS["Ni"] * energy**2 +
                1e4 * NEUTRON_PARAMS["Nj"] * energy**3) / norm


def high_energy_weight(energy):
    """
    High energy weight (valid above 40 GeV)

    Parameters
    ----------
    energy: float 
        Kinetic energy of the given particle
    """
    if energy < 0.2:
        return 0.292

    logE = np.log10(energy)
    mlc = (HE_PARAMS["Ma"] - HE_PARAMS["Mf"]) / HE_PARAMS["Mkref"]
    denom = HE_PARAMS["Mi"] + HE_PARAMS["Mh"] * logE + HE_PARAMS[
        "Mg"] * logE**2 + HE_PARAMS["Mf"] * logE**3 + mlc * logE**4
    if denom <= 0:
        return 0
    num = HE_PARAMS[
        "Me"] + HE_PARAMS["Md"] * logE + HE_PARAMS["Mc"] * logE**2 + HE_PARAMS[
            "Mb"] * logE**3 + HE_PARAMS["Ma"] * logE**4 + mlc * logE**5
    lognp = num / denom
    Ee = 10.**(lognp - HE_PARAMS["Mkref"])
    return Ee / energy
