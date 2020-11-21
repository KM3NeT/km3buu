# Filename: propagation.py
"""
Propagation functions for km3buu

"""

__author__ = "Johannes Schumann"
__copyright__ = "Copyright 2020, Johannes Schumann and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Johannes Schumann"
__email__ = "jschumann@km3net.de"
__status__ = "Development"

import numpy as np
import scipy.constants as const
import proposal as pp
from particle import Particle
import awkward1 as ak
from collections import defaultdict

from .config import Config

PROPOSAL_LEPTON_DEFINITIONS = {
    11: pp.particle.EMinusDef,
    -11: pp.particle.EPlusDef,
    12: pp.particle.NuEDef,
    -12: pp.particle.NuEBarDef,
    13: pp.particle.MuMinusDef,
    -13: pp.particle.MuPlusDef,
    14: pp.particle.NuMuDef,
    -14: pp.particle.NuMuBarDef,
    15: pp.particle.TauMinusDef,
    -15: pp.particle.TauPlusDef,
    16: pp.particle.NuTauDef,
    -16: pp.particle.NuTauBarDef
}


def propagate_lepton(lepout_data, pdgid):
    lepton_info = Particle.from_pdgid(pdgid)
    prop_range = const.c * lepton_info.lifetime * 1e11 * np.max(
        lepout_data.lepOut_E) / lepton_info.mass  #[cm]

    sector = pp.SectorDefinition()
    sector.medium = pp.medium.AntaresWater(1.0)
    geometry = pp.geometry.Sphere(pp.Vector3D(), 10 * prop_range, 0)
    sector.geometry = geometry
    sector.scattering_model = pp.scattering.ScatteringModel.Highland

    cfg = Config()
    interpolation = pp.InterpolationDef()
    interpolation.path_to_tables = cfg.proposal_itp_tables
    interpolation.path_to_tables_readonly = cfg.proposal_itp_tables

    lepton_def = PROPOSAL_LEPTON_DEFINITIONS[pdgid]()
    propagator = pp.Propagator(sector_defs=[sector],
                               particle_def=lepton_def,
                               detector=geometry,
                               interpolation_def=interpolation)

    lepton = pp.particle.DynamicData(lepton_def.particle_type)

    propagated_data = defaultdict(list)

    for entry in lepout_data:
        lepton.energy = entry.lepOut_E * 1e3
        lepton.direction = pp.Vector3D(entry.lepOut_Px, entry.lepOut_Py,
                                       entry.lepOut_Pz)
        lepton.direction.normalize()
        secondaries = propagator.propagate(lepton, 5 * prop_range)

        E = np.array(secondaries.energy) / 1e3
        pdgid = [p.type for p in secondaries.particles]
        Px = [p.direction.x * p.momentum for p in secondaries.particles]
        Py = [p.direction.y * p.momentum for p in secondaries.particles]
        Pz = [p.direction.z * p.momentum for p in secondaries.particles]

        propagated_data["E"].append(E)
        propagated_data["Px"].append(Px)
        propagated_data["Py"].append(Py)
        propagated_data["Pz"].append(Pz)
        propagated_data["barcode"].append(pdgid)
        propagated_data["x"].append(p.position.x)
        propagated_data["y"].append(p.position.y)
        propagated_data["z"].append(p.position.z)

    return ak.Array(propagated_data)
