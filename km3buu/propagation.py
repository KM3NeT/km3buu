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


def _setup_propagator(max_distance, lepton_definition):
    sector = pp.SectorDefinition()
    sector.medium = pp.medium.AntaresWater(1.0)
    geometry = pp.geometry.Sphere(pp.Vector3D(), max_distance, 0)
    sector.geometry = geometry
    sector.scattering_model = pp.scattering.ScatteringModel.Highland

    cfg = Config()
    interpolation = pp.InterpolationDef()
    interpolation.path_to_tables = cfg.proposal_itp_tables
    interpolation.path_to_tables_readonly = cfg.proposal_itp_tables

    return pp.Propagator(sector_defs=[sector],
                         particle_def=lepton_definition,
                         detector=geometry,
                         interpolation_def=interpolation)


def propagate_lepton(lepout_data, pdgid):
    """
    Lepton propagation based on PROPOSAL

    Parameters
    ----------
    lepout_data: awkward1.highlevel.Array
        Lepton data in the GiBUU output shape containing the fields 
        'lepOut_E, lepOut_Px, lepOut_Py, lepOut_Pz'
    pdgid:
        The pdgid of the propagated lepton

    Returns
    -------
    awkward1.highlevel.Array (E, Px, Py, Pz, x, y, z)
    """
    lepton_info = Particle.from_pdgid(pdgid)
    prop_range = const.c * lepton_info.lifetime * 1e11 * np.max(
        lepout_data.lepOut_E) / lepton_info.mass  #[cm]

    lepton_def = PROPOSAL_LEPTON_DEFINITIONS[pdgid]()
    lepton = pp.particle.DynamicData(lepton_def.particle_type)
    propagated_data = defaultdict(list)

    propagator = _setup_propagator(10 * prop_range, lepton_def)

    for entry in lepout_data:
        lepton.energy = entry.lepOut_E * 1e3
        lepton.position = pp.Vector3D(0, 0, 0)
        lepton.direction = pp.Vector3D(entry.lepOut_Px, entry.lepOut_Py,
                                       entry.lepOut_Pz)
        lepton.direction.normalize()
        secondaries = propagator.propagate(lepton, 5 * prop_range)

        E = np.array(secondaries.energy) / 1e3
        pdgid = [p.type for p in secondaries.particles]
        Px = [p.direction.x * p.momentum / 1e3 for p in secondaries.particles]
        Py = [p.direction.y * p.momentum / 1e3 for p in secondaries.particles]
        Pz = [p.direction.z * p.momentum / 1e3 for p in secondaries.particles]
        x = [p.position.x / 100 for p in secondaries.particles]
        y = [p.position.y / 100 for p in secondaries.particles]
        z = [p.position.z / 100 for p in secondaries.particles]
        dt = [p.time for p in secondaries.particles]

        propagated_data["E"].append(E)
        propagated_data["Px"].append(Px)
        propagated_data["Py"].append(Py)
        propagated_data["Pz"].append(Pz)
        propagated_data["barcode"].append(pdgid)
        propagated_data["x"].append(x)
        propagated_data["y"].append(y)
        propagated_data["z"].append(z)
        propagated_data["deltaT"].append(dt)

    return ak.Array(propagated_data)
