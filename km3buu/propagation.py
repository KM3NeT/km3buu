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
import awkward as ak
from collections import defaultdict
import pathlib

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

PROPOSAL_TARGET = pp.medium.AntaresWater()


def _setup_propagator(max_distance, particle_definition):
    itp_table_path = Config().proposal_itp_tables
    pathlib.Path(itp_table_path).mkdir(parents=True, exist_ok=True)
    pp.InterpolationSettings.tables_path = itp_table_path
    crosssection = pp.crosssection.make_std_crosssection(
        particle_def=particle_definition,
        target=PROPOSAL_TARGET,
        interpolate=True,
        cuts=pp.EnergyCutSettings(np.inf, 0.05, False))
    collection = pp.PropagationUtilityCollection()
    collection.displacement = pp.make_displacement(crosssection, True)
    collection.interaction = pp.make_interaction(crosssection, True)
    collection.decay = pp.make_decay(crosssection, particle_definition, True)
    collection.time = pp.make_time(crosssection, particle_definition, True)

    utility = pp.PropagationUtility(collection=collection)

    geometry = pp.geometry.Sphere(pp.Cartesian3D(), max_distance)
    density_distr = pp.density_distribution.density_homogeneous(
        PROPOSAL_TARGET.mass_density)

    propagator = pp.Propagator(particle_definition,
                               [(geometry, utility, density_distr)])

    return propagator


def propagate_lepton(lepout_data, pdgid):
    """
    Lepton propagation based on PROPOSAL

    Parameters
    ----------
    lepout_data: awkward.highlevel.Array
        Lepton data in the GiBUU output shape containing the fields 
        'lepOut_E, lepOut_Px, lepOut_Py, lepOut_Pz'
    pdgid:
        The pdgid of the propagated lepton

    Returns
    -------
    awkward.highlevel.Array (E, Px, Py, Pz, x, y, z)
    """
    lepton_info = Particle.from_pdgid(pdgid)
    prop_range = const.c * lepton_info.lifetime * 1e11 * np.max(
        lepout_data.lepOut_E) / lepton_info.mass  #[cm]

    lepton_def = PROPOSAL_LEPTON_DEFINITIONS[pdgid]()

    propagator = _setup_propagator(10 * prop_range, lepton_def)

    propagated_data = defaultdict(list)

    for entry in lepout_data:
        init_state = pp.particle.ParticleState()
        init_state.energy = entry.lepOut_E * 1e3
        init_state.position = pp.Cartesian3D(0, 0, 0)
        init_state.direction = pp.Cartesian3D(entry.lepOut_Px, entry.lepOut_Py,
                                              entry.lepOut_Pz)
        init_state.direction.normalize()
        track = propagator.propagate(init_state, 5 * prop_range)
        secondaries = track.decay_products()

        E = [s.energy / 1e3 for s in secondaries]
        pdgid = [p.type for p in secondaries]
        Px = [p.direction.x * p.momentum / 1e3 for p in secondaries]
        Py = [p.direction.y * p.momentum / 1e3 for p in secondaries]
        Pz = [p.direction.z * p.momentum / 1e3 for p in secondaries]
        x = [p.position.x / 100 for p in secondaries]
        y = [p.position.y / 100 for p in secondaries]
        z = [p.position.z / 100 for p in secondaries]
        dt = [p.time for p in secondaries]

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
