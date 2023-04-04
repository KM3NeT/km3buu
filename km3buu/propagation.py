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

M_TO_CM = 1e2

itp_table_path = Config().proposal_itp_tables
pathlib.Path(itp_table_path).mkdir(parents=True, exist_ok=True)
pp.InterpolationSettings.tables_path = itp_table_path

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

PROPOSAL_TARGET_WATER = pp.medium.AntaresWater()
PROPOSAL_TARGET_ROCK = pp.medium.StandardRock()

PROPOSAL_STOP_HIERARCHY_LEVEL = 0
PROPOSAL_PROPAGATION_HIERARCHY_LEVEL = 4
PROPOSAL_CAN_HIERARCHY_LEVEL = 2

PROPOSAL_MUON_STOP_CONDITION = 3
PROPOSAL_TAU_STOP_CONDITION = 1


def _setup_utility(particle_definition, target):
    crosssection = pp.crosssection.make_std_crosssection(
        particle_def=particle_definition,
        target=target,
        interpolate=True,
        cuts=pp.EnergyCutSettings(np.inf, 0.05, False))
    collection = pp.PropagationUtilityCollection()
    collection.displacement = pp.make_displacement(crosssection, True)
    collection.interaction = pp.make_interaction(crosssection, True)
    collection.decay = pp.make_decay(crosssection, particle_definition, True)
    collection.time = pp.make_time(crosssection, particle_definition, True)
    return pp.PropagationUtility(collection=collection)


def setup_propagator(particle_definition, proposal_geometries):
    utility_sw = _setup_utility(particle_definition, PROPOSAL_TARGET_WATER)
    utility_sr = _setup_utility(
        particle_definition,
        PROPOSAL_TARGET_ROCK) if "sr" in proposal_geometries.keys() else None

    density_sw = pp.density_distribution.density_homogeneous(
        PROPOSAL_TARGET_WATER.mass_density)
    density_sr = pp.density_distribution.density_homogeneous(
        PROPOSAL_TARGET_ROCK.mass_density)

    sectors = [(geometry, utility_sw, density_sw)
               for k, geometry in proposal_geometries.items() if k != "sr"]
    if "sr" in proposal_geometries.keys():
        sectors.append((proposal_geometries["sr"], utility_sr, density_sr))

    propagator = pp.Propagator(particle_definition, sectors)

    return propagator


class Propagator(object):

    def __init__(self, pdgids, geometry):
        self._geometry = geometry
        self._pdgids = pdgids
        self._pp_propagators = dict()
        self._setup()

    def _setup(self):
        for p in self._pdgids:
            self._pp_propagators[p] = setup_propagator(
                PROPOSAL_LEPTON_DEFINITIONS[p](), self._geometry)

    # @geometry.setter
    # def geometry(self, geodct):
    #     self._geometry = geodct
    #     # Update propagators
    #     self._setup()

    @staticmethod
    def _addparticles(dct, particle_infos):
        for prtcl in particle_infos:
            dct['barcode'].append(prtcl.type)
            dct['E'].append(prtcl.energy / 1e3)
            dct['x'].append(prtcl.position.x / M_TO_CM)
            dct['y'].append(prtcl.position.y / M_TO_CM)
            dct['z'].append(prtcl.position.z / M_TO_CM)
            dct['Px'].append(prtcl.direction.x * prtcl.momentum / 1e3)
            dct['Py'].append(prtcl.direction.y * prtcl.momentum / 1e3)
            dct['Pz'].append(prtcl.direction.z * prtcl.momentum / 1e3)
            dct['deltaT'].append(prtcl.time * 1e-9)

    def propagate_particle(self, prtcl_state):
        stop_condition = PROPOSAL_MUON_STOP_CONDITION if abs(
            prtcl_state.type) == 13 else PROPOSAL_TAU_STOP_CONDITION
        return self._pp_propagators[prtcl_state.type].propagate(
            prtcl_state, hierarchy_condition=stop_condition)

    def propagate(self, lep_pdgid, lep_E, lep_pos, lep_dir):
        """
        Lepton propagation to can based on PROPOSAL

        Parameters
        ----------
        vtx_pos: np.array
            Vertex positions of the given events
        lep_E: np.array
            Outgoing lepton energy
        lep_dir: np.array 
            Lepton direction positions of the given events
        pdgid:
            The pdgid of the propagated lepton

        Returns
        -------
        awkward.highlevel.Array (barcode, deltaT, E, Px, Py, Pz, x, y, z)
        """
        if not lep_pdgid in self._pdgids:
            return None

        init_state = pp.particle.ParticleState()
        init_state.type = lep_pdgid
        init_state.energy = lep_E * 1e3
        init_state.direction = pp.Cartesian3D(lep_dir)
        init_state.direction.normalize()
        init_state.position = pp.Cartesian3D(*(lep_pos * M_TO_CM))

        particle_dct = defaultdict(list)

        track = self.propagate_particle(init_state)
        fsi = track.final_state()
        decay = track.decay_products()

        if self._geometry['can'].is_inside(fsi.position, fsi.direction):
            if len(decay) > 0:
                self._addparticles(particle_dct, decay)
            else:
                self._addparticles(particle_dct, [fsi])
        else:
            for d in decay:
                if d.type in self._pp_propagators.keys():
                    track = self.propagate_particle(d)
                    fsi = track.final_state()
                    decay = track.decay_products()
                    if self._geometry['can'].is_inside(fsi.position,
                                                       fsi.direction):
                        if len(decay) > 0:
                            self._addparticles(particle_dct, decay)
                        else:
                            self._addparticles(particle_dct, [fsi])
        if len(particle_dct["E"]) == 0:
            return None
        return ak.Array(particle_dct)
