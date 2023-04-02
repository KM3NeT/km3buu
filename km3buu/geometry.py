# Filename: geometry.py
"""
Detector geometry related functionalities
"""

__author__ = "Johannes Schumann"
__copyright__ = "Copyright 2021, Johannes Schumann and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Johannes Schumann"
__email__ = "jschumann@km3net.de"
__status__ = "Development"

import numpy as np
from abc import ABC, abstractmethod
from collections import defaultdict
from scipy.spatial.transform import Rotation
from particle import Particle

import proposal as pp
from .propagation import *

M_TO_CM = 1e2


class DetectorVolume(ABC):
    """
    Detector geometry class
    """

    def __init__(self):
        self._volume = -1.0
        self._coord_origin = (0., 0., 0.)
        self._solid_angle = 1

    @abstractmethod
    def random_dir(self, n=1):
        """
        Generate a random direction for the interaction

        Parameter
        ---------
        n: int (default: 1)
            Number of vertex positions to sample

        Return
        ------
        tuple [rad, 1] (phi, cos(theta))
        """
        pass

    @abstractmethod
    def random_pos(self, n=1):
        """
        Generate a random position in the detector volume based on a uniform
        event distribution

        Parameter
        ---------
        n: int (default: 1)
            Number of vertex directions to sample

        Returns
        -------
        tuple [m] (x, y, z)
        """
        pass

    def distribute_event(self, *pargs, **kwargs):
        """
        Integrated event distribution method which also handles propagation

        Returns
        -------
        vtx_pos: tuple
            Position of the vertex in the interaction volume
        vtx_dir: tuple
            Direction of the vertex in the interaction volume
        weight_correction: float
            If the sampling requires a correction to the weight this factor is
            unequal 1
        additional_events: awkward.highlevel.Array (
            E, Px, Py, Pz, x, y, z, pdgid)
            Propagated / Additional Particles from decays (None if no propagation is done)
        """
        return self.random_pos(), self.random_dir(), 1.0, None

    @abstractmethod
    def header_entries(self):
        """
        Returns the header information for the detector volume geometry

        Returns
        -------
        dict {header_key: header_information}
        """
        pass

    @property
    def solid_angle(self):
        """
        Solid angle used for the direction sampling

        Returns
        -------
        float [1]
        """
        return self._solid_angle

    @property
    def volume(self):
        """
        Detector volume

        Returns
        -------
        float [m^3]
        """
        return self._volume

    @property
    def coord_origin(self):
        """
        Coordinate origin

        Returns
        -------
        tuple [m] (x, y, z)
        """
        return self._coord_origin


class NoVolume(DetectorVolume):
    """
    Dummy volume to write out data w/o geometry
    """

    def __init__(self):
        self._solid_angle = 1
        self._volume = 1
        self._coord_origin = (.0, .0, .0)

    def header_entries(self, nevents=0):
        retdct = dict()
        key = "genvol"
        value = "0 0 0 0 {}".format(nevents)
        retdct[key] = value
        return retdct

    def random_pos(self, n=1):
        if n == 1:
            return np.zeros(3)
        else:
            return np.zeros((n, 3))

    def random_dir(self, n=1):
        if n == 1:
            return (0, 1)
        else:
            return np.concatenate([np.zeros(n), np.ones(n)]).reshape((2, -1)).T

    def distribute_event(self, evt):
        vtx_pos = self.random_pos()
        vtx_dir = self.random_dir()
        weight = 1
        evts = None
        return vtx_pos, vtx_dir, weight, None


class CANVolume(DetectorVolume):
    """
    Cylindrical detector geometry, only CAN / no propagation

    Parameters
    ----------
    radius: float [m] (default: 403.5)
        Cylinder radius given in metres
    zmin: float [m] (default: 0.0)
        Cylinder bottom z position
    zmax: float [m] (default: 476.5)
        Cylinder top z position
    detector_center: tuple [m] (default: (0.0, 0.0) )
        Detector center position in the xy-plane
    zenith: float [1] (default: (-1.0, 1.0) )
        Zenith range given as cos(θ)
    taupropagation: bool (default True)
        Do secondary tau lepton propagation / decay
    """

    def __init__(self,
                 radius=403.4,
                 zmin=0.0,
                 zmax=476.5,
                 detector_center=(0., 0.),
                 zenith=(-1, 1),
                 taupropagation=True):
        super().__init__()
        self._radius = radius
        self._zmin = zmin
        self._zmax = zmax
        self._volume = self._calc_volume()
        self._detector_center = detector_center
        self._cosZmin = zenith[0]
        self._cosZmax = zenith[1]
        self._solid_angle = 2 * np.pi * (self._cosZmax - self._cosZmin)
        self._propagator = None
        if taupropagation:
            self._pp_geometry = self.make_proposal_geometries()
            self._propagator = Propagator([15, -15], self._pp_geometry)

    def make_proposal_geometries(self):
        """
        Setup the geometries for the propagation using PROPOSAL
        """
        geometries = dict()
        # General
        center_x = self._detector_center[0] * M_TO_CM
        center_y = self._detector_center[1] * M_TO_CM
        # StopVolume
        geometry_stop = pp.geometry.Sphere(pp.Cartesian3D(), 1e20)
        geometry_stop.hierarchy = PROPOSAL_STOP_HIERARCHY_LEVEL
        geometries["stop"] = geometry_stop
        # CAN
        can_zpos = (self._zmin + self._zmax) / 2 * M_TO_CM
        geometry_can = pp.geometry.Cylinder(
            pp.Cartesian3D(center_x, center_y, can_zpos),
            (self._zmax - self._zmin) * M_TO_CM, self._radius * M_TO_CM, 0.)
        geometry_can.hierarchy = PROPOSAL_CAN_HIERARCHY_LEVEL
        geometries["can"] = geometry_can
        return geometries

    def _calc_volume(self):
        return np.pi * (self._zmax - self._zmin) * np.power(self._radius, 2)

    def random_pos(self, n=1):
        r = self._radius * np.sqrt(np.random.uniform(0, 1, n))
        phi = np.random.uniform(0, 2 * np.pi, n)
        pos_x = r * np.cos(phi) + self._detector_center[0]
        pos_y = r * np.sin(phi) + self._detector_center[1]
        pos_z = np.random.uniform(self._zmin, self._zmax, n)
        pos = np.concatenate([pos_x, pos_y, pos_z]).reshape((3, -1)).T
        if pos.shape[0] == 1:
            return pos[0, :]
        else:
            return pos

    def in_can(self, pos):
        """
        Check if position is inside the CAN

        Parameters
        ----------
        pos: np.array
            The positions which should be checked

        Return
        ------
        boolean / np.array
        """
        if type(pos) is tuple or pos.ndim == 1:
            pos = np.reshape(pos, (-1, 3))
        zmask = (pos[:, 2] >= self._zmin) & (pos[:, 2] <= self._zmax)
        r2 = (pos[:, 0] - self._coord_origin[0])**2 + \
            (pos[:, 1] - self._coord_origin[1])**2
        rmask = r2 < (self._radius**2)
        mask = zmask & rmask
        if len(mask) == 1:
            return mask[0]
        else:
            return mask

    def random_dir(self, n=1):
        phi = np.random.uniform(0, 2 * np.pi, n)
        cos_theta = np.random.uniform(self._cosZmin, self._cosZmax, n)
        direction = np.concatenate([phi, cos_theta]).reshape((2, -1)).T
        if direction.shape[0] == 1:
            return direction[0, :]
        else:
            return direction

    def header_entries(self, nevents=0):
        retdct = dict()
        key = "genvol"
        value = "{} {} {} {} {}".format(self._zmin, self._zmax, self._radius,
                                        self._volume, nevents)
        retdct[key] = value
        key = "fixedcan"
        value = "{} {} {} {} {}".format(self._detector_center[0],
                                        self._detector_center[1], self._zmin,
                                        self._zmax, self._radius)
        return retdct

    def distribute_event(self, evt):
        vtx_pos = self.random_pos()
        vtx_angles = self.random_dir()
        weight = 1
        evts = None

        if evt.flavor_ID == 3 and abs(evt.process_ID) == 2:
            charged_lepton_type = np.sign(
                evt.process_ID) * (2 * evt.flavor_ID + 9)
            lepout_dir = np.array(
                [evt.lepOut_Px, evt.lepOut_Py, evt.lepOut_Pz])
            R = Rotation.from_euler("yz", vtx_angles)
            evts = self._propagator.propagate(charged_lepton_type,
                                              evt.lepOut_E, vtx_pos,
                                              R.apply(lepout_dir))

        return vtx_pos, vtx_angles, weight, evts


class CylindricalVolume(DetectorVolume):
    """
    Cylindrical detector geometry

    Parameters
    ----------
    radius: float [m] (default: 403.5)
        Radius of the interaction volume
    sw_height: float [m] (default: xxx)
        Height of the seawater in the interaction volume
    sr_height: float [m] (default: xxx)
        Height of the seabottom rock in the interaction volume
    can_radius: float [m] (default: 0.0)
        Cylinder bottom z position
    can_zmin: float [m] (default: 0.0)
        Cylinder bottom z position
    can_zmax: float [m] (default: 476.5)
        Cylinder top z position
    detector_center: tuple [m] (default: (0.0, 0.0) )
        Detector center position in the xy-plane
    zenith: float [1] (default: (-1.0, 1.0) )
        Zenith range given as cos(θ)
    """

    def __init__(self,
                 radius=500.0,
                 sw_height=650.0,
                 sr_height=100.0,
                 can_radius=200.4,
                 can_zmin=0.0,
                 can_zmax=350,
                 detector_center=(0., 0.),
                 zenith=(-1, 1)):
        super().__init__()
        self._detector_center = detector_center
        # Interaction Volume
        self._radius = radius
        self._water_height = sw_height
        self._rock_height = sr_height
        # Can Volume
        self._canradius = can_radius
        self._canzmin = can_zmin
        self._canzmax = can_zmax
        # Direction
        self._cosZmin = zenith[0]
        self._cosZmax = zenith[1]
        # Properties
        self._solid_angle = 2 * np.pi * (self._cosZmax - self._cosZmin)
        self._volume = self._calc_volume()
        # PROPOSAL
        self._pp_geometry = None
        self._propagator = None

    def _calc_volume(self):
        return np.pi * (self._water_height + self._rock_height) * np.power(
            self._radius, 2)

    def random_pos(self, n=1):
        r = self._radius * np.sqrt(np.random.uniform(0, 1, n))
        phi = np.random.uniform(0, 2 * np.pi, n)
        pos_x = r * np.cos(phi) + self._detector_center[0]
        pos_y = r * np.sin(phi) + self._detector_center[1]
        pos_z = np.random.uniform(-self._rock_height, self._water_height, n)
        pos = np.concatenate([pos_x, pos_y, pos_z]).reshape((3, -1)).T
        if pos.shape[0] == 1:
            return pos[0, :]
        else:
            return pos

    def in_can(self, pos):
        """
        Check if position is inside the CAN

        Parameters
        ----------
        pos: np.array
            The positions which should be checked

        Return
        ------
        boolean / np.array
        """
        if type(pos) is tuple or pos.ndim == 1:
            pos = np.reshape(pos, (-1, 3))
        zmask = (pos[:, 2] >= self._canzmin) & (pos[:, 2] <= self._canzmax)
        r2 = (pos[:, 0] - self._coord_origin[0])**2 + \
            (pos[:, 1] - self._coord_origin[1])**2
        rmask = r2 < (self._canradius**2)
        mask = zmask & rmask
        if len(mask) == 1:
            return mask[0]
        else:
            return mask

    def random_dir(self, n=1):
        phi = np.random.uniform(0, 2 * np.pi, n)
        cos_theta = np.random.uniform(self._cosZmin, self._cosZmax, n)
        direction = np.concatenate([phi, cos_theta]).reshape((2, -1)).T
        if direction.shape[0] == 1:
            return direction[0, :]
        else:
            return direction

    def header_entries(self, nevents=0):
        retdct = dict()
        key = "genvol"
        value = "{} {} {} {} {}".format(-self._rock_height, self._water_height,
                                        self._radius, self._volume, nevents)
        retdct[key] = value
        key = "fixedcan"
        value = "{} {} {} {} {}".format(self._detector_center[0],
                                        self._detector_center[1],
                                        self._canzmin, self._canzmax,
                                        self._canradius)
        return retdct

    def make_proposal_geometries(self):
        """
        Setup the geometries for the propagation using PROPOSAL
        """
        geometries = dict()
        # General
        center_x = self._detector_center[0] * M_TO_CM
        center_y = self._detector_center[1] * M_TO_CM
        # StopVolume
        geometry_stop = pp.geometry.Sphere(pp.Cartesian3D(), 1e20)
        geometry_stop.hierarchy = PROPOSAL_STOP_HIERARCHY_LEVEL
        geometries["stop"] = geometry_stop
        # CAN
        can_zpos = (self._canzmin + self._canzmax) / 2 * M_TO_CM
        geometry_can = pp.geometry.Cylinder(
            pp.Cartesian3D(center_x, center_y, can_zpos),
            (self._canzmax - self._canzmin) * M_TO_CM,
            self._canradius * M_TO_CM, 0.)
        geometry_can.hierarchy = PROPOSAL_CAN_HIERARCHY_LEVEL
        geometries["can"] = geometry_can
        # Interaction Volume
        geometry_mantle = pp.geometry.Cylinder(
            pp.Cartesian3D(center_x, center_y, can_zpos),
            (self._canzmax - self._canzmin) * M_TO_CM, self._radius * M_TO_CM,
            self._canradius * M_TO_CM)
        geometry_mantle.hierarchy = PROPOSAL_PROPAGATION_HIERARCHY_LEVEL
        geometries["can_mantle"] = geometry_mantle
        if self._canzmin > 0:
            floor_zpos = self._canzmin / 2 * M_TO_CM
            geometry_floor = pp.geometry.Cylinder(
                pp.Cartesian3D(center_x, center_y, floor_zpos),
                (self._canzmax - self._canzmin) * M_TO_CM,
                self._radius * M_TO_CM, 0.)
            geometry_floor.hierarchy = PROPOSAL_PROPAGATION_HIERARCHY_LEVEL
            geometries["can_floor"] = geometry_floor
        if self._water_height > self._canzmax:
            ceil_zpos = (self._water_height + self._canzmax) / 2 * M_TO_CM
            geometry_ceil = pp.geometry.Cylinder(
                pp.Cartesian3D(center_x, center_y, ceil_zpos),
                (self._water_height - self._canzmax) * M_TO_CM,
                self._radius * M_TO_CM, 0.)
            geometry_ceil.hierarchy = PROPOSAL_PROPAGATION_HIERARCHY_LEVEL
            geometries["can_ceiling"] = geometry_ceil
        if self._rock_height > 0.:
            sr_zpos = -self._rock_height / 2 * M_TO_CM
            geometry_sr = pp.geometry.Cylinder(
                pp.Cartesian3D(center_x, center_y, sr_zpos),
                self._rock_height * M_TO_CM, self._radius * M_TO_CM, 0)
            geometry_sr.hierarchy = PROPOSAL_PROPAGATION_HIERARCHY_LEVEL
            geometries["sr"] = geometry_sr
        return geometries

    @staticmethod
    def _addparticles(dct, particle_infos):
        for prtcl in particle_infos:
            dct['barcode'].append(prtcl.type)
            dct['E'].append(prtcl.energy)
            dct['x'].append(prtcl.position.x / 100)
            dct['y'].append(prtcl.position.y / 100)
            dct['z'].append(prtcl.position.z / 100)
            dct['Px'].append(prtcl.direction.x * prtcl.momentum / 1e3)
            dct['Py'].append(prtcl.direction.y * prtcl.momentum / 1e3)
            dct['Pz'].append(prtcl.direction.z * prtcl.momentum / 1e3)
            dct['deltaT'].append(prtcl.time)

    def distribute_event(self, evt):
        # NC -> doesn't require any propagation
        if abs(evt.process_ID) == 3 or evt.flavor_ID == 1:
            vtx_pos = self.random_pos()
            vtx_dir = self.random_dir()
            weight = 1
            evts = None
            return vtx_pos, vtx_dir, weight, evts

        if not self._pp_geometry:
            self._pp_geometry = self.make_proposal_geometries()

        if not self._propagator:
            self._propagator = Propagator([13, -13, 15, -15],
                                          self._pp_geometry)

        charged_lepton_type = np.sign(evt.process_ID) * (2 * evt.flavor_ID + 9)

        samples = 0
        lepout_dir = np.array([evt.lepOut_Px, evt.lepOut_Py, evt.lepOut_Pz])

        while True:
            samples += 1
            vtx_pos = self.random_pos()
            vtx_angles = self.random_dir()
            if self.in_can(vtx_pos) and evt.flavor_ID == 2:
                return vtx_pos, vtx_angles, samples, None
            R = Rotation.from_euler("yz", vtx_angles)
            particles = self._propagator.propagate(charged_lepton_type,
                                                   evt.lepOut_E, vtx_pos,
                                                   R.apply(lepout_dir))
            if not particles is None:
                return vtx_pos, vtx_angles, samples, particles


class SphericalVolume(DetectorVolume):
    """
    Spherical detector geometry

    Parameters
    ----------
    radius: float [m]
        The radius of the sphere
    center: tuple [m]
        Coordinate center of the sphere
        (x, y, z)
    zenith: float [1] (default: (-1.0, 1.0) )
        Zenith range given as cos(θ)
    """

    def __init__(self, radius, coord_origin=(0, 0, 0), zenith=(-1, 1)):
        super().__init__()
        self._radius = radius
        self._coord_origin = coord_origin
        self._volume = self._calc_volume()
        self._cosZmin = zenith[0]
        self._cosZmax = zenith[1]
        self._solid_angle = 2 * np.pi * (self._cosZmax - self._cosZmin)

    def _calc_volume(self):
        return 4 / 3 * np.pi * np.power(self._radius, 3)

    def random_pos(self, n=1):
        r = self._radius * np.power(np.random.random(n), 1 / 3)
        phi = np.random.uniform(0, 2 * np.pi, n)
        cosTheta = np.random.uniform(-1, 1, n)
        pos_x = r * np.cos(phi) * np.sqrt(
            1 - np.power(cosTheta, 2)) + self._coord_origin[0]
        pos_y = r * np.sin(phi) * np.sqrt(
            1 - np.power(cosTheta, 2)) + self._coord_origin[1]
        pos_z = r * cosTheta + self._coord_origin[2]
        pos = np.concatenate([pos_x, pos_y, pos_z]).reshape((3, -1)).T
        if pos.shape[0] == 1:
            return pos[0, :]
        else:
            return pos

    def random_dir(self, n=1):
        phi = np.random.uniform(0, 2 * np.pi, n)
        cos_theta = np.random.uniform(self._cosZmin, self._cosZmax, n)
        direction = np.concatenate([phi, cos_theta]).reshape((2, -1)).T
        if direction.shape[0] == 1:
            return direction[0, :]
        else:
            return direction

    def header_entries(self, nevents=0):
        retdct = dict()
        key = "sphere"
        value = "radius: {} center_x: {} center_y: {} center_z: {}".format(
            self._radius, self._coord_origin[0], self._coord_origin[1],
            self._coord_origin[2])
        retdct[key] = value
        key = "genvol"
        value = "0 0 {} {} {}".format(self._radius, self._volume, nevents)
        retdct[key] = value
        return retdct
