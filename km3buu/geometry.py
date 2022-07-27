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


class DetectorVolume(ABC):
    """
    Detector geometry class
    """

    def __init__(self):
        self._volume = -1.0
        self._coord_origin = (0., 0., 0.)

    @abstractmethod
    def random_pos(self):
        """
        Generate a random position in the detector volume based on a uniform
        event distribution

        Returns
        -------
        tuple [m] (x, y, z)
        """
        pass

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


class CanVolume(DetectorVolume):
    """
    Cylindrical detector geometry

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
    """

    def __init__(self,
                 radius=403.4,
                 zmin=0.0,
                 zmax=476.5,
                 detector_center=(0., 0.)):
        super().__init__()
        self._radius = radius
        self._zmin = zmin
        self._zmax = zmax
        self._volume = self._calc_volume()
        self._detector_center = detector_center

    def _calc_volume(self):
        return np.pi * (self._zmax - self._zmin) * np.power(self._radius, 2)

    def random_pos(self):
        r = self._radius * np.sqrt(np.random.uniform(0, 1))
        phi = np.random.uniform(0, 2 * np.pi)
        pos_x = r * np.cos(phi) + self._detector_center[0]
        pos_y = r * np.sin(phi) + self._detector_center[1]
        pos_z = np.random.uniform(self._zmin, self._zmax)
        return (pos_x, pos_y, pos_z)

    def header_entries(self, nevents=0):
        retdct = dict()
        key = "genvol"
        value = "{} {} {} {} {}".format(self._zmin, self._zmax, self._radius,
                                        self._volume, nevents)
        retdct[key] = value
        key = "fixedcan"
        value = "0 0 {} {} {}".format(self._zmin, self._zmax, self._radius)
        return retdct


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
    """

    def __init__(self, radius, coord_origin=(0, 0, 0)):
        super().__init__()
        self._radius = radius
        self._coord_origin = coord_origin
        self._volume = self._calc_volume()

    def _calc_volume(self):
        return 4 / 3 * np.pi * np.power(self._radius, 3)

    def random_pos(self):
        r = self._radius * np.power(np.random.random(), 1 / 3)
        phi = np.random.uniform(0, 2 * np.pi)
        cosTheta = np.random.uniform(-1, 1)
        pos_x = r * np.cos(phi) * np.sqrt(1 - np.power(cosTheta, 2))
        pos_y = r * np.sin(phi) * np.sqrt(1 - np.power(cosTheta, 2))
        pos_z = r * cosTheta
        pos = (pos_x, pos_y, pos_z)
        return tuple(np.add(self._coord_origin, pos))

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
