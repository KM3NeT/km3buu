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

    @property
    def volume(self):
        """
        Returns
        -------
        float [m^3] 
           The detector volume
        """
        return self._volume


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
    """
    def __init__(self, radius=403.4, zmin=0.0, zmax=476.5):
        super().__init__()
        self._radius = radius
        self._zmin = zmin
        self._zmax = zmax
        self._volume = self._calc_volume()

    def _calc_volume(self):
        return np.pi * (self._zmax - self._zmin) * np.power(self._radius, 2)

    def random_pos(self):
        r = self._radius * np.sqrt(np.random.uniform(0, 1))
        phi = np.random.uniform(0, 2 * np.pi)
        pos_x = r * np.cos(phi)
        pos_y = r * np.sin(phi)
        pos_z = np.random.uniform(self._zmin, self._zmax)
        return (pos_x, pos_y, pos_z)


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
    def __init__(self, radius, center=(0, 0, 0)):
        super().__init__()
        self._radius = radius
        self._center = center
        self._volume = self._calc_volume()

    def _calc_volume(self):
        return 4 / 3 * np.pi * np.power(self._radius, 3)

    def random_pos(self):
        r = np.power(self._radius, 1 / 3)
        phi = np.random.uniform(0, np.pi)
        cosTheta = np.random.uniform(-1, 1)
        pos_x = r * np.cos(phi) * np.sqrt(1 - np.power(cosTheta, 2))
        pos_y = r * np.sin(phi) * np.sqrt(1 - np.power(cosTheta, 2))
        pos_z = r * cosTheta
        pos = (pos_x, pos_y, pos_z)
        return tuple(np.add(self._center, pos))
