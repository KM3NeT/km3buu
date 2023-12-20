# Filename: output.py
"""
IO for km3buu

"""

__author__ = "Johannes Schumann"
__copyright__ = "Copyright 2020, Johannes Schumann and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Johannes Schumann"
__email__ = "jschumann@km3net.de"
__status__ = "Development"

import re
import numpy as np
from io import StringIO
from os import listdir, environ
from os.path import isfile, join, abspath, exists
from tempfile import TemporaryDirectory
import awkward as ak
import uproot
from scipy.interpolate import UnivariateSpline, interp1d
from scipy.spatial.transform import Rotation
from scipy.optimize import curve_fit
from datetime import datetime
import km3io

from .physics import visible_energy_fraction, get_targets_per_volume
from .jobcard import Jobcard, read_jobcard, PDGID_LOOKUP
from .geometry import *
from .config import Config, read_default_media_compositions
from .__version__ import version

try:
    import ROOT
    libpath = environ.get("KM3NET_LIB")
    if libpath is None:
        libpath = Config().km3net_lib_path
    if libpath is None or ROOT.gSystem.Load(join(libpath,
                                                 "libKM3NeTROOT.so")) < 0:
        raise ModuleNotFoundError("KM3NeT dataformat library not found!")
except (ImportError, ModuleNotFoundError):
    import warnings
    warnings.warn("KM3NeT dataformat library was not loaded.", ImportWarning)

EVENT_FILENAME = "FinalEvents.dat"
ROOT_PERT_FILENAME = "EventOutput.Pert.*.root"
ROOT_REAL_FILENAME = "EventOutput.Real.*.root"
FLUXDESCR_FILENAME = "neutrino_initialized_energyFlux.dat"
XSECTION_FILENAMES = {"all": "neutrino_absorption_cross_section_ALL.dat"}

SECONDS_PER_YEAR = 365.25 * 24 * 60 * 60
SECONDS_WEIGHT_TIMESPAN = 1

PARTICLE_COLUMNS = ["E", "Px", "Py", "Pz", "x", "y", "z", "barcode"]
EVENTINFO_COLUMNS = [
    "weight", "evType", "lepIn_E", "lepIn_Px", "lepIn_Py", "lepIn_Pz",
    "lepOut_E", "lepOut_Px", "lepOut_Py", "lepOut_Pz", "nuc_E", "nuc_Px",
    "nuc_Py", "nuc_Pz", "nucleus_A", "nucleus_Z", "flavor_ID", "process_ID",
    "numRuns", "numEnsembles"
]

LHE_NU_INFO_DTYPE = np.dtype([
    ("type", np.int64),
    ("weight", np.float64),
    ("mom_lepton_in_E", np.float64),
    ("mom_lepton_in_x", np.float64),
    ("mom_lepton_in_y", np.float64),
    ("mom_lepton_in_z", np.float64),
    ("mom_lepton_out_E", np.float64),
    ("mom_lepton_out_x", np.float64),
    ("mom_lepton_out_y", np.float64),
    ("mom_lepton_out_z", np.float64),
    ("mom_nucleon_in_E", np.float64),
    ("mom_nucleon_in_x", np.float64),
    ("mom_nucleon_in_y", np.float64),
    ("mom_nucleon_in_z", np.float64),
])

FLUX_INFORMATION_DTYPE = np.dtype([("energy", np.float64),
                                   ("flux", np.float64),
                                   ("events", np.float64)])

SCATTERING_TYPE = {
    1: "QE",
    2: "P33(1232)",
    3: "P11(1440)",
    4: "S11(1535)",
    5: "S11(1650)",
    6: "S11(2090)",
    7: "D13(1520)",
    8: "D13(1700)",
    9: "D13(2080)",
    10: "D15(1675)",
    11: "G17(2190)",
    12: "P11(1710)",
    13: "P11(2100)",
    14: "P13(1720)",
    15: "P13(1900)",
    16: "F15(1680)",
    17: "F15(2000)",
    18: "F17(1990)",
    19: "S31(1620)",
    20: "S31(1900)",
    21: "D33(1700)",
    22: "D33(1940)",
    23: "D35(1930)",
    24: "D35(2350)",
    25: "P31(1750)",
    26: "P31(1910)",
    27: "P33(1600)",
    28: "P33(1920)",
    29: "F35(1750)",
    30: "F35(1905)",
    31: "F37(1950)",
    32: "pi neutron-background",
    33: "pi proton-background",
    34: "DIS",
    35: "2p2h QE",
    36: "2p2h Delta",
    37: "2pi background",
}

SCATTERING_TYPE_TO_GENIE = {
    1: 1,  # QE        -> kScQuasiElastic
    2: 4,  # P33(1232) -> kScResonant
    3: 4,  # P11(1440) -> kScResonant
    4: 4,  # S11(1535) -> kScResonant
    5: 4,  # S11(1650) -> kScResonant
    6: 4,  # S11(2090) -> kScResonant
    7: 4,  # D13(1520) -> kScResonant
    8: 4,  # D13(1700) -> kScResonant
    9: 4,  # D13(2080) -> kScResonant
    10: 4,  # D15(1675) -> kScResonant
    11: 4,  # G17(2190) -> kScResonant
    12: 4,  # P11(1710) -> kScResonant
    13: 4,  # P11(2100) -> kScResonant
    14: 4,  # P13(1720) -> kScResonant
    15: 4,  # P13(1900) -> kScResonant
    16: 4,  # F15(1680) -> kScResonant
    17: 4,  # F15(2000) -> kScResonant
    18: 4,  # F17(1990) -> kScResonant
    19: 4,  # S31(1620) -> kScResonant
    20: 4,  # S31(1900) -> kScResonant
    21: 4,  # D33(1700) -> kScResonant
    22: 4,  # D33(1940) -> kScResonant
    23: 4,  # D35(1930) -> kScResonant
    24: 4,  # D35(2350) -> kScResonant
    25: 4,  # P31(1750) -> kScResonant
    26: 4,  # P31(1910) -> kScResonant
    27: 4,  # P33(1600) -> kScResonant
    28: 4,  # P33(1920) -> kScResonant
    29: 4,  # F35(1750) -> kScResonant
    30: 4,  # F35(1905) -> kScResonant
    31: 4,  # F37(1950) -> kScResonant
    32: 0,  # pi neutron-background -> kScNull
    33: 0,  # pi proton-background -> kScNull
    34: 3,  # DIS -> kScDeepInelastic
    35: 0,  # 2p2h QE -> kScNull
    36: 0,  # 2p2h Delta -> kScNull
    37: 0,  # 2pi background -> kScNull
}

ROOTTUPLE_KEY = "RootTuple"

EMPTY_KM3NET_HEADER_DICT = {
    "start_run": "0",
    "drawing": "volume",
    "depth": "2475.0",
    "target": "",
    "cut_nu": "0 0 0 0",
    "spectrum": "0",
    "flux": "0 0 0",
    "coord_origin": "0 0 0",
    "norma": "0 0",
    "tgen": "0",
    "simul": "",
    "primary": "0",
    "target": "None"
}

W2LIST_LENGTH = max(km3io.definitions.w2list_km3buu.values()) + 1

GIBUU_FIELDNAMES = [
    'weight', 'barcode', 'Px', 'Py', 'Pz', 'E', 'evType', 'lepIn_E',
    'lepIn_Px', 'lepIn_Py', 'lepIn_Pz', 'lepOut_E', 'lepOut_Px', 'lepOut_Py',
    'lepOut_Pz', 'nuc_E', 'nuc_Px', 'nuc_Py', 'nuc_Pz'
]


def read_nu_abs_xsection(filepath):
    """
    Read the crosssections calculated by GiBUU

    Parameters
    ----------
    filepath: str
        Filepath to the GiBUU output file with neutrino absorption cross-section
        (neutrino_absorption_cross_section_*.dat)

    """
    with open(filepath, "r") as f:
        lines = f.readlines()
    header = re.sub(r"\d+:|#", "", lines[0]).split()
    dt = np.dtype([(field, np.float64) for field in header])
    values = np.genfromtxt(StringIO(lines[-1]), dtype=dt)
    return values


class GiBUUOutput:

    def __init__(self, data_dir):
        """
        Class for parsing GiBUU output files

        Parameters
        ----------
        data_dir: str
            Path to the GiBUU output directory
        """
        if isinstance(data_dir, TemporaryDirectory):
            self._tmp_dir = data_dir
            self._data_path = abspath(data_dir.name)
        else:
            self._data_path = abspath(data_dir)
        self.output_files = [
            f for f in listdir(self._data_path)
            if isfile(join(self._data_path, f))
        ]
        self._read_xsection_file()
        self._read_root_output()
        self._read_jobcard()

        self.flux_data = None
        self._min_energy = np.nan
        self._max_energy = np.nan
        self._written_events = len(self._get_raw_arrays())
        self._generated_events = -1
        self._flux_index = np.nan

        if self._read_flux_file():
            self._determine_flux_index()
        else:
            self._read_single_energy()

    def _read_root_output(self):
        root_pert_regex = re.compile(ROOT_PERT_FILENAME)
        self.root_pert_files = list(
            filter(root_pert_regex.match, self.output_files))

        root_real_regex = re.compile(ROOT_REAL_FILENAME)
        self.root_real_files = list(
            filter(root_real_regex.match, self.output_files))

    def _read_xsection_file(self):
        if XSECTION_FILENAMES["all"] in self.output_files:
            setattr(
                self,
                "xsection",
                read_nu_abs_xsection(
                    join(self._data_path, XSECTION_FILENAMES["all"])),
            )

    def _read_jobcard(self):
        jobcard_regex = re.compile(".*.job")
        jobcard_files = list(filter(jobcard_regex.match, self.output_files))
        if len(jobcard_files) == 1:
            self._jobcard_fname = jobcard_files[0]
            self.jobcard = read_jobcard(
                join(self._data_path, self._jobcard_fname))
        else:
            self.jobcard = None

    def _read_single_energy(self):
        root_tupledata = self.arrays
        energies = np.array(root_tupledata.lepIn_E)
        if np.std(energies) > 1e-10:
            raise NotImplementedError(
                "Energy not constant; run data cannot be interpreted")
        self._min_energy = np.mean(energies)
        self._max_energy = self._max_energy
        num_ensembles = int(self.jobcard["input"]["numensembles"])
        num_runs = int(self.jobcard["input"]["num_runs_sameenergy"])
        self._generated_events = num_ensembles * num_runs

    def _read_flux_file(self):
        fpath = join(self._data_path, FLUXDESCR_FILENAME)
        if not exists(fpath):
            return False
        self.flux_data = np.loadtxt(fpath,
                                    dtype=FLUX_INFORMATION_DTYPE,
                                    usecols=(
                                        0,
                                        1,
                                        2,
                                    ))
        self.flux_interpolation = UnivariateSpline(self.flux_data["energy"],
                                                   self.flux_data["events"],
                                                   s=0)
        self._energy_min = np.min(self.flux_data["energy"])
        self._energy_max = np.max(self.flux_data["energy"])
        self._generated_events = int(np.sum(self.flux_data["events"]))
        return True

    def _event_xsec(self, root_tupledata):
        weights = np.array(root_tupledata.weight)
        total_events = self._generated_events
        n_files = len(self.root_pert_files)
        xsec = np.divide(total_events * weights, n_files)
        return xsec

    @property
    def mean_xsec(self):
        if self.flux_data is None:
            return lambda energy: self.xsection["sum"]
        else:
            root_tupledata = self.arrays
            energies = np.array(root_tupledata.lepIn_E)
            weights = self._event_xsec(root_tupledata)
            Emin = np.min(energies)
            Emax = np.max(energies)
            xsec, energy_bins = np.histogram(energies,
                                             weights=weights,
                                             bins=np.logspace(
                                                 np.log10(Emin),
                                                 np.log10(Emax), 15))
            deltaE = np.mean(self.flux_data["energy"][1:] -
                             self.flux_data["energy"][:-1])
            bin_events = np.array([
                self.flux_interpolation.integral(energy_bins[i],
                                                 energy_bins[i + 1]) / deltaE
                for i in range(len(energy_bins) - 1)
            ])
            x = (energy_bins[1:] + energy_bins[:-1]) / 2
            y = xsec / bin_events / x
            xsec_interp = interp1d(x,
                                   y,
                                   kind="linear",
                                   fill_value=(y[0], y[-1]),
                                   bounds_error=False)
            return lambda e: xsec_interp(e) * e

    def global_generation_weight(self, solid_angle):
        # I_E * I_theta * t_gen (* #NuTypes)
        if self.flux_data is not None:
            energy_phase_space = self.flux_interpolation.integral(
                self._energy_min, self._energy_max)
        else:
            energy_phase_space = 1
        return solid_angle * energy_phase_space * SECONDS_WEIGHT_TIMESPAN

    def w2weights(self, volume, target_density, solid_angle):
        """
        Calculate w2weights

        Parameters
        ----------
            volume: float [m^3]
                The interaction volume
            target_density: float [m^-3]
                N_A * ρ
            solid_angle: float
                Solid angle of the possible neutrino incident direction

        """
        root_tupledata = self.arrays
        xsec = self._event_xsec(
            root_tupledata
        ) * self.A  # xsec_per_nucleon * no_nucleons in the core
        if self.flux_data is not None:
            inv_gen_flux = np.power(
                self.flux_interpolation(root_tupledata.lepIn_E), -1)
            energy_phase_space = self.flux_interpolation.integral(
                self._energy_min, self._energy_max)
            energy_factor = energy_phase_space * inv_gen_flux
        else:
            energy_factor = 1
        env_factor = volume * SECONDS_WEIGHT_TIMESPAN
        retval = env_factor * solid_angle * \
            energy_factor * xsec * 1e-42 * target_density
        return retval

    @staticmethod
    def _q(roottuple_data):
        """
        Calculate the difference of the four vectors from 
        in and out going lepton (k_in - k_out)
        """
        d = roottuple_data
        k_in = np.vstack([
            np.array(d.lepIn_E),
            np.array(d.lepIn_Px),
            np.array(d.lepIn_Py),
            np.array(d.lepIn_Pz)
        ])
        k_out = np.vstack([
            np.array(d.lepOut_E),
            np.array(d.lepOut_Px),
            np.array(d.lepOut_Py),
            np.array(d.lepOut_Pz)
        ])
        q = k_in - k_out
        return q

    @staticmethod
    def Qsquared(roottuple_data):
        """
        Calculate the passed energy from the neutrino interaction to the 
        nucleus denoted by the variable Q²

        Parameters
        ----------
            roottuple_data: awkward.highlevel.Array                

        Returns
        -------

        """
        q = GiBUUOutput._q(roottuple_data)
        qtmp = np.power(q, 2)
        q2 = qtmp[0, :] - np.sum(qtmp[1:4, :], axis=0)
        return q2

    @staticmethod
    def bjorken_x(roottuple_data):
        """
        Calculate Bjorken x variable for the GiBUU events

        Parameters
        ----------
            roottuple_data: awkward.highlevel.Array                
        """
        d = roottuple_data
        q = GiBUUOutput._q(d)
        q2 = GiBUUOutput.Qsquared(d)
        pq = q[0, :] * d.nuc_E - q[1, :] * d.nuc_Px - q[2, :] * d.nuc_Py - q[
            3, :] * d.nuc_Pz
        x = np.divide(-q2, 2 * pq)
        return np.array(x)

    @staticmethod
    def bjorken_y(roottuple_data):
        """
        Calculate Bjorken y scaling variable for the GiBUU events

        Definition: y = pq/pk

        Parameters
        ----------
            roottuple_data: awkward.highlevel.Array                
        """
        d = roottuple_data
        q = GiBUUOutput._q(d)
        pq = q[0, :] * d.nuc_E - q[1, :] * d.nuc_Px - q[2, :] * d.nuc_Py - q[
            3, :] * d.nuc_Pz
        pk = d.lepIn_E * d.nuc_E - d.lepIn_Px * d.nuc_Px - \
            d.lepIn_Py * d.nuc_Py - d.lepIn_Pz * d.nuc_Pz
        y = pq / pk
        return y

    @property
    def A(self):
        grp = self.jobcard["target"]
        if "a" in grp.keys():
            return grp["a"]
        elif "target_a" in grp.keys():
            return grp["target_a"]

    @property
    def Z(self):
        grp = self.jobcard["target"]
        if "z" in grp.keys():
            return grp["z"]
        elif "target_z" in grp.keys():
            return grp["target_z"]

    @property
    def data_path(self):
        return self._data_path

    @property
    def df(self):
        """
        GiBUU output data in pandas dataframe format
        """
        import pandas as pd
        df = ak.to_pandas(self.arrays)
        if len(df) == 0:
            return df
        sec_df = df[df.index.get_level_values(1) == 0].copy()
        sec_df.loc[:, "E"] = sec_df.lepOut_E
        sec_df.loc[:, "Px"] = sec_df.lepOut_Px
        sec_df.loc[:, "Py"] = sec_df.lepOut_Py
        sec_df.loc[:, "Pz"] = sec_df.lepOut_Pz
        sec_pdgid = (
            PDGID_LOOKUP[self.jobcard["neutrino_induced"]["flavor_id"]] -
            1) * np.sign(self.jobcard["neutrino_induced"]["process_id"])
        sec_df.loc[:, "barcode"] = sec_pdgid
        sec_df.index = pd.MultiIndex.from_tuples(
            zip(*np.unique(df.index.get_level_values(0), return_counts=True)))
        df = pd.concat([df, sec_df])
        return df

    def _get_raw_arrays(self):
        path_descr = join(self.data_path, ROOT_PERT_FILENAME) + ":RootTuple"
        return uproot.concatenate(path_descr)

    @staticmethod
    def _invariant_target_mass(roottuple_data):
        d = roottuple_data
        q = GiBUUOutput._q(d)
        W2 = (q[0, :] + d.nuc_E)**2 - (q[1, :] + d.nuc_Px)**2 - (
            q[2, :] + d.nuc_Py)**2 - (q[3, :] + d.nuc_Pz)**2
        return W2

    @property
    def arrays(self):
        """
        GiBUU output data in awkward format
        """
        retval = self._get_raw_arrays()
        # Calculate additional information
        counts = ak.num(retval.E)
        retval["xsec"] = self._event_xsec(retval)
        retval["Bx"] = GiBUUOutput.bjorken_x(retval)
        retval["By"] = GiBUUOutput.bjorken_y(retval)
        retval["Q2"] = GiBUUOutput.Qsquared(retval)
        retval["M2"] = retval["E"]**2 - retval["Px"]**2 - retval[
            "Py"]**2 - retval["Pz"]**2
        retval["W2"] = GiBUUOutput._invariant_target_mass(retval)
        if "x" in retval.fields:
            retval["R"] = retval["x"]**2 + retval["y"]**2 + retval["z"]**2
        visEfrac = visible_energy_fraction(ak.flatten(retval.E),
                                           ak.flatten(retval.barcode))
        retval["visEfrac"] = ak.unflatten(visEfrac, counts)
        return retval

    @property
    def free_particle_mask(self):
        from particle import Particle
        arr = self.arrays
        nums = ak.num(arr.barcode)
        pdgid = ak.flatten(arr.barcode)
        masses = ak.flatten(arr.M2)
        mask = np.greater_equal(
            masses,
            ak.from_iter(
                map(lambda x: (Particle.from_pdgid(x).mass * 1e-3)**2, pdgid)))
        mask = mask | ~np.isin(np.array(np.abs(pdgid)), [2112, 2212])
        return ak.unflatten(mask, nums)

    @property
    def energy_min(self):
        return self._min_energy

    @property
    def energy_max(self):
        return self._max_energy

    @property
    def generated_events(self):
        return self._generated_events

    def _determine_flux_index(self):

        def fluxfunc(x, a, b):
            return a * x**b

        energy_mask = self.flux_data["flux"] > 0
        lower_limit = np.min(self.flux_data["energy"][energy_mask]) * 1.2
        upper_limit = np.max(self.flux_data["energy"][energy_mask]) * 0.8
        mask = (self.flux_data["energy"]
                > lower_limit) & (self.flux_data["energy"] < upper_limit)
        try:
            popt, pcov = curve_fit(fluxfunc,
                                   self.flux_data["energy"][mask],
                                   self.flux_data["flux"][mask],
                                   p0=[1, -1])
            self._flux_index = popt[1]
        except:
            self._flux_index = np.nan

    @property
    def flux_index(self):
        return self._flux_index


def write_detector_file(gibuu_output,
                        ofile="gibuu.offline.root",
                        no_files=1,
                        run_number=1,
                        geometry=CylindricalVolume(),
                        timeinterval=(0.0, 1684345837.0),
                        free_particle_cuts=True):  # pragma: no cover
    """
    Convert the GiBUU output to a KM3NeT MC (OfflineFormat) file

    Parameters
    ----------
    gibuu_output: GiBUUOutput
        Output object which wraps the information from the GiBUU output files
    ofile: str
        Output filename
    no_files: int (default: 1)
        Number of output files written
    run_number: int (default: 1)
        Run number which is written to the file header(s)
    geometry: DetectorVolume
        The detector geometry which should be used
    timeinterval: float [s]
        The unix time time interval where the events are distributed in
    free_particle_cuts: boolean (default: True)
        Apply cuts in order to select particles which exit the nucleus
    """
    if not isinstance(geometry, DetectorVolume):
        raise TypeError("Geometry needs to be a DetectorVolume")

    def add_particles(particle_data, start_mc_trk_id, timestamp, status,
                      mother_id):
        nonlocal evt
        for i in range(len(particle_data.E)):
            trk = ROOT.Trk()
            trk.id = start_mc_trk_id + i
            p_dir = None
            if "Px" in particle_data.fields:
                mom = np.array([
                    particle_data.Px[i], particle_data.Py[i],
                    particle_data.Pz[i]
                ])
                p_dir = mom / np.linalg.norm(mom)
            else:
                p_dir = np.array([
                    particle_data.Dx[i], particle_data.Dy[i],
                    particle_data.Dz[i]
                ])
            trk.dir.set(*p_dir)
            prtcl_pos = np.array(
                [particle_data.x[i], particle_data.y[i], particle_data.z[i]])
            trk.pos.set(*prtcl_pos)
            trk.t = timestamp + particle_data.deltaT[i]
            trk.mother_id = mother_id
            trk.type = int(particle_data.barcode[i])
            trk.E = particle_data.E[i]
            trk.status = status
            evt.mc_trks.push_back(trk)

    is_cc = False

    if gibuu_output.jobcard is None:
        raise EnvironmentError("No jobcard provided within the GiBUU output!")

    nu_type = PDGID_LOOKUP[gibuu_output.jobcard["neutrino_induced"]
                           ["flavor_id"]]
    sec_lep_type = nu_type
    ichan = abs(gibuu_output.jobcard["neutrino_induced"]["process_id"])
    if ichan == 2:
        is_cc = True
        sec_lep_type -= 1
    if gibuu_output.jobcard["neutrino_induced"]["process_id"] < 0:
        nu_type *= -1
        sec_lep_type *= -1

    event_data = gibuu_output.arrays

    if free_particle_cuts:
        mask = gibuu_output.free_particle_mask
        for field in PARTICLE_COLUMNS:
            if field in event_data.fields:
                event_data[field] = event_data[field][mask]

    if no_files > len(event_data):
        raise IndexError("More files to write than contained events!")

    bjorkenx = event_data.Bx
    bjorkeny = event_data.By

    w2 = gibuu_output.w2weights(geometry.volume, 1, geometry.solid_angle)
    global_generation_weight = gibuu_output.global_generation_weight(
        geometry.solid_angle)
    mean_xsec_func = gibuu_output.mean_xsec

    header_dct = EMPTY_KM3NET_HEADER_DICT.copy()

    header_dct["target"] = "A{:d}Z{:d}".format(gibuu_output.A, gibuu_output.Z)
    header_dct["gibuu_Nevents"] = str(gibuu_output._generated_events)
    header_dct["n_split_files"] = str(no_files)
    header_dct["coord_origin"] = "{} {} {}".format(*geometry.coord_origin)
    header_dct["flux"] = "{:d} 0 0".format(nu_type)
    header_dct["cut_nu"] = "{:.2f} {:.2f} -1 1".format(gibuu_output.energy_min,
                                                       gibuu_output.energy_max)
    livetime = (timeinterval[1] - timeinterval[0])
    header_dct["tgen"] = "{:.1f}".format(livetime)
    header_dct["norma"] = "0 {}".format(gibuu_output.generated_events)
    timestamp = datetime.now()
    header_dct["simul"] = "KM3BUU {} {}".format(
        version, timestamp.strftime("%Y%m%d %H%M%S"))
    header_dct["primary"] = "{:d}".format(nu_type)
    header_dct["start_run"] = str(run_number)
    header_dct["target"] = "A{}Z{}".format(gibuu_output.A, gibuu_output.Z)

    event_times = np.sort(
        np.random.uniform(timeinterval[0], timeinterval[1], len(event_data)))

    for i in range(no_files):
        start_id = 0
        stop_id = len(event_data)
        tmp_filename = ofile
        if no_files > 1:
            tmp_filename = ofile.replace(".root", ".{}.root".format(i + 1))
            bunch_size = stop_id // no_files
            start_id = i * bunch_size
            stop_id = (i + 1) * bunch_size if i < (no_files - 1) else stop_id

        evt = ROOT.Evt()
        outfile = ROOT.TFile.Open(tmp_filename, "RECREATE")
        tree = ROOT.TTree("E", "KM3NeT Evt Tree")
        tree.Branch("Evt", evt, 32000, 4)
        for mc_event_id, event in enumerate(event_data[start_id:stop_id]):
            mc_trk_id = 0
            total_id = start_id + mc_event_id
            evt.clear()
            evt.id = mc_event_id
            evt.mc_run_id = mc_event_id
            t = event_times[total_id]
            seconds = int(t)
            nano_seconds = int((t % 1) * 1e9)
            evt.mc_event_time = ROOT.TTimeStamp(seconds, nano_seconds)
            # Vertex Positioning & Propagation
            try:
                vtx_pos, vtx_angles, samples, prop_particles, targets_per_volume = geometry.distribute_event(
                    event)
            except:
                continue
            # Weights
            evt.w.push_back(geometry.volume)  # w1 (can volume)
            evt.w.push_back(w2[total_id] * targets_per_volume / samples)  # w2
            evt.w.push_back(-1.0)  # w3 (= w2*flux)
            # Event Information (w2list)
            evt.w2list.resize(W2LIST_LENGTH)
            evt.w2list[km3io.definitions.w2list_km3buu[
                "W2LIST_KM3BUU_PS"]] = global_generation_weight
            evt.w2list[km3io.definitions.w2list_km3buu[
                "W2LIST_KM3BUU_EG"]] = gibuu_output.flux_index
            evt.w2list[km3io.definitions.w2list_km3buu[
                "W2LIST_KM3BUU_XSEC_MEAN"]] = mean_xsec_func(event.lepIn_E)
            evt.w2list[km3io.definitions.
                       w2list_km3buu["W2LIST_KM3BUU_XSEC"]] = event.xsec
            evt.w2list[km3io.definitions.
                       w2list_km3buu["W2LIST_KM3BUU_TARGETA"]] = gibuu_output.A
            evt.w2list[km3io.definitions.
                       w2list_km3buu["W2LIST_KM3BUU_TARGETZ"]] = gibuu_output.Z
            evt.w2list[km3io.definitions.w2list_km3buu[
                "W2LIST_KM3BUU_BX"]] = bjorkenx[mc_event_id]
            evt.w2list[km3io.definitions.w2list_km3buu[
                "W2LIST_KM3BUU_BY"]] = bjorkeny[mc_event_id]
            evt.w2list[
                km3io.definitions.w2list_km3buu["W2LIST_KM3BUU_CC"]] = ichan
            evt.w2list[km3io.definitions.w2list_km3buu[
                "W2LIST_KM3BUU_ICHAN"]] = SCATTERING_TYPE_TO_GENIE[
                    event.evType]
            evt.w2list[km3io.definitions.w2list_km3buu[
                "W2LIST_KM3BUU_VERINCAN"]] = 1 if geometry.in_can(
                    vtx_pos) else 0
            evt.w2list[km3io.definitions.w2list_km3buu[
                "W2LIST_KM3BUU_LEPINCAN"]] = 1  # Only LepInCan events are written out currently
            evt.w2list[km3io.definitions.w2list_km3buu[
                "W2LIST_KM3BUU_GIBUU_WEIGHT"]] = event.weight
            evt.w2list[km3io.definitions.w2list_km3buu[
                "W2LIST_KM3BUU_GIBUU_SCAT_TYPE"]] = event.evType
            # TODO
            evt.w2list[km3io.definitions.
                       w2list_km3buu["W2LIST_KM3BUU_DXSEC"]] = np.nan
            evt.w2list[km3io.definitions.
                       w2list_km3buu["W2LIST_KM3BUU_COLUMN_DEPTH"]] = np.nan
            evt.w2list[km3io.definitions.
                       w2list_km3buu["W2LIST_KM3BUU_P_EARTH"]] = np.nan
            evt.w2list[km3io.definitions.
                       w2list_km3buu["W2LIST_KM3BUU_WATER_INT_LEN"]] = np.nan

            timestamp = 0.0
            # Direction
            phi, cos_theta = vtx_angles
            sin_theta = np.sqrt(1 - cos_theta**2)

            dir_x = np.cos(phi) * sin_theta
            dir_y = np.sin(phi) * sin_theta
            dir_z = cos_theta

            direction = np.array([dir_x, dir_y, dir_z])
            theta = np.arccos(cos_theta)
            R = Rotation.from_euler("yz", [theta, phi])

            nu_in_trk = ROOT.Trk()
            nu_in_trk.id = mc_trk_id
            mc_trk_id += 1
            nu_in_trk.mother_id = -1
            nu_in_trk.type = nu_type
            nu_in_trk.pos.set(*vtx_pos)
            nu_in_trk.dir.set(*direction)
            nu_in_trk.E = event.lepIn_E
            nu_in_trk.t = timestamp
            nu_in_trk.status = km3io.definitions.trkmembers[
                "TRK_ST_PRIMARYNEUTRINO"]
            evt.mc_trks.push_back(nu_in_trk)

            lep_out_trk = ROOT.Trk()
            lep_out_trk.id = mc_trk_id
            mc_trk_id += 1
            lep_out_trk.mother_id = 0
            lep_out_trk.type = sec_lep_type
            lep_out_trk.pos.set(*vtx_pos)
            mom = np.array([event.lepOut_Px, event.lepOut_Py, event.lepOut_Pz])
            p_dir = R.apply(mom / np.linalg.norm(mom))
            lep_out_trk.dir.set(*p_dir)
            lep_out_trk.E = event.lepOut_E
            lep_out_trk.t = timestamp

            generator_particle_state = km3io.definitions.trkmembers[
                "TRK_ST_UNDEFINED"]
            if geometry.in_can(vtx_pos):
                generator_particle_state = km3io.definitions.trkmembers[
                    "TRK_ST_FINALSTATE"]
                lep_out_trk.status = km3io.definitions.trkmembers[
                    "TRK_ST_FINALSTATE"]

            if prop_particles is not None:
                if abs(sec_lep_type) == 15:
                    lep_out_trk.status = km3io.definitions.trkmembers[
                        "TRK_ST_PROPDECLEPTON"]
                else:
                    lep_out_trk.status = km3io.definitions.trkmembers[
                        "TRK_ST_PROPLEPTON"]

            evt.mc_trks.push_back(lep_out_trk)

            event.x = np.ones(len(event.E)) * vtx_pos[0]
            event.y = np.ones(len(event.E)) * vtx_pos[1]
            event.z = np.ones(len(event.E)) * vtx_pos[2]

            directions = R.apply(np.array([event.Px, event.Py, event.Pz]).T)
            event.Px = directions[:, 0]
            event.Py = directions[:, 1]
            event.Pz = directions[:, 2]

            event.deltaT = np.zeros(len(event.E))

            add_particles(event, mc_trk_id, timestamp,
                          generator_particle_state, 0)
            mc_trk_id += len(event.E)

            if prop_particles is not None:
                add_particles(
                    prop_particles, mc_trk_id, timestamp,
                    km3io.definitions.trkmembers["TRK_ST_FINALSTATE"], 1)
                mc_trk_id += len(prop_particles.E)

            tree.Fill()

        for k, v in geometry.header_entries(
                gibuu_output._generated_events).items():
            header_dct[k] = v

        head = ROOT.Head()
        for k, v in header_dct.items():
            head.set_line(k, v)
        head.Write("Head")

        outfile.Write()
        outfile.Close()
        del head
        del outfile
        del evt
        del tree
