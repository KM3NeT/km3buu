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
from os.path import isfile, join, abspath
from tempfile import TemporaryDirectory
import awkward as ak
import uproot
from scipy.interpolate import UnivariateSpline, interp1d
from scipy.spatial.transform import Rotation
import scipy.constants as constants
from scipy.optimize import curve_fit
import mendeleev
from datetime import datetime

from .physics import visible_energy_fraction
from .jobcard import Jobcard, read_jobcard, PDGID_LOOKUP
from .geometry import DetectorVolume, CanVolume
from .config import Config, read_default_media_compositions
from .__version__ import version

try:
    import ROOT
    libpath = environ.get("KM3NET_LIB")
    if libpath is None:
        libpath = Config().km3net_lib_path
    if ROOT.gSystem.Load(join(libpath, "libKM3NeTROOT.so")) < 0:
        raise ModuleNotFoundError("KM3NeT dataformat library not found!")
except ModuleNotFoundError:
    import warnings
    warnings.warn("KM3NeT dataformat library was not loaded.", ImportWarning)

EVENT_FILENAME = "FinalEvents.dat"
ROOT_PERT_FILENAME = "EventOutput.Pert.[0-9]{8}.root"
ROOT_REAL_FILENAME = "EventOutput.Real.[0-9]{8}.root"
FLUXDESCR_FILENAME = "neutrino_initialized_energyFlux.dat"
XSECTION_FILENAMES = {"all": "neutrino_absorption_cross_section_ALL.dat"}

SECONDS_PER_YEAR = 365.25 * 24 * 60 * 60
SECONDS_WEIGHT_TIMESPAN = 1

PARTICLE_COLUMNS = ["E", "Px", "Py", "Pz", "barcode"]
EVENTINFO_COLUMNS = [
    "weight", "evType", "lepIn_E", "lepIn_Px", "lepIn_Py", "lepIn_Pz",
    "lepOut_E", "lepOut_Px", "lepOut_Py", "lepOut_Pz", "nuc_E", "nuc_Px",
    "nuc_Py", "nuc_Pz"
]

LHE_NU_INFO_DTYPE = np.dtype([
    ("type", np.int),
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
    "can": "0 0 0",
    "flux": "0 0 0",
    "coord_origin": "0 0 0",
    "norma": "0 0",
    "tgen": "0",
    "simul": "",
    "primary": "0",
    "genvol": "0 0 0 0 0"
}

PARTICLE_MC_STATUS = {
    "TRK_MOTHER_UNDEFINED":
    -1,  # mother id was not defined for this MC track (all reco tracks have this value)
    "TRK_MOTHER_NONE": -2,  # mother id of a particle if it has no parent
    "TRK_ST_UNDEFINED":
    0,  # status was not defined for this MC track (all reco tracks have this value)
    "TRK_ST_FINALSTATE":
    1,  # particle to be tracked by detector-level MC ('track_in' tag in evt files from gseagen, genhen, mupage).
    "TRK_ST_PRIMARYNEUTRINO":
    100,  # initial state neutrino ('neutrino' tag in evt files from gseagen and genhen).
    "TRK_ST_PRIMARYCOSMIC":
    200,  # initial state cosmic ray ('track_primary' tag in evt files from corant).
    "TRK_ST_ININUCLEI": 5,  # Initial state nuclei (gseagen)
    "TRK_ST_INTERSTATE":
    2,  # Intermidiate state particles produced in hadronic showers (gseagen)
    "TRK_ST_DECSTATE":
    3,  # Short-lived particles that are forced to decay, like pi0 (gseagen)
    "TRK_ST_NUCTGT": 11,  # Nucleon target (gseagen)
    "TRK_ST_PREHAD": 12,  # DIS pre-fragmentation hadronic state (gseagen)
    "TRK_ST_PRERES": 13,  # resonant pre-decayed state (gseagen)
    "TRK_ST_HADNUC": 14,  # Hadrons inside the nucleus before FSI (gseagen)
    "TRK_ST_NUCLREM": 15,  #Low energy nuclear fragments (gseagen)
    "TRK_ST_NUCLCLT":
    16,  #For composite nucleons before phase space decay (gseagen)
    "TRK_ST_FAKECORSIKA":
    21,  # fake particle from corant/CORSIKA to add parent information (gseagen)
    "TRK_ST_FAKECORSIKA_DEC_MU_START":
    22,  # fake particle from CORSIKA: decaying mu at start (gseagen)
    "TRK_ST_FAKECORSIKA_DEC_MU_END":
    23,  # fake particle from CORSIKA: decaying mu at end (gseagen)
    "TRK_ST_FAKECORSIKA_ETA_2GAMMA":
    24,  # fake particle from CORSIKA: eta -> 2 gamma (gseagen)
    "TRK_ST_FAKECORSIKA_ETA_3PI0":
    25,  # fake particle from CORSIKA: eta -> 3 pi0 (gseagen)
    "TRK_ST_FAKECORSIKA_ETA_PIP_PIM_PI0":
    26,  # fake particle from CORSIKA: eta -> pi+ pi- pi0 (gseagen)
    "TRK_ST_FAKECORSIKA_ETA_2PI_GAMMA":
    27,  # fake particle from CORSIKA: eta -> pi+ pi- gamma (gseagen)
    "TRK_ST_FAKECORSIKA_CHERENKOV_GAMMA":
    28,  # fake particle from CORSIKA: Cherenkov photons on particle output file (gseagen)
    "TRK_ST_PROPLEPTON":
    1001,  # lepton propagated that reaches the can (gseagen)
    "TRK_ST_PROPDECLEPTON":
    2001  # lepton propagated and decayed before got to the can (gseagen)
}

W2LIST_LOOKUP = {
    "PS": 0,
    "EG": 1,
    "XSEC_MEAN": 2,
    "COLUMN_DEPTH": 3,
    "P_EARTH": 4,
    "WATER_INT_LEN": 5,
    "BX": 7,
    "BY": 8,
    "ICHAN": 9,
    "CC": 10,
    "XSEC": 13,
    "DXSEC": 14,
    "TARGETA": 15,
    "TARGETZ": 16,
    "VERINCAN": 17,
    "LEPINCAN": 18,
    "GIBUU_WEIGHT": 23,
    "GIBUU_SCAT_TYPE": 24
}

W2LIST_LENGTH = max(W2LIST_LOOKUP.values()) + 1

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
        self._generated_events = -1
        self._flux_index = np.nan

        try:
            self._read_flux_file()
            self._determine_flux_index()
        except OSError:
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
        self.flux_data = np.loadtxt(fpath,
                                    dtype=FLUX_INFORMATION_DTYPE,
                                    usecols=(
                                        0,
                                        1,
                                        2,
                                    ))
        self.flux_interpolation = UnivariateSpline(self.flux_data["energy"],
                                                   self.flux_data["events"])
        self._energy_min = np.min(self.flux_data["energy"])
        self._energy_max = np.max(self.flux_data["energy"])
        self._generated_events = int(np.sum(self.flux_data["events"]))

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
        ) * self.A  #xsec_per_nucleon * no_nucleons in the core
        if self.flux_data is not None:
            inv_gen_flux = np.power(
                self.flux_interpolation(root_tupledata.lepIn_E), -1)
            energy_phase_space = self.flux_interpolation.integral(
                self._energy_min, self._energy_max)
            energy_factor = energy_phase_space * inv_gen_flux
        else:
            energy_factor = 1
        env_factor = volume * SECONDS_WEIGHT_TIMESPAN
        retval = env_factor * solid_angle * energy_factor * xsec * 10**-42 * target_density
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
        pk = d.lepIn_E * d.nuc_E - d.lepIn_Px * d.nuc_Px - d.lepIn_Py * d.nuc_Py - d.lepIn_Pz * d.nuc_Pz
        y = pq / pk
        return y

    @property
    def A(self):
        return self.jobcard["target"]["target_a"]

    @property
    def Z(self):
        return self.jobcard["target"]["target_z"]

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
        df = df.append(sec_df)
        return df

    @property
    def arrays(self):
        """
        GiBUU output data in awkward format
        """
        retval = None
        for ifile in self.root_pert_files:
            fobj = uproot.open(join(self.data_path, ifile))
            if retval is None:
                retval = fobj["RootTuple"].arrays()
            else:
                tmp = fobj["RootTuple"].arrays()
                retval = np.concatenate((retval, tmp))
        if retval is None or len(retval) == 0:
            return ak.Array(
                np.recarray((
                    0,
                    0,
                ),
                            dtype=list(
                                zip(GIBUU_FIELDNAMES,
                                    len(GIBUU_FIELDNAMES) * [float]))))
        # Calculate additional information
        counts = ak.num(retval.E)
        retval["xsec"] = self._event_xsec(retval)
        retval["Bx"] = GiBUUOutput.bjorken_x(retval)
        retval["By"] = GiBUUOutput.bjorken_y(retval)
        retval["Q2"] = GiBUUOutput.Qsquared(retval)
        visEfrac = visible_energy_fraction(ak.flatten(retval.E),
                                           ak.flatten(retval.barcode))
        retval["visEfrac"] = ak.unflatten(visEfrac, counts)
        return retval

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

        lower_limit = np.exp(np.log(np.max(self.flux_data["flux"])) * 0.2)
        upper_limit = np.exp(np.log(np.max(self.flux_data["flux"])) * 0.8)
        mask = (self.flux_data["flux"] > lower_limit) & (self.flux_data["flux"]
                                                         < upper_limit)
        popt, pcov = curve_fit(fluxfunc,
                               self.flux_data["energy"][mask],
                               self.flux_data["flux"][mask],
                               p0=[1, -1])

        self._flux_index = popt[1]

    @property
    def flux_index(self):
        return self._flux_index


def write_detector_file(gibuu_output,
                        ofile="gibuu.offline.root",
                        no_files=1,
                        geometry=CanVolume(),
                        livetime=3.156e7,
                        propagate_tau=True):  # pragma: no cover
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
    geometry: DetectorVolume
        The detector geometry which should be used
    livetime: float
        The data livetime
    """
    if not isinstance(geometry, DetectorVolume):
        raise TypeError("Geometry needs to be a DetectorVolume")

    def add_particles(particle_data, pos_offset, rotation, start_mc_trk_id,
                      timestamp, status):
        nonlocal evt
        for i in range(len(particle_data.E)):
            trk = ROOT.Trk()
            trk.id = start_mc_trk_id + i
            mom = np.array([
                particle_data.Px[i], particle_data.Py[i], particle_data.Pz[i]
            ])
            p_dir = rotation.apply(mom / np.linalg.norm(mom))
            try:
                prtcl_pos = np.array([
                    particle_data.x[i], particle_data.y[i], particle_data.z[i]
                ])
                prtcl_pos = rotation.apply(prtcl_pos)
                trk.pos.set(*np.add(pos_offset, prtcl_pos))
                trk.t = timestamp + particle_data.deltaT[i] * 1e9
            except AttributeError:
                trk.pos.set(*pos_offset)
                trk.t = timestamp
            trk.dir.set(*p_dir)
            trk.mother_id = 0
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

    if no_files > len(event_data):
        raise IndexError("More files to write than contained events!")

    bjorkenx = event_data.Bx
    bjorkeny = event_data.By

    tau_secondaries = None
    if propagate_tau and abs(nu_type) == 16 and ichan == 2:
        from .propagation import propagate_lepton
        tau_secondaries = propagate_lepton(event_data, np.sign(nu_type) * 15)

    media = read_default_media_compositions()
    density = media["SeaWater"]["density"]  # [g/cm^3]
    element = mendeleev.element(gibuu_output.Z)
    target = media["SeaWater"]["elements"][element.symbol]
    target_density = 1e3 * density * target[1]  # [kg/m^3]
    targets_per_volume = target_density / target[
        0].atomic_weight / constants.atomic_mass

    w2 = gibuu_output.w2weights(geometry.volume, targets_per_volume, 4 * np.pi)
    global_generation_weight = gibuu_output.global_generation_weight(4 * np.pi)
    mean_xsec_func = gibuu_output.mean_xsec

    header_dct = EMPTY_KM3NET_HEADER_DICT.copy()

    header_dct["target"] = element.name
    for k, v in geometry.header_entries(gibuu_output._generated_events //
                                        no_files).items():
        header_dct[k] = v
    header_dct["coord_origin"] = "{} {} {}".format(*geometry.coord_origin)
    header_dct["flux"] = "{:d} 0 0".format(nu_type)
    header_dct["cut_nu"] = "{:.2f} {:.2f} -1 1".format(gibuu_output.energy_min,
                                                       gibuu_output.energy_max)
    header_dct["tgen"] = "{:.1f}".format(livetime)
    header_dct["norma"] = "0 {}".format(gibuu_output.generated_events)
    timestamp = datetime.now()
    header_dct["simul"] = "KM3BUU {} {}".format(
        version, timestamp.strftime("%Y%m%d %H%M%S"))
    header_dct["primary"] = "{:d}".format(nu_type)

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
        mc_trk_id = 0

        head = ROOT.Head()
        for k, v in header_dct.items():
            head.set_line(k, v)
        head.Write("Head")

        for mc_event_id, event in enumerate(event_data[start_id:stop_id]):
            evt.clear()
            evt.id = mc_event_id
            evt.mc_run_id = mc_event_id
            # Weights
            evt.w.push_back(geometry.volume)  #w1 (can volume)
            evt.w.push_back(w2[start_id + mc_event_id])  #w2
            evt.w.push_back(-1.0)  #w3 (= w2*flux)
            # Event Information (w2list)
            evt.w2list.resize(W2LIST_LENGTH)
            evt.w2list[W2LIST_LOOKUP["PS"]] = global_generation_weight
            evt.w2list[W2LIST_LOOKUP["EG"]] = gibuu_output.flux_index
            evt.w2list[W2LIST_LOOKUP["XSEC_MEAN"]] = mean_xsec_func(
                event.lepIn_E)
            evt.w2list[W2LIST_LOOKUP["XSEC"]] = event.xsec
            evt.w2list[W2LIST_LOOKUP["TARGETA"]] = gibuu_output.A
            evt.w2list[W2LIST_LOOKUP["TARGETZ"]] = gibuu_output.Z
            evt.w2list[W2LIST_LOOKUP["BX"]] = bjorkenx[mc_event_id]
            evt.w2list[W2LIST_LOOKUP["BY"]] = bjorkeny[mc_event_id]
            evt.w2list[W2LIST_LOOKUP["CC"]] = ichan
            evt.w2list[W2LIST_LOOKUP["ICHAN"]] = SCATTERING_TYPE_TO_GENIE[
                event.evType]
            evt.w2list[W2LIST_LOOKUP["VERINCAN"]] = 1
            evt.w2list[W2LIST_LOOKUP["LEPINCAN"]] = 1
            evt.w2list[W2LIST_LOOKUP["GIBUU_WEIGHT"]] = event.weight
            evt.w2list[W2LIST_LOOKUP["GIBUU_SCAT_TYPE"]] = event.evType
            #TODO
            evt.w2list[W2LIST_LOOKUP["DXSEC"]] = np.nan
            evt.w2list[W2LIST_LOOKUP["COLUMN_DEPTH"]] = np.nan
            evt.w2list[W2LIST_LOOKUP["P_EARTH"]] = np.nan
            evt.w2list[W2LIST_LOOKUP["WATER_INT_LEN"]] = np.nan

            # Vertex Position
            vtx_pos = np.array(geometry.random_pos())
            # Direction
            phi = np.random.uniform(0, 2 * np.pi)
            cos_theta = np.random.uniform(-1, 1)
            sin_theta = np.sqrt(1 - cos_theta**2)

            dir_x = np.cos(phi) * sin_theta
            dir_y = np.sin(phi) * sin_theta
            dir_z = cos_theta

            direction = np.array([dir_x, dir_y, dir_z])
            theta = np.arccos(cos_theta)
            R = Rotation.from_euler("yz", [theta, phi])

            timestamp = np.random.uniform(0, livetime)

            nu_in_trk = ROOT.Trk()
            nu_in_trk.id = mc_trk_id
            mc_trk_id += 1
            nu_in_trk.mother_id = -1
            nu_in_trk.type = nu_type
            nu_in_trk.pos.set(*vtx_pos)
            nu_in_trk.dir.set(*direction)
            nu_in_trk.E = event.lepIn_E
            nu_in_trk.t = timestamp
            nu_in_trk.status = PARTICLE_MC_STATUS["TRK_ST_PRIMARYNEUTRINO"]
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

            if tau_secondaries is not None:
                lep_out_trk.status = PARTICLE_MC_STATUS["TRK_ST_UNDEFINED"]
            else:
                lep_out_trk.status = PARTICLE_MC_STATUS["TRK_ST_FINALSTATE"]

            evt.mc_trks.push_back(lep_out_trk)

            if tau_secondaries is not None:
                event_tau_sec = tau_secondaries[mc_event_id]
                add_particles(event_tau_sec, vtx_pos, R, mc_trk_id, timestamp,
                              PARTICLE_MC_STATUS["TRK_ST_FINALSTATE"])
                mc_trk_id += len(event_tau_sec.E)

            add_particles(event, vtx_pos, R, mc_trk_id, timestamp,
                          PARTICLE_MC_STATUS["TRK_ST_FINALSTATE"])
            tree.Fill()
        outfile.Write()
        outfile.Close()
        del head
        del outfile
        del evt
        del tree
