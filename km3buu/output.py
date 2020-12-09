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
from os import listdir
from os.path import isfile, join, abspath
from tempfile import TemporaryDirectory
import awkward as ak
import uproot
from scipy.interpolate import UnivariateSpline
from scipy.spatial.transform import Rotation
import scipy.constants as constants
import mendeleev

from .jobcard import Jobcard, read_jobcard, PDGID_LOOKUP
from .config import read_default_media_compositions

EVENT_FILENAME = "FinalEvents.dat"
ROOT_PERT_FILENAME = "EventOutput.Pert.[0-9]{8}.root"
ROOT_REAL_FILENAME = "EventOutput.Real.[0-9]{8}.root"
FLUXDESCR_FILENAME = "neutrino_initialized_energyFlux.dat"
XSECTION_FILENAMES = {"all": "neutrino_absorption_cross_section_ALL.dat"}

SECONDS_PER_YEAR = 365.25 * 24 * 60 * 60

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

EVENT_TYPE = {
    1: "QE",
    32: "pi neutron-background",
    33: "pi proton-background",
    34: "DIS",
    35: "2p2h QE",
    36: "2p2h Delta",
    37: "2pi background",
}

ROOTTUPLE_KEY = "RootTuple"


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
        self._read_flux_file()
        self._read_jobcard()

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

    def _read_flux_file(self):
        fpath = join(self._data_path, FLUXDESCR_FILENAME)
        self.flux_data = np.loadtxt(fpath, dtype=FLUX_INFORMATION_DTYPE)
        self.flux_interpolation = UnivariateSpline(self.flux_data["energy"],
                                                   self.flux_data["events"])

    def _event_xsec(self, root_tupledata):
        weights = np.array(root_tupledata.weight)
        deltaE = np.mean(self.flux_data['energy'][1:] -
                         self.flux_data['energy'][:-1])
        energy_min = np.min(self.flux_data["energy"])
        energy_max = np.max(self.flux_data["energy"])
        total_flux_events = self.flux_interpolation.integral(
            energy_min, energy_max)
        n_files = len(self.root_pert_files)
        xsec = np.divide(total_flux_events * weights, deltaE * n_files)
        return xsec

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
        energy_min = np.min(self.flux_data["energy"])
        energy_max = np.max(self.flux_data["energy"])
        energy_phase_space = self.flux_interpolation.integral(
            energy_min, energy_max)
        xsec = self._event_xsec(
            root_tupledata
        ) * self.A  #xsec_per_nucleon * no_nucleons in the core
        inv_gen_flux = np.power(
            self.flux_interpolation(root_tupledata.lepIn_E), -1)
        phase_space = solid_angle * energy_phase_space
        env_factor = volume * SECONDS_PER_YEAR
        retval = env_factor * phase_space * inv_gen_flux * xsec * 10**-42 * target_density
        return retval

    @staticmethod
    def bjorken_x(roottuple_data):
        """
        Calculate Bjorken x variable for the GiBUU events

        Parameters
        ----------
            roottuple_data: awkward.highlevel.Array                
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
        qtmp = np.power(q, 2)
        q2 = qtmp[0, :] - np.sum(qtmp[1:4, :], axis=0)
        pq = q[0, :] * d.nuc_E - q[1, :] * d.nuc_Px - q[2, :] * d.nuc_Py - q[
            3, :] * d.nuc_Pz
        x = np.divide(-q2, 2 * pq)
        return np.array(x)

    @staticmethod
    def bjorken_y(roottuple_data):
        """
        Calculate Bjorken y scaling variable for the GiBUU events
        (Lab. frame)

        Parameters
        ----------
            roottuple_data: awkward.highlevel.Array                
        """
        d = roottuple_data
        y = 1 - np.divide(np.array(d.lepOut_E), np.array(d.lepIn_E))
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
        # Calculate additional information
        retval["xsec"] = self._event_xsec(retval)
        retval["Bx"] = GiBUUOutput.bjorken_x(retval)
        retval["By"] = GiBUUOutput.bjorken_y(retval)
        return retval


def write_detector_file(gibuu_output,
                        ofile="gibuu.aanet.root",
                        can=(0, 476.5, 403.4),
                        livetime=3.156e7,
                        propagate_tau=True):  # pragma: no cover
    """
    Convert the GiBUU output to a KM3NeT MC (AANET) file

    Parameters
    ----------
    gibuu_output: GiBUUOutput
        Output object which wraps the information from the GiBUU output files
    ofile: str
        Output filename
    can: tuple
        The can dimensions which are used to distribute the events 
        (z_min, z_max, radius)
    livetime: float
        The data livetime
    """
    import aa, ROOT

    aafile = ROOT.EventFile()
    aafile.set_output(ofile)
    mc_trk_id = 0

    def add_particles(particle_data, pos_offset, rotation, start_mc_trk_id,
                      timestamp):
        nonlocal aafile
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
            aafile.evt.mc_trks.push_back(trk)

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
    bjorkenx = event_data.Bx
    bjorkeny = event_data.By

    tau_secondaries = None
    if propagate_tau and abs(nu_type) == 16:
        from .propagation import propagate_lepton
        tau_secondaries = propagate_lepton(event_data, np.sign(nu_type) * 15)

    media = read_default_media_compositions()
    density = media["SeaWater"]["density"]
    symbol = mendeleev.element(gibuu_output.Z).symbol
    target = media["SeaWater"]["elements"][symbol]
    target_density = 1e3 * density * target[1]
    targets_per_volume = target_density * (1e3 * constants.Avogadro /
                                           target[0].atomic_weight)

    can_volume = np.pi * (can[1] - can[0]) * np.power(can[2], 2)
    w2 = gibuu_output.w2weights(can_volume, targets_per_volume, 4 * np.pi)

    for mc_event_id, event in enumerate(event_data):
        aafile.evt.clear()
        aafile.evt.id = mc_event_id
        aafile.evt.mc_run_id = mc_event_id
        # Weights
        aafile.evt.w.push_back(can_volume)  #w1 (can volume)
        aafile.evt.w.push_back(w2[mc_event_id])  #w2
        aafile.evt.w.push_back(-1.0)  #w3 (= w2*flux)
        # Vertex Position
        r = can[2] * np.sqrt(np.random.uniform(0, 1))
        phi = np.random.uniform(0, 2 * np.pi)
        pos_x = r * np.cos(phi)
        pos_y = r * np.sin(phi)
        pos_z = np.random.uniform(can[0], can[1])
        vtx_pos = np.array([pos_x, pos_y, pos_z])
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

        if not propagate_tau:
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
            aafile.evt.mc_trks.push_back(lep_out_trk)

        # bjorken_y = 1.0 - float(event.lepOut_E / event.lepIn_E)
        nu_in_trk.setusr('bx', bjorkenx[mc_event_id])
        nu_in_trk.setusr('by', bjorkeny[mc_event_id])
        nu_in_trk.setusr('ichan', ichan)
        nu_in_trk.setusr("cc", is_cc)

        aafile.evt.mc_trks.push_back(nu_in_trk)

        if tau_secondaries is not None:
            event_tau_sec = tau_secondaries[mc_event_id]
            add_particles(event_tau_sec, vtx_pos, R, mc_trk_id, timestamp)
            mc_trk_id += len(event_tau_sec.E)

        add_particles(event, vtx_pos, R, mc_trk_id, timestamp)

        aafile.write()

    del aafile
