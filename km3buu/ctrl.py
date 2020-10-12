# Filename: ctrl.py
"""
Run and control tools for GiBUU

"""

__author__ = "Johannes Schumann"
__copyright__ = "Copyright 2020, Johannes Schumann and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Johannes Schumann"
__email__ = "jschumann@km3net.de"
__status__ = "Development"

import os
from spython.main import Client
from os.path import join, abspath, basename, isdir, isfile
from tempfile import NamedTemporaryFile, TemporaryDirectory
from thepipe.logger import get_logger

from . import IMAGE_NAME
from .config import Config
from .jobcard import Jobcard, read_jobcard
from .environment import is_singularity_version_greater, MIN_SINGULARITY_VERSION

log = get_logger(basename(__file__))

if not is_singularity_version_greater(
        MIN_SINGULARITY_VERSION):  # pragma: no cover
    log.error("Singularity version lower than %s" % MIN_SINGULARITY_VERSION)
    raise OSError("Singularity version below %s" % MIN_SINGULARITY_VERSION)

GIBUU_SHELL = """
#!/bin/bash

export LD_LIBRARY_PATH=/usr/local/lib

if [ -z "$CONTAINER_GIBUU_EXEC+x" ];
then
    echo "No GIBUU executable provided via CONTAINER_GIBUU_EXEC";
    exit 1
fi;

cd {0};

$CONTAINER_GIBUU_EXEC < {1};
"""


def run_jobcard(jobcard, outdir, fluxfile=None):
    """
    Method for run

    Parameters
    ----------
    jobcard: str, km3buu.JobCard
        The jobcard which should be run, which can be an instance
        of a jobcard object or a path to a jobcard
    outdir: str 
        The path to the directory the output should be written to.
    fluxfile: str
        Filepath of the custom flux file if initNeutrino/nuExp = 99
    """
    input_dir = TemporaryDirectory()
    outdir = abspath(outdir)
    log.info("Create temporary file for jobcard")
    jobcard_fpath = join(input_dir.name, "tmp.job")

    if isinstance(jobcard, str) and isfile(jobcard):
        jobcard = read_jobcard(jobcard)

    if "neutrino_induced" in jobcard and "nuexp" in jobcard[
            "neutrino_induced"] and jobcard["neutrino_induced"]["nuexp"] == 99:
        if fluxfile is None or not isfile(fluxfile):
            raise IOError("Fluxfile not found!")
        tmp_fluxfile = join(input_dir.name, basename(fluxfile))
        os.system("cp %s %s" % (fluxfile, tmp_fluxfile))
        log.info("Set FileNameFlux to: %s" % tmp_fluxfile)
        jobcard["neutrino_induced"]["FileNameflux"] = tmp_fluxfile
    if isinstance(jobcard, Jobcard):
        with open(jobcard_fpath, "w") as f:
            f.write(str(jobcard))
    else:
        log.error("No valid jobcard reference given: %s" % jobcard)
    log.info("Create temporary file for associated runscript")
    script_fpath = join(input_dir.name, "run.sh")
    with open(script_fpath, "w") as f:
        ctnt = GIBUU_SHELL.format(outdir, jobcard_fpath)
        f.write(ctnt)
    output = Client.execute(
        Config().gibuu_image_path,
        ["/bin/sh", script_fpath],
        bind=[outdir, input_dir.name],
        return_result=True,
    )
    msg = output["message"]
    if isinstance(msg, str):
        log.info("GiBUU output:\n %s" % msg)
    else:
        log.info("GiBUU output:\n %s" % msg[0])
        log.error("GiBUU stacktrace:\n%s" % msg[1])
    return output["return_code"]
