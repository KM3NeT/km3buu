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
from .jobcard import Jobcard

log = get_logger(basename(__file__))

GIBUU_SHELL = """
#!/bin/bash

if [ -z "$CONTAINER_GIBUU_EXEC+x" ];
then
    echo "No GIBUU executable provided via CONTAINER_GIBUU_EXEC";
    exit 1
fi;

cd {0};

$CONTAINER_GIBUU_EXEC < {1};
"""


def run_jobcard(jobcard, outdir):
    """
    Method for run

    Parameters
    ----------
    jobcard: str, km3buu.JobCard
        The jobcard which should be run, which can be an instance
        of a jobcard object or a path to a jobcard
    outdir: str 
        The path to the directory the output should be written to.
    """
    script_dir = TemporaryDirectory()
    outdir = abspath(outdir)
    log.info("Create temporary file for jobcard")
    jobcard_fpath = join(script_dir.name, "tmp.job")
    if isinstance(jobcard, Jobcard):
        with open(jobcard_fpath, 'w') as f:
            f.write(str(jobcard))
    elif isfile(jobcard):
        os.system("cp %s %s" % (jobcard, jobcard_fpath.name))
    else:
        log.error("No valid jobcard reference given: %s" % jobcard)
    log.info("Create temporary file for associated runscript")
    script_fpath = join(script_dir.name, "run.sh")
    with open(script_fpath, 'w') as f:
        ctnt = GIBUU_SHELL.format(outdir, jobcard_fpath)
        f.write(ctnt)
    os.system("ls %s" % (script_dir.name))
    output = Client.execute(Config().gibuu_image_path,
                            ['/bin/sh', script_fpath],
                            bind=[outdir, script_dir.name],
                            return_result=True)
    msg = output['message']
    if isinstance(msg, str):
        log.info("GiBUU output:\n %s" % msg)
    else:
        log.info("GiBUU output:\n %s" % msg[0])
        log.error("GiBUU stacktrace:\n%s" % msg[1])
    return output["return_code"]
