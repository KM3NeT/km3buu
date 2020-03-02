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
from os.path import join, abspath, basename, isdir
from tempfile import NamedTemporaryFile, TemporaryDirectory
from thepipe.logger import get_logger

from . import IMAGE_NAME
from .config import Config

log = get_logger(basename(__file__))

INPUT_PATH = "/opt/buuinput2019/"
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
    method for run

    Parameters
    ----------
    jobcard: km3buu.JobCard
        The jobcard which should be run
    outdir: str
        The path to the directory the output should be written to
    """
    tmp_dir = TemporaryDirectory()
    outdir = abspath(outdir)
    log.info("Create temporary file for jobcard")
    jobcard_fpath = join(tmp_dir.name, "tmp.job")
    with open(jobcard_fpath, 'w') as f:
        f.write(str(jobcard))
    log.info("Create temporary file for associated runscript")
    script_fpath = join(tmp_dir.name, "run.sh")
    with open(script_fpath, 'w') as f:
        ctnt = GIBUU_SHELL.format(outdir, jobcard_fpath)
        f.write(ctnt)
    os.system("ls %s" % (tmp_dir.name))
    output = Client.execute(Config().gibuu_image_path,
                            ['/bin/sh', script_fpath],
                            bind=[outdir, tmp_dir.name],
                            return_result=True)
    msg = output['message']
    if isinstance(msg, str):
        log.info("GiBUU output:\n %s" % msg)
    else:
        log.info("GiBUU output:\n %s" % msg[0])
        log.error("GiBUU stacktrace:\n%s" % msg[1])
    return output["return_code"]
