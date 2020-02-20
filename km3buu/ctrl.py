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
from os.path import join, abspath
from tempfile import NamedTemporaryFile
from . import MODULE_PATH, IMAGE_PATH
from thepipe.logger import get_logger

log = get_logger(__file__)

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
    tmp_dir = join(MODULE_PATH, 'tmp')
    os.makedirs(tmp_dir, exist_ok=True)
    outdir = abspath(outdir)
    log.info("Create temporary file for jobcard")
    tmpfile_jobcard = NamedTemporaryFile(suffix='.job', dir=tmp_dir)
    with open(tmpfile_jobcard.name, 'w') as f:
        f.write(str(jobcard))
    log.info("Create temporary file for associated runscript")
    tmpfile_shell = NamedTemporaryFile(suffix='.sh', dir=tmp_dir)
    with open(tmpfile_shell.name, 'w') as f:
        ctnt = GIBUU_SHELL.format(outdir, tmpfile_jobcard.name)
        f.write(ctnt)
    output = Client.execute(IMAGE_PATH, ['/bin/sh', tmpfile_shell.name],
                            bind=[outdir, tmp_dir])
    log.info("GiBUU exited with return code {0}".format(output["return_code"]))
    return output["return_code"]
