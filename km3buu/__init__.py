from .__version__ import version

from os.path import isfile, abspath, join, dirname

MODULE_PATH = abspath(join(dirname(__file__), ".."))
IMAGE_PATH = join(MODULE_PATH, "GiBUU.simg")

if not isfile(IMAGE_PATH):
    raise EnvironmentError(
        "GiBUU image was not found at %s; please run `make build` or `make buildremote`"
        % IMAGE_PATH)
