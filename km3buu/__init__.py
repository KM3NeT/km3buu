from .__version__ import version

from os.path import isfile, abspath, join, dirname

image_path = abspath(join(dirname(__file__), "..", "GiBUU.simg"))
if not isfile(image_path):
    raise EnvironmentError(
        "GiBUU image was not found at %s; please run `make build` or `make buildremote`"
        % image_path)
