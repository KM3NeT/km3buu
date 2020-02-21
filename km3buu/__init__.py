from os.path import isfile, abspath, join, dirname

MODULE_PATH = abspath(join(dirname(__file__), ".."))
IMAGE_PATH = join(MODULE_PATH, "GiBUU.simg")

if not isfile(IMAGE_PATH):
    errmsg = "GiBUU image not found at %s;"
    errmsg += "please run `make build` or `make buildremote`"
    raise EnvironmentError(errmsg % IMAGE_PATH)
