# Filename: jobcard.py
"""
Tools for creation of GiBUU jobcards

"""

__author__ = "Johannes Schumann"
__copyright__ = "Copyright 2020, Johannes Schumann and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Johannes Schumann"
__email__ = "jschumann@km3net.de"
__status__ = "Development"

INPUT_PATH = "/opt/buuinput2019/"


class Jobcard(object):
    def __init__(self, input_path=INPUT_PATH):
        self.input_path = input_path
        self._groups = { 
                "input": {"path_to_input": input_path}    
                }

    def add_property(self, group, name, value)
        self._groups[group][name] = value

    def __str__(self):
        retval = ""
        for group, attrs in self._groups.items():
            retval += "&%s\n" % group
            for attr, value in attrs.items():
                if type(value) is str:
                    retval += "\t%s = '%s'\n" % (attr, value)
                elif type(value) is bool:
                    retval += "\t%s = .%s.\n" % (attr, value)
                else:
                    retval += "\t%s = %s\n" % (attr, value)
            retval += "/\n"
        return retval



class NeutrinoJobcard(Jobcard):
    def __init__(self, input_path=INPUT_PATH):
        super(NeutrinoJobcard, self).__init__(input_path)
