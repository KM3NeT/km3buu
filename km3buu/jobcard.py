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
    """
    A object to manage GiBUU jobcard properties and format them

    Parameters
    ----------
    input_path: str
        The input path pointing to the GiBUU lookup data which should be used
    """
    def __init__(self, input_path=INPUT_PATH):
        self.input_path = input_path
        self._groups = {}
        # Default [input]
        self.add_property("input", "path_to_input", input_path)
        self.add_property("input", "numTimeSteps", 0)
        self.add_property("input", "numEnsembles", 100000)
        self.add_property("input", "delta_T", 0.2)
        self.add_property("input", "num_runs_SameEnergy", 1)
        # Default [target]
        self.add_property("target", "Z", 1)
        self.add_property("target", "A", 1)

    def set_target(self, Z, A):
        """
        Set the interaction target nucleus

        Parameters
        ----------
        A: int
            Number of nucleons in the nucleus
        Z: int  
            Number of protons in the nucleus
        """
        self.add_property("target", "Z", Z)
        self.add_property("target", "A", A)

    @property
    def timesteps(self):
        """
        input/numTimeSteps property
        """
        return self._groups["input"]["numTimeSteps"]

    @timesteps.setter
    def timesteps(self, timesteps):
        self._groups["input"]["numTimeSteps"] = timesteps

    def add_property(self, group, name, value):
        """ Add a property to the jobcard

        Parameters
        ----------
        group: str
            The parameter group where the property is incorporated
        name: str
            The property name
        value: 
            The property value
        """
        if not group in self._groups.keys():
            self._groups[group] = {}
        self._groups[group][name] = value

    def __str__(self):
        retval = ""
        for group, attrs in self._groups.items():
            if len(self._groups[group]) == 0:
                continue
            retval += "&%s\n" % group
            for attr, value in attrs.items():
                if type(value) is str:
                    retval += "\t%s = '%s'\n" % (attr, value)
                elif type(value) is bool:
                    retval += "\t%s = .%s.\n" % (attr, value)
                else:
                    retval += "\t%s = %s\n" % (attr, value)
            retval += "/\n\n"
        return retval


class NeutrinoJobcard(Jobcard):
    """
    Jobcard object for neutrino interactions 

    Parameters
    ----------
    input_path: str
        The input path pointing to the GiBUU lookup data which should be used
    """
    def __init__(self, input_path=INPUT_PATH):
        super(NeutrinoJobcard, self).__init__(input_path)
        ## Default [input]
        self.add_property("input", "eventtype", 5)
        self.add_property("input", "localEnsemble", True)
