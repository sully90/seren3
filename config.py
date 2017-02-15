import ConfigParser
_configParser = ConfigParser.RawConfigParser()
_configFilePath = "/home/ds381/seren3/._seren3_config.txt"
_configParser.read(_configFilePath)

def get_config():
    return _configParser

def get(section, key):
    value = get_config().get(section, key) 
    if value in ["True", "False"]:
        return value == "True"
    else:
        return value

# Module wide base unit system
class _BASE_UNITS(object):
    def __init__(self):
        self._BASE_UNITS = {"length" : "m", "velocity" : "m s**-1", "mass" : "kg"}  # DO NOT CHANGE

    def __getitem__(self, item):
        return self._BASE_UNITS[item]

    def __contains__(self, item):
        return item in self._BASE_UNITS

    def __len__(self):
        return len(self._BASE_UNITS)

    def __str__(self):
        return self._BASE_UNITS.__str__()

    def __repr__(self):
        return self._BASE_UNITS.__repr__()

    def keys(self):
        return self._BASE_UNITS.keys()

BASE_UNITS = _BASE_UNITS()