import ConfigParser

def get_seren3_abspath():
    import seren3, os
    path = os.path.abspath(seren3.__file__).replace("__init__.pyc", "").replace("__init__.py", "")
    return path

def get_config_fname():
    import os

    seren3_dir = get_seren3_abspath()
    if (os.path.isfile("%s/._seren3_config.txt" % seren3_dir) is False):
        return "%s/._default_seren3_config.txt" % seren3_dir
    else:
        return "%s/._seren3_config.txt" % seren3_dir

_configParser = ConfigParser.RawConfigParser()
_configFilePath = get_config_fname()
_configParser.read(_configFilePath)

def get_config():
    return _configParser

def get(section, key):
    value = get_config().get(section, key) 
    if value in ["True", "False"]:
        return value == "True"
    else:
        return value

BASE_UNITS = {"length" : "m", "velocity" : "m s**-1", "mass" : "kg"}  # DO NOT CHANGE