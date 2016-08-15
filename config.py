import ConfigParser

_configParser = ConfigParser.RawConfigParser()
_configFilePath = "/home/ds381/seren3/._seren3_config.txt"
_configParser.read(_configFilePath)

def get_config():
    return _configParser

def get(section, key):
    return get_config().get(section, key)