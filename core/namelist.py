import numpy as np

def _is_number(str):
    '''
    Dirty check to support scientific notation
    '''
    try:
        float(str)
        return True
    except ValueError:
        return False
    except TypeError:
        return False


def load_nml(snapshot):
    '''
    Read the namelist file
    '''
    import os
    fname = '%s/nml.nml' % snapshot.path
    if os.path.isfile(fname) is False:
        # Look for *.nml and take the first file
        import glob
        files = glob.glob("%s/*.nml" % snapshot.path)
        if len(files) > 0:
            fname = files[0]
        else:
            raise IOError("File: %s not found" % fname)
    
    nml_dict = {}
    block = None
    with open(fname, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if line.startswith('&'):
                # New block
                block = line[1:]
                nml_dict[block] = {}
            else:
                if '!' in line:
                    line = line.split('!')[0]
                if '=' in line:
                    # Param
                    try:
                        var, values = line.split('=')
                        if var.startswith('!'):
                            # Commented out, skip
                            continue
                        values = values.split(',')
                        if len(values[0].split(' ')) > 1:
                            # Keep only first
                            nml_dict[block][var] = values[0].split(' ')[0]
                        elif len(values) == 1:
                            nml_dict[block][var] = values[0]
                        else:
                            nml_dict[block][var] = values
                    except ValueError:
                        print 'NML Failed on line: ', line
    return Namelist(nml_dict)


class NML(object):
    RUN_PARAMS = 'RUN_PARAMS'
    PHYSICS_PARAMS = 'PHYSICS_PARAMS'
    HYDRO_PARAMS = 'HYDRO_PARAMS'
    OUTPUT_PARAMS = 'OUTPUT_PARAMS'
    INIT_PARAMS = 'INIT_PARAMS'
    AMR_PARAMS = 'AMR_PARAMS'
    POISSON_PARAMS = 'POISSON_PARAMS'
    REFINE_PARAMS = 'REFINE_PARAMS'
    RT_PARAMS = 'RT_PARAMS'
    RT_GROUPS = 'RT_GROUPS'


class Namelist():
    '''
    Class to wrap the namelist dict
    '''
    def __init__(self, nml_dict):
        self._nml = nml_dict
        self.NML = NML

    def __getitem__(self, key):
        param_dict = self._nml[key]
        for k in param_dict.keys():
            vals = param_dict[k]
            if isinstance(vals, list):
                numbers = True
                for v in vals:
                    if not _is_number(v):
                        numbers = False
                        break
                if numbers:
                    param_dict[k] = np.array(vals, dtype=float)
            param_dict[k] = float(vals) if _is_number(vals) else vals
        return self._nml[key]

    def __str__(self):
        return str(self._nml.keys())

    def __repr__(self):
        return self.__str__()

    def __contains__(self, item):
        return item in self._nml

    def get_param(self, block, key):
        vals = self._nml[block][key]
        if isinstance(vals, list):
            numbers = True
            for v in vals:
                if not _is_number(v):
                    numbers = False
                    break
            if numbers:
                return np.array(vals, dtype=float)
        return float(vals) if _is_number(vals) else vals