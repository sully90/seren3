import numpy as np
from seren3.core.pymses_snapshot import PymsesSnapshot

class Simulation(object):
    '''
    Object to encapsulate a simulation directory and offer snapshot access
    '''
    def __init__(self, path):
        self.path = path

    def __str__(self):
        return "Simulation: %s" % self.path

    def __repr__(self):
        return self.__str__()

    def __getitem__(self, ioutput):
        return self.snapshot(ioutput)

    def __iter__(self):
        for ioutput in self.numbered_outputs:
            yield self[ioutput]

    def snapshot(self, ioutput):
        return PymsesSnapshot(self.path, ioutput)

    def redshift(self, z):
        '''
        Return ioutput of snapshot closest to this redshift
        '''
        idx = (np.abs(self.redshifts - z)).argmin()
        outputs = self.outputs
        iout = int(outputs[idx][-6:-1])
        return iout

    @property
    def redshifts(self):
        '''
        Returns a list of available redshifts
        '''
        redshifts = []
        outputs = self.outputs
        for output in outputs:
            s = output[:-1]
            info = '%s/info_%s.txt' % (s, s[-5:])
            f = open(info, 'r')
            nline = 1
            while nline <= 10:
                line = f.readline()
                if(nline == 10):
                    aexp = np.float32(line.split("=")[1])
                nline += 1
            redshift = 1.0 / aexp - 1.0
            if (redshift >= zmin) and (redshift < zmax):
                redshifts.append(float(redshift))

        return np.array(redshifts)

    @property
    def numbered_outputs(self):
        import re

        outputs = self.outputs
        numbered = np.zeros(len(outputs))
        for i in range(len(outputs)):
            result = re.findall(r'\d+', outputs[i])[0]
            ioutput = int(result)
            numbered[i] = ioutput
        return numbered

    @property
    def outputs(self):
        import glob
        from seren3.utils import string_utils

        outputs = glob.glob("%s/output_*/" % self.path)
        outputs.sort(key=string_utils.natural_keys)
        return outputs

    @property
    def initial_redshift(self):
        '''
        Returns redshift of first output
        '''
        output = self.outputs[0]
        s = output[:-1]
        info = '%s/info_%s.txt' % (s, s[-5:])
        with open(info, 'r') as f:
            nline = 1
            while nline <= 10:
                line = f.readline()
                if(nline == 10):
                    aexp = np.float32(line.split("=")[1])
                nline += 1
        return (1. / aexp) - 1.