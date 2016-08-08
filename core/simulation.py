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

    def z_reion(self, thresh=0.999, return_vw_z=False):
        '''
        Return the redshift of reionization, if the xHII_reion_history.p
        table exists
        '''
        import os
        if os.path.isfile("%s/xHII_reion_history.p" % self.path):
            from seren3.scripts.mpi import reion_history
            from seren3.utils import first_above

            table = reion_history.load_xHII_table(self.path)
            vw = np.zeros(len(table))
            z = np.zeros(len(table))

            for i in range(len(table)):
                vw[i] = table[i+1]["volume_weighted"]
                z[i] = table[i+1]["z"]
            eor_idx = first_above(thresh, vw)

            if return_vw_z:
                return z[eor_idx], z, vw
            return z[eor_idx]

    def write_rockstar_info(self):
        '''
        If a rockstar directory exists, writes a rockstar_info.txt file
        with out_list numbers against aexp
        '''
        import os
        if os.path.isdir("%s/rockstar/" % self.path):
            import glob

            info = {}
            files = glob.glob("%s/rockstar/out_*.list" % self.path)
            for fname in files:
                out_num = int( filter(str.isdigit, fname) )
                with open(fname, "r") as f:
                    while True:
                        line = f.readline()
                        if line.startswith('#a'):
                            spl = line.split('=')
                            aexp = float(spl[1])
                            info[out_num] = aexp
                            break
            # Write to file
            with open("%s/rockstar/info_rockstar.txt" % self.path, "w") as f:
                for i in range(1, len(info)+1):
                    line = "%i\t%f\n" % (i, info[i])
                    f.write(line)
            return info

        else:
            raise IOError("Could not locate Rockstar directory")