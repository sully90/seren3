import numpy as np
from seren3.core.pymses_snapshot import PymsesSnapshot

class Simulation(object):
    '''
    Object to encapsulate a simulation directory and offer snapshot access
    '''
    def __init__(self, path):
        self.path = path
        self.store = {}

    def __str__(self):
        return "Simulation: %s" % self.path

    def __repr__(self):
        return self.__str__()

    def __getitem__(self, ioutput):
        return self.snapshot(ioutput)

    def __len__(self):
        return len(self.numbered_outputs)

    def __iter__(self):
        for ioutput in self.numbered_outputs:
            yield self[ioutput]

    def snapshot(self, ioutput, **kwargs):
        return PymsesSnapshot(self.path, ioutput, **kwargs)

    def redshift(self, z):
        '''
        Return ioutput of snapshot closest to this redshift
        '''
        idx = (np.abs(self.redshifts - z)).argmin()
        outputs = self.outputs
        iout = int(outputs[idx][-5:])
        return iout

    @property
    def redshifts(self):
        '''
        Returns a list of available redshifts
        '''
        redshifts = []
        for iout in self.numbered_outputs:
            info = "%s/output_%05i/info_%05i.txt" % (self.path, iout, iout)
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
        return sorted(numbered)

    @property
    def outputs(self):
        import glob
        from seren3.utils import string_utils

        outputs = glob.glob("%s/output_*" % self.path)
        outputs.sort(key=string_utils.natural_keys)

	result = []
	for o in outputs:
		if '/' in o:
			result.append( o.split('/')[-1] )
		else:
			result.append(o)
        return result

    @property
    def age(self):
        '''
        Returns age of simulation at last snapshot
        '''
        last_iout = self.numbered_outputs[-1]
        return self[last_iout].age

    @property
    def initial_redshift(self):
        '''
        Returns redshift of first output
        '''
        ifirst = self.numbered_outputs[0]
        info = '%s/output_%05i/info_%05i.txt' % (self.path, ifirst, ifirst)
        with open(info, 'r') as f:
            nline = 1
            while nline <= 10:
                line = f.readline()
                if(nline == 10):
                    aexp = np.float32(line.split("=")[1])
                nline += 1
        return (1. / aexp) - 1.

    def redshift_func(self, zmax=20.0, zmin=0.0, zstep=0.001):
        '''
        Returns an interpolation function that gives redshift as a function
        of age
        '''
        import cosmolopy.distance as cd

        init_z = self.initial_redshift
        cosmo = self[self.redshift(init_z)].cosmo
        del cosmo['z'], cosmo['aexp']

        func = cd.quick_redshift_age_function(zmax, zmin, zstep, **cosmo)
        return lambda age: func(age.in_units("s"))


    def age_func(self, zmax=20.0, zmin=0.0, zstep=0.001, return_inverse=False):
        '''
        Returns an interpolation function that gives age as a function
        of redshift
        '''
        import cosmolopy.distance as cd
        from seren3.array import SimArray

        init_z = self.initial_redshift
        cosmo = self[self.redshift(init_z)].cosmo
        del cosmo['z'], cosmo['aexp']

        func = cd.quick_age_function(zmax, zmin, zstep, return_inverse, **cosmo)
        return lambda z: SimArray(func(z), "s").in_units("Gyr")

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
        from seren3 import config
        rockstar_base = config.get("halo", "rockstar_base")

        if os.path.isdir("%s/%s/" % (self.path, rockstar_base)):
            import glob

            info = {}
            files = glob.glob("%s/%s/out_*.list" % (self.path, rockstar_base))
            for fname in files:
                out_num = int( filter(str.isdigit, fname.replace(rockstar_base, '')) )
                with open(fname, "r") as f:
                    while True:
                        line = f.readline()
                        if line.startswith('#a'):
                            spl = line.split('=')
                            aexp = float(spl[1])
                            info[out_num] = aexp
                            break
            # Write to file
            keys = sorted(info.keys())
            with open("%s/%s/info_rockstar.txt" % (self.path, rockstar_base), "w") as f:
                for i in keys:
                    line = "%i\t%f\n" % (i, info[i])
                    f.write(line)
            return info

        else:
            raise IOError("Could not locate Rockstar directory")
