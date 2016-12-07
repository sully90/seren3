'''
This file provides basic I/O access to radgpu_%05i.out%05i files
Note: boundary regions have not been accounted for, this is just for I/O as is on disk
'''
import numpy as np
from seren3.utils.f90.fortranfile import FortranFile

class RadGPUReader(object):
    def __init__(self, path, ioutput):
        self.path = path
        self.ioutput = ioutput
        self.info = self._read_rad_hdr()

    def _open_fortran_file(self, icpu):
        '''
        Returns a fortranfile.FortranFile for this cpu
        '''
        fname = "%s/radgpu_%05i.out%05i" % (self.output_filename, self.ioutput, icpu)
        return FortranFile(fname)

    def __getitem__(self, field):
        '''
        Field can be rho (photon density) [photons / m^3] or photon flux [photons / m^2 / s]
        Note: boundary regions have not been accounted for, this is just for I/O as is on disk
        '''
        result = None
        if field == 'rho':
            result = np.zeros(self.nmem)
        elif field == 'flux':
            result = np.zeros(self.nmemf)
        else:
            raise Exception("Unknown field: %s" % field)

        for i in range(self.info["ncpu"]):
            ff = self._open_fortran_file(i+1)
            # skip grid dims
            ff.readInts()

            if field == 'flux':
                # skip rho
                ff.readRecord()

            # read record
            rec = ff.readReals(prec='d')
            # replace nan with 0.
            rec[np.isnan(rec)] = 0.
            result += rec

        return result


    def _read_rad_hdr(self):
        '''
        Reads header information from rad_%05i.out00001 file
        '''
        fname = "%s/rad_%05i.out00001" % (self.output_filename, self.ioutput)
        ff = FortranFile(fname)
        ncpu = ff.readInts()
        nradvar = ff.readInts()
        ndim = ff.readInts()
        nlevelmax = ff.readInts()
        nboundary = ff.readInts()

        ff.close()
        info = {"ncpu" : int(ncpu), "nradvar" : nradvar, "ndim" : int(ndim), \
                    "nlevelmax" : int(nlevelmax), "nboundary" : int(nboundary)}

        # Read grid dims from radgpu_%05i.out00001
        fname = "%s/radgpu_%05i.out00001" % (self.output_filename, self.ioutput)
        ff = FortranFile(fname)
        rad_grid_dims = ff.readInts()
        ff.close()

        info["grid_dims"] = rad_grid_dims
        return info


    @property
    def nmem(self):
        '''
        Number of elements for scalar arrays
        '''
        ncellx, ncelly, ncellz = self.info["grid_dims"]
        nbnd = self.info["nboundary"] + 1
        return (ncellx + 2*nbnd)*(ncelly + 2*nbnd)*(ncellz + 2*nbnd)


    @property
    def nmemf(self):
        '''
        Number of elements for vector (flux) arrays
        '''
        return self.nmem*3


    @property
    def output_filename(self):
        '''
        Reutns formatted path to output_%05i
        '''
        return "%s/output_%05i/" % (self.path, self.ioutput)