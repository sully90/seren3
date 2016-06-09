import numpy as np
from seren3 import config
from seren3.halos import Halo, HaloCatalogue
import logging

logger = logging.getLogger('seren3.halos.halos')

class AHFCatalogue(HaloCatalogue):
    '''
    Class to handle catalogues produced by AHF.
    '''

    # assume file structure like this

# ID(1)  hostHalo(2)     numSubStruct(3) Mvir(4) npart(5)        Xc(6)
# Yc(7)   Zc(8)   VXc(9)  VYc(10) VZc(11) Rvir(12)        Rmax(13)
# r2(14)  mbp_offset(15)  com_offset(16)  Vmax(17)        v_esc(18)
# sigV(19)        lambda(20)      lambdaE(21)     Lx(22)  Ly(23)  Lz(24)
# b(25)   c(26)   Eax(27) Eay(28) Eaz(29) Ebx(30) Eby(31) Ebz(32) Ecx(33)
# Ecy(34) Ecz(35) ovdens(36)      nbins(37)       fMhires(38)     Ekin(39)
# Epot(40)        SurfP(41)       Phi0(42)        cNFW(43)
# n_gas(44)       M_gas(45)       lambda_gas(46)  lambdaE_gas(47)
# Lx_gas(48)      Ly_gas(49)      Lz_gas(50)      b_gas(51)
# c_gas(52)       Eax_gas(53)     Eay_gas(54)     Eaz_gas(55)
# Ebx_gas(56)     Eby_gas(57)     Ebz_gas(58)     Ecx_gas(59)
# Ecy_gas(60)     Ecz_gas(61)     Ekin_gas(62)    Epot_gas(63)
# n_star(64)      M_star(65)      lambda_star(66) lambdaE_star(67)
# Lx_star(68)     Ly_star(69)     Lz_star(70)     b_star(71)
# c_star(72)      Eax_star(73)    Eay_star(74)    Eaz_star(75)
# Ebx_star(76)    Eby_star(77)    Ebz_star(78)    Ecx_star(79)
# Ecy_star(80)    Ecz_star(81)    Ekin_star(82)   Epot_star(83)
    halo_type = np.dtype([('id', np.int64), ('hosthalo', np.int64), ('numsubstruct', np.int64), ('mvir', 'f'),
                          ('num_p', np.int64), ('pos',
                                                'f', 3), ('vel', 'f', 3),
                          ('rvir', 'f'), ('rmax', 'f'), ('r2',
                                                         'f'), ('mpb_offset', 'f'),
                          ('com_offset', 'f'), ('v_max',
                                                'f'), ('v_esc', 'f'), ('sigv', 'f'),
                          ('bullock_spin', 'f'), ('spin', 'f'), ('l',
                                                                 'f', 3), ('b', 'f'), ('c', 'f'),
                          ('ea', 'f', 3), ('eb', 'f',
                                           3), ('ec', 'f', 3), ('ovdens', 'f'),
                          ('nbins', np.int64), ('fmhires', 'f'), ('ekin', 'f'),
                          ('epot', 'f'), ('surfp', 'f'), ('phiO',
                                                          'f'), ('cnfw', 'f'), ('n_gas', np.int64),
                          ('m_gas', 'f'), ('bullock_spin_gas',
                                           'f'), ('spin_gas', 'f'),
                          ('l_gas', 'f', 3), ('b_gas', 'f'), ('c_gas', 'f'),
                          ('ea_gas', 'f', 3), ('eb_gas',
                                               'f', 3), ('ec_gas', 'f', 3,),
                          ('ekin_gas', 'f'), ('epot_gas',
                                              'f'), ('n_star', np.int64),
                          ('m_star', 'f'), ('bullock_spin_star',
                                            'f'), ('spin_star', 'f'),
                          ('l_star', 'f', 3), ('b_star', 'f'), ('c_star', 'f'),
                          ('ea_star', 'f', 3), ('eb_star',
                                                'f', 3), ('ec_star', 'f', 3,),
                          ('ekin_star', 'f'), ('epot_star', 'f')])

    units = {'id': 'dimensionless',
             'hosthalo': 'dimensionless',
             'numsubstruct': 'dimensionless',
             'mvir': 'Msun / h',
             'num_p': 'dimensionless',
             'pos': 'kpccm / h',
             'vel': 'km / s',
             'rvir': 'kpccm / h',
             'rmax': 'kpccm / h',
             'r2': 'kpccm / h',
             'mpb_offset': 'kpccm / h',
             'com_offset': 'kpccm / h',
             'v_max': 'km / s',
             'v_esc': 'km / s',
             'sigv': 'km / s',
             'bullock_spin': 'dimensionless',
             'spin': 'dimensionless',
             'l': 'dimensionless',
             'b_to_a': 'kpccm / h',
             'c_to_a': 'kpccm / h',
             'ea': 'dimensionless',
             'eb': 'dimensionless',
             'ec': 'dimensionless',
             'ovdens': 'dimensionless',
             'nbins': 'dimensionless',
             'fmhires': 'dimensionless',
             'ekin': 'Msun / h (km / sec)**2',
             'epot': 'Msun / h (km / sec)**2',
             'surfp': 'Msun / h (km / sec)**2',
             'phiO': '(km / sec)**2',
             'cnfw': 'dimensionless',
             'm_gas': 'Msun / h',
             'bullock_spin_gas': 'dimensionless',
             'spin_gas': 'dimensionless',
             'l_gas': 'dimensionless',
             'b_to_a_gas': 'kpccm / h',
             'c_to_a_gas': 'kpccm / h',
             'ea_gas': 'dimensionless',
             'eb_gas': 'dimensionless',
             'ec_gas': 'dimensionless',
             'n_gas': 'dimensionless',
             'ekin_gas': 'Msun / h (km / sec)**2',
             'epot_gas': 'Msun / h (km / sec)**2',
             'm_star': 'Msun / h',
             'bullock_spin_star': 'dimensionless',
             'spin_star': 'dimensionless',
             'l_star': 'dimensionless',
             'b_to_a_star': 'kpccm / h',
             'c_to_a_star': 'kpccm / h',
             'ea_star': 'dimensionless',
             'eb_star': 'dimensionless',
             'ec_star': 'dimensionless',
             'n_star': 'dimensionless',
             'ekin_star': 'Msun / h (km / sec)**2',
             'epot_star': 'Msun / h (km / sec)**2',
             }

    def __init__(self, pymses_snapshot, filename=None, **kwargs):
        super(AHFCatalogue, self).__init__(
            pymses_snapshot, "AHF", filename=filename, **kwargs)

    ################## IMPLEMENT ABSTRACT FUNCTIONS ##################

    def gadget_format_exists(self):
        '''
        Checks if ramses2gadget has been ran
        '''
        import glob
        path = "%s/output_%05d/" % (self.base.path, self.base.ioutput)

        return len(glob.glob("%s/ramses2gadget*" % path)) > 0

    def run(self, **kwargs):
        '''
        Run ramses2gadget then AHF
        '''
        import subprocess, os
        from seren2.utils.sge import ncpu
        from seren2.utils import which
        r2g = which("ramses2gadget")
        ahf = which("AHF-v1.0-084")

        tasks = []

        # Check if GADGET data exists
        print 'GADGET format exists: ', self.gadget_format_exists()
        if self.gadget_format_exists() is False:
            r2g_mode = kwargs.pop("r2g_mode", "g")  # default to sim with gas
            # Build the ramses2gadget input_dir
            r2g_input_dir = "%s/output_%05d/" % (self.base.path, self.base.ioutput)

            # Build exe string
            r2g_exe = "{EXE} -{MODE} {INPUT_DIR} | tee {INPUT_DIR}/r2g.log".format(EXE=r2g, MODE=r2g_mode, INPUT_DIR=r2g_input_dir)
            tasks.append(r2g_exe)

        # Repeat for AHF
        ahf_path = "%s/AHF/%03d/" % (self.base.path, self.base.ioutput)
        ahf_input_fname = "%s/ahf.input" % ahf_path
        if os.path.isdir("%s/halos" % ahf_path) is False:
            os.mkdir("%s/halos" % ahf_path)

        ahf_exe = "{EXE} {FNAME}".format(EXE=ahf, FNAME=ahf_input_fname)
        tasks.append(ahf_exe)

        # Run the tasks
        NSLOTS = kwargs.get("NSLOTS", int(ncpu() / 2.))
        for task in tasks:
            mpi_task = "mpirun -np {NSLOTS} {EXE}".format(NSLOTS=NSLOTS, EXE=task)
            print mpi_task
            subprocess.check_output(mpi_task, shell=True)

        subprocess.check_output("cat {AHF_PATH}/halos/*_halos > {AHF_PATH}/halos/all_halos".format(AHF_PATH=ahf_path), shell=True)
        super(AHFCatalogue, self).__init__(self.base, "AHF", filename=None, **kwargs)
        return True

    @property
    def ahf_path(self):
        return "%s/AHF/%03d/halos/" % (self.base.path, self.base.ioutput)

    def get_boxsize(self, **kwargs):
        '''
        Returns the boxsize, according to AHF, in Mpccm/h
        '''
        import glob
        list_files = glob.glob("%s/*.log" % self.ahf_path)
        with open(list_files[0], 'r') as f:
            while True:
                l = f.readline()
                if l.startswith('simu.boxsize'):
                    box_size = float(l.split(':')[1])
                    return box_size

    def can_load(self, **kwargs):
        '''
        Check if hlist files exist
        '''
        import os
        if os.path.isfile('%s/all_halos' % self.ahf_path) is False:
            path = "%s/AHF/%03d/" % (self.base.path, self.base.ioutput)
            if os.path.isfile("%s/ahf.input" % path) is False:
                if self.write_cfg(**kwargs):
                    print "AHFCatalogue wrote a partial(!) config file."
                else:
                    raise Exception("AHFCatalogue unable to write config file!")
            else:
                print "AHFCatalogue not found - ahf.input already written!"
            return False
        return True

    def get_filename(self, **kwargs):
        return "%s/all_halos" % self.ahf_path

    def load(self, within_r=None, center=np.array([0.5, 0.5, 0.5]), **kwargs):
        # Ensures file is closed at the end. If within_r is specified, it must be in code units
        with open(self.filename, 'r') as f:
            haloprops = np.loadtxt(f, dtype=self.halo_type, comments="#")
            if within_r:
                d = np.array([np.sqrt( (center[0] - (h['pos'][0]/self.boxsize/1.e3))**2 + \
                    (center[1] - (h['pos'][1]/self.boxsize/1.e3))**2 + \
                    (center[2] - (h['pos'][2]/self.boxsize/1.e3))**2 ) for h in haloprops])
                idx = np.where(d <= within_r)
                haloprops = haloprops[idx]

            self._nhalos = len(haloprops)
            self._haloprops = haloprops

            #for h in xrange(self._nhalos):
            #    self._halos[h] = Halo(self._haloprops[h]['id'], self, self._haloprops[h])

    def _get_halo(self, item):
        haloprops = self._haloprops[item]
        return Halo(haloprops['id'], self, haloprops)

    def write_cfg(self, **kwargs):
        '''
        Internal function to write an appropriate AHF input file
        '''
        import os

        path = "%s/AHF/%03d/" % (self.base.path, self.base.ioutput)
        if os.path.isdir(path) is False:
            if os.path.isdir("%s/AHF/" % self.base.path) is False:
                os.mkdir("%s/AHF/" % self.base.path)
            os.mkdir(path)

        with open("%s/ahf.input" % path, "w") as f:
            f.write("[AHF]\n")
            f.write("ic_filename       = %s/output_%05d/ramses2gadget_%03d.\n" % (self.base.path, self.base.ioutput, self.base.ioutput))
            f.write("ic_filetype       = 61\n")  # GADGET
            f.write("outfile_prefix    = %s/AHF/%03d/halos/ahf_\n" % (self.base.path, self.base.ioutput))

            LgridDomain = kwargs.pop("LgridDomain", 128)
            LgridMax = kwargs.pop("LgridMax", 16777216)
            NperDomCell = kwargs.pop("NperDomCell", 5.0)
            NperRefCell = kwargs.pop("NperRefCell", 5.0)
            VescTune = kwargs.pop("VescTune", 1.5)
            NminPerHalo = kwargs.pop("NminPerHalo", 20)
            RhoVir = kwargs.pop("RhoVir", 0)
            Dvir = kwargs.pop("Dvir", 200)
            MaxGatherRad = kwargs.pop("MaxGatherRad", 3.0)
            LevelDomainDecomp = kwargs.pop("LevelDomainDecomp", 6)
            NcpuReading = kwargs.pop("NcpuReading", 1)

            GADGET_LUNIT = kwargs.pop("GADGET_LUNIT", 1e-3)
            GADGET_MUNIT = kwargs.pop("GADGET_MUNIT", 1e10)

            f.write("LgridDomain       = %d\n" % LgridDomain)
            f.write("LgridMax       = %d\n" % LgridMax)
            f.write("NperDomCell       = %f\n" % NperDomCell)
            f.write("NperRefCell       = %f\n" % NperRefCell)
            f.write("VescTune       = %f\n" % VescTune)
            f.write("NminPerHalo       = %d\n" % NminPerHalo)
            f.write("RhoVir       = %f\n" % RhoVir)
            f.write("Dvir       = %f\n" % Dvir)
            f.write("MaxGatherRad       = %f\n" % MaxGatherRad)
            f.write("LevelDomainDecomp       = %d\n" % LevelDomainDecomp)
            f.write("NcpuReading       = %d\n" % NcpuReading)

            f.write("[GADGET]\n")
            f.write("GADGET_LUNIT       = %e\n" % GADGET_LUNIT)
            f.write("GADGET_MUNIT       = %e\n" % GADGET_MUNIT)

            # Any params we missed
            # for key in kwargs.keys():
            #     f.write("%s = %s\n" % (key, kwargs[key]))

        logger.info(
            "%sCatalogue wrote a partial(!) config file. Exiting" % self.finder)
        return True
