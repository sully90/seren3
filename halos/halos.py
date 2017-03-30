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

    units = {'mvir': 'Msol h**-1',
             'pos': 'kpc a h**-1',
             'vel': 'km s**-1',
             'rvir': 'kpc a h**-1',
             'rmax': 'kpc a h**-1',
             'r2': 'kpc a h**-1',
             'mpb_offset': 'kpc a h**-1',
             'com_offset': 'kpc a h**-1',
             'v_max': 'km s**-1',
             'v_esc': 'km s**-1',
             'sigv': 'km s**-1',
             'b_to_a': 'kpc a h**-1',
             'c_to_a': 'kpc a h**-1',
             'ekin': 'Msol h**-1 (km s**-1ec)**2',
             'epot': 'Msol h**-1 (km s**-1ec)**2',
             'surfp': 'Msol h**-1 (km s**-1ec)**2',
             'phiO': '(km s**-1ec)**2',
             'm_gas': 'Msol h**-1',
             'b_to_a_gas': 'kpc a h**-1',
             'c_to_a_gas': 'kpc a h**-1',
             'ekin_gas': 'Msol h**-1 (km s**-1ec)**2',
             'epot_gas': 'Msol h**-1 (km s**-1ec)**2',
             'm_star': 'Msol h**-1',
             'b_to_a_star': 'kpc a h**-1',
             'c_to_a_star': 'kpc a h**-1',
             'ekin_star': 'Msol h**-1 (km s**-1ec)**2',
             'epot_star': 'Msol h**-1 (km s**-1ec)**2',
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
        from seren3.utils.sge import ncpu
        from seren3.utils import which
        
        r2g = which("ramses2gadget")
        ahf = which("AHF-v1.0-084")

        tasks = []

        # Write the config
        path = "%s/AHF/%03d/" % (self.base.path, self.base.ioutput)
        if os.path.isfile("%s/ahf.input" % path) is False:
            if os.path.isfile("%s/ahf.input" % path) is False:
                if self.write_cfg(**kwargs):
                    print "AHFCatalogue wrote a partial(!) config file."
                else:
                    raise Exception("AHFCatalogue unable to write config file!")

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
        NSLOTS = kwargs.get("NSLOTS", int(ncpu() / 4.))
        for task in tasks:
            mpi_task = "mpirun -np {NSLOTS} {EXE}".format(NSLOTS=NSLOTS, EXE=task)
            print mpi_task
            subprocess.check_output(mpi_task, shell=True)

        subprocess.check_output("cat {AHF_PATH}/halos/*_halos > {AHF_PATH}/halos/all_halos".format(AHF_PATH=ahf_path), shell=True)
        super(AHFCatalogue, self).__init__(self.base, "AHF", filename=None, **kwargs)
        return True

    @property
    def ahf_path(self):
        return "%s/%03d/halos/" % (self.finder_base_dir, self.base.ioutput)

    def get_boxsize(self, **kwargs):
        '''
        Returns the boxsize, according to AHF, in Mpc a h**-1
        '''
        import glob
        list_files = glob.glob("%s/*.log" % self.ahf_path)
        with open(list_files[0], 'r') as f:
            while True:
                l = f.readline()
                if l.startswith('simu.boxsize'):
                    box_size = float(l.split(':')[1])
                    return self.base.array(box_size, "Mpc a h**-1")

    # def can_load(self, **kwargs):
    #     '''
    #     Check if hlist files exist
    #     '''
    #     import os
    #     if os.path.isfile('%s/all_halos' % self.ahf_path) is False:
    #         path = "%s/AHF/%03d/" % (self.base.path, self.base.ioutput)
    #         if os.path.isfile("%s/ahf.input" % path) is False:
    #             if self.write_cfg(**kwargs):
    #                 print "AHFCatalogue wrote a partial(!) config file."
    #             else:
    #                 raise Exception("AHFCatalogue unable to write config file!")
    #         else:
    #             print "AHFCatalogue not found - ahf.input already written!"
    #         return False
    #     return True

    def can_load(self, **kwargs):
        import os
        if os.path.isfile("%s/all_halos" % self.ahf_path):
          return True, "exists"
        else:
          return False, "Cannot locate all_halos file"

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
        return Halo(haloprops, self.base, self.units, self.get_boxsize())

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

class RockstarCatalogue(HaloCatalogue):
    '''
    Class to handle catalogues produced by Rockstar
    Reads the out.list files
    '''
    # halo_type = np.dtype( [('id', np.int64), ('descid', np.int64), \
    #             ('mvir', 'f'), ('vmax', 'f'), \
    #             ('vrms', 'f'), ('rvir', 'f'), \
    #             ('rs', 'f'), ('np', 'f'), \
    #             ('pos', 'f', 3), ('vel', 'f', 3), \
    #             ('J', 'f', 3), ('spin', 'f'), \
    #             ('rs_klypin', 'f'), ('mvir_all', 'f'), \
    #             ('m200b', 'f'), ('m200c', 'f'), \
    #             ('m500c', 'f'), ('m2500c', 'f'), \
    #             ('r200b', 'f'), ('r200c', 'f'), \
    #             ('r500c', 'f'), ('r2500c', 'f'), \
    #             ('xoff', 'f'), ('voff', 'f'), \
    #             ('spin_bullock', 'f'), ('b_to_a', 'f'), \
    #             ('c_to_a', 'f'), ('A', 'f', 3), \
    #             ('b_to_a_500c', 'f'), ('c_to_a_500c', 'f'), \
    #             ('A500c', 'f', 3), ('T/U', 'f'), \
    #             ('m_pe_behroozi', 'f'), ('M_pe_Diemer', 'f'), \
    #             ('halfmass_radius', 'f')] )

    halo_type = np.dtype( [('id', np.int64), ('descid', np.int64), \
                ('mvir', 'f'), ('vmax', 'f'), \
                ('vrms', 'f'), ('rvir', 'f'), \
                ('rs', 'f'), ('np', 'f'), \
                ('pos', 'f', 3), ('vel', 'f', 3), \
                ('J', 'f', 3), ('spin', 'f'), \
                ('rs_klypin', 'f'), ('mvir_all', 'f'), \
                ('m200b', 'f'), ('m200c', 'f'), \
                ('m500c', 'f'), ('m2500c', 'f'), \
                ('xoff', 'f'), ('voff', 'f'), \
                ('spin_bullock', 'f'), ('b_to_a', 'f'), \
                ('c_to_a', 'f'), ('A', 'f', 3), \
                ('b_to_a_500c', 'f'), ('c_to_a_500c', 'f'), \
                ('A500c', 'f', 3), ('T/U', 'f'), \
                ('m_pe_behroozi', 'f'), ('M_pe_Diemer', 'f'), \
                ('halfmass_radius', 'f')] )

    units = {'sam_mvir': 'Msol h**-1',
        'mvir': 'Msol h**-1',
        'rvir': 'kpc a h**-1',
        'rs': 'kpc a h**-1',
        'vrms': 'km s**-1',
        'vmax': 'km s**-1',
        'pos': 'Mpc a h**-1',
        'vel': 'km s**-1',
        'J': 'Msol h**-1 Mpc h**-1 km s**-1',
        'mvir_all': 'Msol h**-1',
        'm200b': 'Msol h**-1',
        'm200c': 'Msol h**-1',
        'm500c': 'Msol h**-1',
        'm2500c': 'Msol h**-1',
        'm_alt': 'Msol h**-1',
        #'r_alt': 'kpc a h**-1',
        'xoff': 'kpc a h**-1',
        'voff': 'km s**-1',
        'A': 'kpc a h**-1',
        'halfmass_r': 'kpc a h**-1',
        'macc': 'Msol h**-1',
        'mpeak': 'Msol h**-1',
        'vacc': 'km s**-1',
        'vpeak': 'km s**-1',
        'acc_rate_inst': 'Msol h**-1 yr**-1',
        'acc_rate_100myr': 'Msol h**-1 100Myr**-1',
        'first_acc_mvir': 'Msol h**-1',
        'first_acc_vmax': 'km s**-1',
        'vmax_at_mpeak': 'km s**-1'}

    def __init__(self, pymses_snapshot, **kwargs):
        super(RockstarCatalogue, self).__init__(pymses_snapshot, "Rockstar", **kwargs)

    def can_load(self, **kwargs):
        import os
        # return os.path.isdir("%s/%s/" % (self.base.path, config.get("halo", "rockstar_base"))) and os.path.isfile(self.get_rockstar_info_fname())        
        if os.path.isdir(self.finder_base_dir):
          if os.path.isfile(self.get_rockstar_info_fname()):
            return True, "exists"
          else:
            return False, "Cannot locate info file"
        else:
          return False, "rockstar directory doesn't exist"

    def get_rockstar_info_fname(self):
        return "%s/info_rockstar.txt" % self.finder_base_dir

    def get_filename(self, **kwargs):
        '''
        Returns the rockstar catalogue filename
        '''
        rockstar_info_fname = self.get_rockstar_info_fname()
        base_aexp = 1./(1. + self.base.z)

        if kwargs.get("strict_so", False):
          # Used for accurate comparissons of halo mass-function.
          # Uses strict spherical-overdensities for mass calculation, instead
          # of FOF group.
          self.finder_base_dir = "%s/rockstar_strict_so_mass/" % self.base.path

        out_num = []
        aexp = []
        with open(rockstar_info_fname, "r") as f:
            for line in f:
                split_line = line.split('\t')
                out_num.append( int(split_line[0]) )
                aexp.append( float(split_line[1]) )
        aexp = np.array(aexp)
        idx_closest = (np.abs(aexp - base_aexp)).argmin()

        out_fname = "out_%i.list" % (out_num[idx_closest])
        #print 'RockstarCatalogue: matched to %s' % out_fname
        fname = "%s/%s" % (self.finder_base_dir, out_fname)
        return fname

    def get_boxsize(self, **kwargs):
        '''
        Returns boxsize according to rockstar
        '''
        import re

        with open(self.filename, 'r') as f:
            for line in f:
                if line.startswith('#Box size:'):
                    boxsize = re.findall("\d+\.\d+", line)[0]
                    return self.base.array(float(boxsize), "Mpc a h**-1")  # Mpc a h**-1

    def load(self, **kwargs):
        # Ensures file is closed at the end. If within_r is specified, it must be in code units
        with open(self.filename, 'r') as f:
            haloprops = np.loadtxt(f, dtype=self.halo_type, comments="#")

            self._nhalos = len(haloprops)
            self._haloprops = haloprops

    def _get_halo(self, item):
        haloprops = self._haloprops[item]
        return Halo(haloprops, self.base, self.units, self.get_boxsize())

class ConsistentTreesCatalogue(HaloCatalogue):

    halo_type = np.dtype([('aexp', 'f'),
                          ('id', np.int64),
                          ('desc_aexp', 'f'),
                          ('desc_id', 'f'),
                          ('num_prog', np.int64),
                          ('pid', np.int64),
                          ('upid', np.int64),
                          ('desc_pid', np.int64),
                          ('phantom', 'f'),
                          ('sam_mvir', 'f'),
                          ('mvir', 'f'),
                          ('rvir', 'f'),
                          ('rs', 'f'),
                          ('vrms', 'f'),
                          ('mmp', np.int64),  # Bool - most massive progenitor
                          ('scale_of_last_mm', 'f'),
                          ('vmax', 'f'),
                          ('pos', 'f', 3),
                          ('vel', 'f', 3),
                          ('J', 'f', 3),
                          ('spin', 'f'),
                          ('breadth_first_id',
                           np.int64),
                          ('depth_first_id',
                           np.int64),
                          ('tree_root_id', np.int64),
                          ('orig_halo_id', np.int64),
                          ('snap_num', np.int64),
                          ('next_coprog_depth_first_id',
                           np.int64),
                          ('last_prog_depth_first_id',
                           np.int64),
                          ('last_mainlead_depth_first_id',
                            np.int64),
                          ('tidal_force', 'f'),
                          ('tidal_id', np.int64),
                          ('rs_klypin', 'f'),
                          ('mvir_all', 'f'),
                          ('m_alt', 'f', 4),
                          #('r_alt', 'f', 4),
                          ('xoff', 'f'),
                          ('voff', 'f'),
                          ('spin_bullock', 'f'),
                          ('b_to_a', 'f'),
                          ('c_to_a', 'f'),
                          ('A', 'f', 3),
                          ('b_to_a_500c', 'f'),
                          ('c_to_a_500c', 'f'),
                          ('A_500c', 'f', 3),
                          ('T/|U|', 'f'),
                          ('m_pe_behroozi', 'f'),
                          ('m_pe_diemer', 'f'),
                          ('halfmass_r', 'f'),
                          # Consistent Trees Version 1.0 - Mass at accretion
                          ('macc', 'f'),
                          ('mpeak', 'f'),
                          # Consistent Trees Version 1.0 - Vmax at accretion
                          ('vacc', 'f'),
                          ('vpeak', 'f'),
                          ('halfmass_scale', 'f'),
                          ('acc_rate_inst', 'f'),
                          ('acc_rate_100myr', 'f'),
                          ('acc_rate_1tdyn', 'f'),
                          ('acc_rate_2tdyn', 'f'),
                          ('acc_rate_mpeak', 'f'),
                          ('mpeak_scale', 'f'),
                          ('acc_scale', 'f'),
                          ('first_acc_scale', 'f'),
                          ('first_acc_mvir', 'f'),
                          ('first_acc_vmax', 'f'),
                          ('vmax_at_mpeak', 'f'),
                          ('tidal_force_tdyn', 'f'),
                          ('log_vmax_vmax_tdyn_dmpeak', 'f'),
                          ('time_to_future_merger', 'f'),
                          ('future_merger_mmp_id', 'f')])

    units = {
        'sam_mvir': 'Msol h**-1',
        'mvir': 'Msol h**-1',
        'rvir': 'kpc a h**-1',
        'rs': 'kpc a h**-1',
        'vrms': 'km s**-1',
        'vmax': 'km s**-1',
        'pos': 'Mpc a h**-1',
        'vel': 'km s**-1',
        'J': 'Msol h**-1 Mpc h**-1 km s**-1',
        'mvir_all': 'Msol h**-1',
        'm_alt': 'Msol h**-1',
        #'r_alt': 'kpc a h**-1',
        'xoff': 'kpc a h**-1',
        'voff': 'km s**-1',
        'A': 'kpc a h**-1',
        'halfmass_r': 'kpc a h**-1',
        'macc': 'Msol h**-1',
        'mpeak': 'Msol h**-1',
        'vacc': 'km s**-1',
        'vpeak': 'km s**-1',
        'acc_rate_inst': 'Msol h**-1 yr**-1',
        'acc_rate_100myr': 'Msol h**-1 100Myr**-1',
        'first_acc_mvir': 'Msol h**-1',
        'first_acc_vmax': 'km s**-1',
        'vmax_at_mpeak': 'km s**-1'
    }

    def __init__(self, pymses_snapshot, **kwargs):
      super(ConsistentTreesCatalogue, self).__init__(pymses_snapshot, "ConsistentTrees", **kwargs)

    def can_load(self, **kwargs):
        import glob
        if len(glob.glob("%s/hlist_*" % self.finder_base_dir)) > 0.:
            return True, "exists"
        else:
            return False, "Unable to locate hlists files"

    def get_filename(self, **kwargs):
        import glob
        from seren3.exceptions import CatalogueNotFoundException
        # Filename is hlist_aexp.list
        # Look through the outputs and find the closest expansion factor
        aexp = self.base.cosmo['aexp']

        if kwargs.get("strict_so", False):
          # Used for accurate comparissons of halo mass-function.
          # Uses strict spherical-overdensities for mass calculation, instead
          # of FOF group.
          self.finder_base_dir = "%s/rockstar_strict_so_mass/hlists/" % self.base.path

        # Scan halo files for available expansion factors
        outputs = glob.glob( "%s/hlist_*" % (self.finder_base_dir) )

        if len(outputs) == 0:
          raise IOError("ConsistentTreesCatalogue: No outputs found")

        aexp_hlist = np.zeros(len(outputs))
        for i in range(len(outputs)):
            output = outputs[i]
            # Trim the aexp from the string
            aexp_hfile = float(output.split('/')[-1][6:-5])
            aexp_hlist[i] = aexp_hfile

        # Find the closest match
        idx = np.argmin(np.abs(aexp_hlist - aexp))

        if min(aexp_hlist[idx] / aexp, aexp / aexp_hlist[idx]) < 0.985:
          raise CatalogueNotFoundException("Unable to locate catalogue close to this snapshot.\nHlist aexp: %f, Snap aexp: %f" % (aexp_hlist[idx], aexp))

        return outputs[idx]

    def get_boxsize(self, **kwargs):
        '''
        Returns boxsize according to rockstar in Mpc a / h
        '''
        import re

        with open(self.filename, 'r') as f:
            for line in f:
                if line.startswith('#Full box size'):
                    boxsize = re.findall("\d+\.\d+", line)[0]
                    return self.base.array(float(boxsize), "Mpc a h**-1")  # Mpc a / h

    def load(self, **kwargs):
        # Ensures file is closed at the end. If within_r is specified, it must be in code units
        with open(self.filename, 'r') as f:
            haloprops = np.loadtxt(f, dtype=self.halo_type, comments="#")

            self._nhalos = len(haloprops)
            self._haloprops = haloprops

    def _get_halo(self, item):
        haloprops = self._haloprops[item]
        return Halo(haloprops, self.base, self.units, self.get_boxsize())

    @staticmethod
    def _find_mmp(hid, prog_halos):
        '''
        Returns the id for the most massive progenitor
        '''
        search_key = lambda halos: halos[:]["desc_id"] == hid
        progs = prog_halos.search(search_key)
        if len(progs) > 1:
            mmp_search_key = lambda x: x["mvir"]
            progs_sorted = sorted(progs, key=mmp_search_key, reverse=True)
            return progs_sorted[0].hid
        elif len(progs) == 1:
            return progs[0].hid
        else:
            return None

    def find_mmp(self, halo, back_to_iout=None):
        '''
        Locates the most massive progenitor
        '''
        from seren3 import load_snapshot
        if back_to_iout is None:
            back_to_iout = self.base.ioutput-1

        hid = halo.hid
        ioutputs = range(back_to_iout, self.base.ioutput)[::-1]

        last = self.base.ioutput
        for iout_prog in ioutputs:
            # Start with the previous snapshot, find the most massive progenitor and use that
            prog_snap = load_snapshot(self.base.path, iout_prog)
            prog_halos = prog_snap.halos(finder='ctrees')
            mmp_id = self._find_mmp(hid, prog_halos)
            if mmp_id is None:
                print 'Unable to fing progenitor in output %i.\nReturning last know progenitor (output %i)' % (iout_prog, last)
                return hid, prog_halos
            else:
                hid = mmp_id
                last = iout_prog
        return hid, prog_halos

    def iterate_progenitors(self, halo, back_to_aexp = 0.):
        '''
        Iterates through list of progenitors without loading halo catalogues completely
        '''
        import numpy as np
        import glob
        from seren3.utils import natural_sort
        from seren3.core.simulation import Simulation

        outputs = natural_sort(glob.glob("%s/hlist_*" % self.finder_base_dir))

        aexp_hlist = np.zeros(len(outputs))
        for i in range(len(outputs)):
            output = outputs[i]
            # Trim aexp from string
            aexp_hfile = float(output.split('/')[-1][6:-5])
            aexp_hlist[i] = aexp_hfile

        idx_start = np.abs( aexp_hlist - self.base.info["aexp"] ).argmin()
        idx_end = np.abs( aexp_hlist - back_to_aexp ).argmin()

        hid = int(halo.hid)

        # Loop through hlists in reverse and locate progenitors
        for i in range(idx_end, idx_start)[::-1]:
            mmp_props = None
            mmp_mass = 0.
            with open( outputs[i], "r" ) as f:
                haloprops = np.loadtxt(f, dtype=self.halo_type, comments="#")
                for props in haloprops:
                    if (props["desc_id"] == hid) and (props["mvir"] > mmp_mass):
                        # This halo is a candidate for mmp
                        mmp_props = props
                        mmp_mass = props["mvir"]

            if (mmp_props != None):
                sim = Simulation(halo.base.path)
                # print aexp_hlist[::-1][i]
                z = (1./aexp_hlist[::-1][i]) - 1.
                prog_snap = sim[sim.redshift(z)]
                yield Halo(mmp_props, prog_snap, self.units, self.get_boxsize())  # the mmp
                hid = int(mmp_props["id"])
            else:
                print "No descentent found - exiting"
                break
