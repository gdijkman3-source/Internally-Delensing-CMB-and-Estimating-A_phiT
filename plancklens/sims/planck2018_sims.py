r"""Planck 2018 release simulation libraries.

    Note:
        These simulations are located on NERSC systems.

    Note:
        Units of the maps stored at NERSC are :math:`K` but this module returns maps in :math:`\mu K`

"""
from plancklens.utils import clhash, hash_check
from plancklens.helpers import mpi
import os
from os.path import join as opj
import healpy as hp
import numpy as np
import pickle as pk
from plancklens import utils
import lenspyx

has_key = lambda key : key in os.environ.keys()

class empty_map:
    r""" Just nothing    
    """

    def __init__(self, nside=2048):
        self.npix = hp.nside2npix(nside)

    def get_sim_tmap(self, idx):
        return np.zeros(self.npix)

    def get_sim_pmap(self, idx):
        return np.zeros(self.npix), np.zeros(self.npix)

class smica_dx12_custom:
    r""" Custom SMICA simulation and data library.

        This class extends the functionality of SMICA simulations with additional features 
        for delensing and noise inclusion. It supports handling custom simulation directories,
        delensing maps, and more robust noise handling configurations.
    """
    def __init__(self, libdir = None, delens=0.0, include_noise=True, 
                 dlm_folder = None, klm_folder = None, plm_parfile = None,
                 sims_dcl = None, lmax=2048, nside = 2048, kPhi = None, create_gaussian = False):
        """Initializes the custom SMICA library with specified configurations.

        Args:
            libdir (str, optional): Path to the directory for storing output data. Default is None.
            delens (float, optional): Factor for delensing, 0 for no delensing, 1 for fully delensing. Default is 0.0.
            include_noise (bool, optional): Flag to include noise in simulations. Default is True.
            dlm_folder (str, optional): Folder containing the inverse delfection angles in harmonic Alms. Required if delens > 0. Default is None.
            sims_dcl (object, optional): Custom simulation class to add extra noise, as PL18 did. Default is None.
            lmax (int, optional): Maximum multipole to consider. Default is 2048.
            nside (int, optional): HEALPix nside parameter. Default is 2048.

        Raises:
            AssertionError: If invalid configurations are set, such as noise inclusion without simulation data or missing delensing folders.
        """
        
        self.cmbs = opj(os.environ["INPUT"],'cmb_%05d.fits')
        self.noise = opj(os.environ["INPUT"],'noise_%05d.fits')
        self.data = opj(os.environ["INPUT"],'SMICA.fits')

        self.include_noise = include_noise
        self.delens = delens
        self.dlm = None if dlm_folder is None else opj(dlm_folder, 'dlm_%03d.fits')
        self.klm = None if klm_folder is None else opj(klm_folder, 'klm_%03d.fits')
        self.plm_par = None if plm_parfile is None else plm_parfile
        self.WF = None
        self.MF = None
        self.qresp = None
        self.libdir = libdir
        self.sims_dcl = sims_dcl
        self.lmax = lmax
        self.nside = nside
        self.kPhi = 'p' if kPhi is None else kPhi
        self.create_gaussian = create_gaussian

        if libdir is not None:
            fn_hash = os.path.join(libdir, 'hash.pk')
            if mpi.rank == 0 and not os.path.exists(fn_hash):
                if not os.path.exists(libdir): os.makedirs(libdir)
                pk.dump(self.hashdict(), open(fn_hash, 'wb'), protocol=2)
            mpi.barrier()
            hash_check(self.hashdict(), pk.load(open(fn_hash, 'rb')), fn=fn_hash)

    def hashdict(self):
        return {'cmbs':self.cmbs, 
                'noise':self.noise, 
                'libdir': self.libdir,
                'delens': self.delens,
                'include_noise': self.include_noise,
                'dlm': self.dlm,
                'lmax': self.lmax,
                'nside': self.nside,
                'sims_dcl': None if self.sims_dcl is None else self.sims_dcl.hashdict()}

    def get_dlm(self, idx):

        if self.dlm is not None:
            # Load (inverse) deflection angle
            dlm = hp.read_alm(self.dlm % idx)

        elif self.plm_par is not None:
            if self.WF is None:
                from binner_sims import binner_sims
                self.lmax_qlm = self.plm_par.lmax_qlm
                self.WF = binner_sims(self.kPhi, self.kPhi, self.plm_par, "consext8").get_WF()[:self.lmax_qlm+1]

            if self.MF is None:
                self.MF = self.plm_par.qlms_dd.get_sim_qlm_mf(self.kPhi, self.plm_par.mc_sims_mf_dd, lmax=self.lmax_qlm)

            if self.qresp is None:
                self.qresp = utils.cli(self.plm_par.qresp_dd.get_response(self.kPhi, 'p')[:self.lmax_qlm+1])

            dlm = self.plm_par.qlms_dd.get_sim_qlm(self.kPhi, idx, lmax=self.lmax_qlm) - self.MF
            hp.almxfl(dlm, self.WF * self.qresp, inplace = True)

            l = np.arange(self.lmax_qlm+1)
            l *= l + 1
            hp.almxfl(dlm, -1*np.sqrt(l), inplace=True)

        elif self.klm is not None:
            # Load the klm
            dlm = hp.read_alm(self.klm % idx)

            # Convert to (inverse) dlm
            l = np.arange(self.lmax+1)
            l *= l + 1
            dlm = -1*hp.almxfl(dlm, 2*utils.cli(np.sqrt(l)))

        else:
            assert 0, "Trying to delens without input klm, dlm or plm"

        # Setting the amount of delensing
        dlm *= self.delens
        return dlm

    def delens_map(self, idx, fname):
        """Creates and saves delensed maps for a specified simulation index.

        Args:
            idx (int): Simulation index.
            fname (str): Filename for saving the delensed map.

        This method reads and combines CMB and noise maps, converts them to spherical harmonics,
        applies inverse deflection angles, and generates delensed temperature and polarization maps.
        """
        if self.create_gaussian:
            Tlm = self.plm_par.ivfs.get_sim_tmliklm(idx)

            # Create delensed maps
            T = lenspyx.alm2lenmap(Tlm, dlms=self.get_dlm(idx))

            # Save
            hp.write_map(fname, T, dtype=np.float64)
            return

        elif self.include_noise and idx!=-1:
            _t = self.sims_dcl.get_sim_tmap(idx)
            _q, _u = self.sims_dcl.get_sim_pmap(idx)
            T = 1e6 * (hp.read_map(self.cmbs % idx, field=0, dtype=np.float64) + hp.read_map(self.noise % idx, field=0, dtype=np.float64)) + _t
            Q = 1e6 * (hp.read_map(self.cmbs % idx, field=1, dtype=np.float64) + hp.read_map(self.noise % idx, field=1, dtype=np.float64)) + _q
            U = 1e6 * (hp.read_map(self.cmbs % idx, field=2, dtype=np.float64) + hp.read_map(self.noise % idx, field=2, dtype=np.float64)) + _u
            del _t, _q, _u
        elif idx==-1 and self.include_noise:
            print("Custom SIMS: constructing tqu for dat")
            _t = self.sims_dcl.get_sim_tmap(idx)
            _q, _u = self.sims_dcl.get_sim_pmap(idx)
            T = 1e6 * hp.read_map(self.data, field=0, dtype=np.float64) + _t
            Q = 1e6 * hp.read_map(self.data, field=1, dtype=np.float64) + _q
            U = 1e6 * hp.read_map(self.data, field=2, dtype=np.float64) + _u
            del _t, _q, _u
        else:
            T = 1e6 * hp.read_map(self.cmbs % idx, field=0, dtype=np.float64)
            Q = 1e6 * hp.read_map(self.cmbs % idx, field=1, dtype=np.float64)
            U = 1e6 * hp.read_map(self.cmbs % idx, field=2, dtype=np.float64)

        # First convert to Alms
        Tlm, Elm, Blm = hp.map2alm([T, Q, U], lmax=self.lmax)
        del T, Q, U

        # Create delensed maps
        T, Q, U = lenspyx.alm2lenmap([Tlm, Elm, Blm], dlms=self.get_dlm(idx))

        # Save
        hp.write_map(fname, [T, Q, U], dtype=np.float64)
       
    def get_sim_tmap(self, idx):
        r"""Returns dx12 SMICA temperature map for a simulation

            Args:
                idx: simulation index

            Returns:
                SMICA simulation *idx*, including noise. Returns dx12 SMICA data map for *idx* =-1

        """
        print("Custom SIMS: get_sim_tmap triggered")
        if idx==-1: return self.get_dat_tmap()

        if self.libdir is not None:
            fname = os.path.join(self.libdir, 'sim_%05d.fits'%idx)
            if not os.path.exists(fname):
                self.delens_map(idx=idx, fname=fname)
            return hp.read_map(fname, field=0, dtype=np.float64)

        if self.include_noise:
            return 1e6 * (hp.read_map(self.cmbs % idx, field=0, dtype=np.float64) + hp.read_map(self.noise % idx, field=0, dtype=np.float64)) + self.sims_dcl.get_sim_tmap(idx)
        else:
            return 1e6 * hp.read_map(self.cmbs % idx, field=0, dtype=np.float64)

    def get_dat_tmap(self):
        # return 1e6 * hp.read_map(self.data, field=0, dtype=np.float64)
        print("Custom SIMS: Returning dat tmap")
        if self.libdir is not None:
            fname = os.path.join(self.libdir, 'SMICA.fits')
            if not os.path.exists(fname):
                self.delens_map(idx=-1, fname=fname)
            return hp.read_map(fname, field=0, dtype=np.float64)

        return 1e6 * hp.read_map(self.data, field=0, dtype=np.float64)

    def get_sim_pmap(self, idx):
        r"""Returns dx12 SMICA polarization map for a simulation

            Args:
                idx: simulation index

            Returns:
                SMICA Q and U simulation *idx*, including noise. Returns dx12 SMICA data maps for *idx* =-1

        """
        if idx==-1: return self.get_dat_pmap()
        
        if self.libdir is not None:
            fname = os.path.join(self.libdir, 'sim_%05d.fits'%idx)
            if not os.path.exists(fname):
                self.delens_map(idx=idx, fname=fname)
            Q = hp.read_map(fname, field=1, dtype=np.float64)
            U = hp.read_map(fname, field=2, dtype=np.float64)
            return Q, U

        if self.include_noise:
            _q, _u = self.sims_dcl.get_sim_pmap(idx)
            Q = 1e6 * (hp.read_map(self.cmbs % idx, field=1, dtype=np.float64) + hp.read_map(self.noise % idx, field=1, dtype=np.float64)) + _q
            U = 1e6 * (hp.read_map(self.cmbs % idx, field=2, dtype=np.float64) + hp.read_map(self.noise % idx, field=2, dtype=np.float64)) + _u
        else:
            Q = 1e6 * hp.read_map(self.cmbs % idx, field=1, dtype=np.float64)
            U = 1e6 * hp.read_map(self.cmbs % idx, field=2, dtype=np.float64) 
        return Q, U
    
    def get_dat_pmap(self):
        print("Custom SIMS: get dat pmap triggered")
        if self.libdir is not None:
            fname = os.path.join(self.libdir, 'SMICA.fits')
            if not os.path.exists(fname):
                self.delens_map(idx=-1, fname=fname)
            Q = hp.read_map(fname, field=1, dtype=np.float64)
            U = hp.read_map(fname, field=2, dtype=np.float64)
            return Q, U

        Q = 1e6 * hp.read_map(self.data, field=1, dtype=np.float64)
        U = 1e6 * hp.read_map(self.data, field=2, dtype=np.float64) 
        return Q, U


class smica_dx12:
    r""" SMICA 2018 release simulation and data library at NERSC in uK.

        Note:
            This now converts all maps to double precision
            (healpy 1.15 changed read_map default type behavior, breaking in a way that is not very clear as yet the behavior of the conjugate gradient inversion chain)
    """
    def __init__(self):
        self.cmbs = opj(os.environ["INPUT"],'cmb_%05d.fits')
        self.noise = opj(os.environ["INPUT"],'noise_%05d.fits')
        self.data = opj(os.environ["INPUT"], 'SMICA.fits')

    def hashdict(self):
        return {'cmbs':self.cmbs, 'noise':self.noise, 'data':self.data}

    def get_sim_tmap(self, idx):
        r"""Returns dx12 SMICA temperature map for a simulation

            Args:
                idx: simulation index

            Returns:
                SMICA simulation *idx*, including noise. Returns dx12 SMICA data map for *idx* =-1

        """
        if idx == -1:
            return self.get_dat_tmap()
        return 1e6 * (hp.read_map(self.cmbs % idx, field=0, dtype=np.float64) + hp.read_map(self.noise % idx, field=0, dtype=np.float64))

    def get_dat_tmap(self):
        return 1e6 * hp.read_map(self.data, field=0, dtype=np.float64)

    def get_sim_pmap(self, idx):
        r"""Returns dx12 SMICA polarization map for a simulation

            Args:
                idx: simulation index

            Returns:
                SMICA Q and U simulation *idx*, including noise. Returns dx12 SMICA data maps for *idx* =-1

        """
        if idx == -1:
            return self.get_dat_pmap()
        Q = 1e6 * (hp.read_map(self.cmbs % idx, field=1, dtype=np.float64) + hp.read_map(self.noise % idx, field=1, dtype=np.float64))
        U = 1e6 * (hp.read_map(self.cmbs % idx, field=2, dtype=np.float64) + hp.read_map(self.noise % idx, field=2, dtype=np.float64))
        return Q, U

    def get_dat_pmap(self):
        return 1e6 * hp.read_map(self.data, field=1, dtype=np.float64), 1e6 * hp.read_map(self.data, field=2, dtype=np.float64)


class smica_dx12_SZdeproj:
    r"""tSZ-deprojected SMICA 2018 release simulation and data library at NERSC in uK

        Note:

            This now converts all maps to double precision
            (healpy 1.15 changed read_map default type behavior, breaking in a way that is not very clear as yet the behavior of the conjugate gradient inversion chain)


    """
    def __init__(self):
        self.cmbs  = opj(os.environ["CFS"],'planck/data/compsep/comparison/dx12_v3/nosz/mc_cmb/dx12_v3_smica_nosz_cmb_mc_%05d_005a_2048.fits')
        self.noise = opj(os.environ["CFS"],'planck/data/compsep/comparison/dx12_v3/nosz/mc_noise/dx12_v3_smica_nosz_noise_mc_%05d_005a_2048.fits')
        self.data  = opj(os.environ["CFS"],'planck/data/compsep/comparison/dx12_v3/nosz/dx12_v3_smica_nosz_cmb_005a_2048.fits')

    def hashdict(self):
        return {'cmbs':self.cmbs, 'noise':self.noise, 'data':self.data}

    def get_sim_tmap(self, idx):
        r"""Returns dx12 tSZ-deproj SMICA temperature map for a simulation

            Args:
                idx: simulation index

            Returns:
                SMICA simulation *idx*, including noise. Returns dx12 SMICA data map for *idx* =-1

        """
        if idx == -1:
            return self.get_dat_tmap()
        return 1e6 * (hp.read_map(self.cmbs % idx, field=0, dtype=np.float64) + hp.read_map(self.noise % idx, field=0, dtype=np.float64))

    def get_dat_tmap(self):
        r"""Returns dx12 tSZ-deproj SMICA Planck data temperature map

        """
        return 1e6 * hp.read_map(self.data, field=0, dtype=np.float64)

    @staticmethod
    def get_sim_pmap(idx):
        return smica_dx12().get_sim_pmap(idx)

    @staticmethod
    def get_dat_pmap():
        return smica_dx12().get_dat_pmap()



class ffp10cmb_widnoise:
    r"""Simulation library with freq-0 FFP10 lensed CMB together with idealized, homogeneous noise.

        Args:
            transf: transfer function (beam and pixel window)
            nlevt: temperature noise level in :math:`\mu K`-arcmin.
            nlevp: polarization noise level in :math:`\mu K`-arcmin.
            pix_libphas: random phases simulation library (see plancklens.sims.phas.py) of the noise maps.

    """
    def __init__(self, transf, nlevt, nlevp, pix_libphas, nside=2048):
        assert pix_libphas.shape == (hp.nside2npix(nside),), pix_libphas.shape
        self.nlevt = nlevt
        self.nlevp = nlevp
        self.transf = transf
        self.pix_libphas = pix_libphas
        self.nside = nside

    def hashdict(self):
        return {'transf':utils.clhash(self.transf), 'nlevt':np.float32(self.nlevt), 'nlevp':np.float32(self.nlevp),
                'pix_phas':self.pix_libphas.hashdict()}

    def get_sim_tmap(self, idx):
        T = hp.alm2map(hp.almxfl(cmb_len_ffp10.get_sim_tlm(idx), self.transf), self.nside)
        nlevt_pix = self.nlevt / np.sqrt(hp.nside2pixarea(self.nside, degrees=True)) / 60.
        T += self.pix_libphas.get_sim(idx, idf=0) * nlevt_pix
        return T

    def get_sim_pmap(self, idx):
        elm = hp.almxfl(cmb_len_ffp10.get_sim_elm(idx), self.transf)
        blm = hp.almxfl(cmb_len_ffp10.get_sim_blm(idx), self.transf)
        Q, U = hp.alm2map_spin((elm, blm), self.nside, 2, hp.Alm.getlmax(elm.size))
        del elm, blm
        nlevp_pix = self.nlevp / np.sqrt(hp.nside2pixarea(self.nside, degrees=True)) / 60.
        Q += self.pix_libphas.get_sim(idx, idf=1) * nlevp_pix
        U += self.pix_libphas.get_sim(idx, idf=2) * nlevp_pix
        return Q, U

class cmb_len_ffp10:
    """ FFP10 input sim libraries, lensed alms.

        The lensing deflections contain the L=1 aberration term (constant across all maps)
        due to our motion w.r.t. the CMB frame.

    """
    def __init__(self):
        pass

    def hashdict(self):
        return {'sim_lib': 'ffp10 lensed scalar cmb inputs, freq 0'}

    @staticmethod
    def get_sim_tlm(idx):
        """
            Args:
                idx: simulation index

            Returns:
                lensed temperature simulation healpy alm array

        """
        return 1e6 * hp.read_alm(opj(os.environ["CFS"],'cmb/data/generic/cmb/ffp10/mc/scalar/ffp10_lensed_scl_cmb_000_alm_mc_%04d.fits'%idx), hdu=1)

    @staticmethod
    def get_sim_elm(idx):
        """
            Args:
                idx: simulation index

            Returns:
                lensed E-polarization simulation healpy alm array

        """
        return 1e6 * hp.read_alm(opj(os.environ["CFS"],'cmb/data/generic/cmb/ffp10/mc/scalar/ffp10_lensed_scl_cmb_000_alm_mc_%04d.fits'%idx), hdu=2)

    @staticmethod
    def get_sim_blm(idx):
        """
            Args:
                idx: simulation index

            Returns:
                lensed B-polarization simulation healpy alm array

        """
        return 1e6 * hp.read_alm(opj(os.environ["CFS"],'cmb/data/generic/cmb/ffp10/mc/scalar/ffp10_lensed_scl_cmb_000_alm_mc_%04d.fits'%idx), hdu=3)


class cmb_unl_ffp10:
    """FFP10 input sim libraries, unlensed alms.

    """
    def __init__(self):
        pass

    def hashdict(self):
        return {'sim_lib': 'ffp10 unlensed scalar cmb inputs'}

    @staticmethod
    def get_sim_tlm(idx):
        """
            Args:
                idx: simulation index

            Returns:
                unlensed temperature simulation healpy alm array

        """
        return 1e6 * hp.read_alm(opj(os.environ["CFS"],'cmb/data/generic/cmb/ffp10/mc/scalar/ffp10_unlensed_scl_cmb_000_tebplm_mc_%04d.fits'% idx), hdu=1)

    @staticmethod
    def get_sim_elm(idx):
        """
            Args:
                idx: simulation index

            Returns:
                unlensed E-polarization simulation healpy alm array

        """
        return 1e6 * hp.read_alm(opj(os.environ["CFS"],'cmb/data/generic/cmb/ffp10/mc/scalar/ffp10_unlensed_scl_cmb_000_tebplm_mc_%04d.fits'% idx), hdu=2)

    @staticmethod
    def get_sim_blm(idx):
        """
            Args:
                idx: simulation index

            Returns:
                unlensed B-polarization simulation healpy alm array

        """
        return 1e6 * hp.read_alm(opj(os.environ["CFS"],'cmb/data/generic/cmb/ffp10/mc/scalar/ffp10_unlensed_scl_cmb_000_tebplm_mc_%04d.fits'% idx), hdu=3)

    @staticmethod
    def get_sim_plm(idx):
        r"""
            Args:
                idx: simulation index

            Returns:
               lensing potential :math:`\phi_{LM}` simulation healpy alm array

        """
        return hp.read_alm(opj(os.environ["CFS"],'cmb/data/generic/cmb/ffp10/mc/scalar/ffp10_unlensed_scl_cmb_000_tebplm_mc_%04d.fits'% idx), hdu=4)


