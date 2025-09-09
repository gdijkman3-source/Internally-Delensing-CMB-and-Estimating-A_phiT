import os
import healpy as hp
import numpy as np
import env_config
import plancklens
from plancklens.filt import filt_simple, filt_util
from plancklens import utils
from plancklens import qest, qecl, qresp
from plancklens import nhl
from plancklens.n1 import n1
from plancklens.sims import planck2018_sims, phas
from qecl_pt import library_phiT as ptcls

assert 'PLENS' in os.environ.keys(), 'Set env. variable PLENS to a writeable folder'
TEMP =  os.path.join(os.environ['PLENS'], 'noNoise_noMask_idealFilt_fully_delensed')
cls_path = os.path.join(os.path.dirname(os.path.abspath(plancklens.__file__)), 'data', 'cls')

#--- definition of simulation and inverse-variance filtered simulation libraries:
lmax_ivf = 2048
lmin_ivf = 2  # We will use in the QE only CMB modes between lmin_ivf and lmax_ivf
lmax_qlm = 4096 # We will calculate lensing estimates until multipole lmax_qlm.
nside = 2048 # Healpix resolution of the data and sims.
nlev_t = 35. # Filtering noise level in temperature (here also used for the noise simulations generation).
nlev_p = 55. # Filtering noise level in polarization (here also used for the noise simulations generation).
nsims = 300  # Total number of simulations to consider.

transf = hp.gauss_beam(5. / 60. / 180. * np.pi, lmax=lmax_ivf) * hp.pixwin(nside)[:lmax_ivf + 1]
#: CMB transfer function. Here a 5' Gaussian beam and healpix pixel window function.

cl_unl = utils.camb_clfile(os.path.join(cls_path, 'FFP10_wdipole_lenspotentialCls.dat'))
cl_len = utils.camb_clfile(os.path.join(cls_path, 'FFP10_wdipole_lensedCls.dat'))
#: Fiducial unlensed and lensed power spectra used for the analysis.

nl_TT =  (nlev_t / 60. / 180. * np.pi / transf) ** 2

cl_weight = utils.camb_clfile(os.path.join(cls_path, 'FFP10_wdipole_lensedCls.dat'))
cl_weight['bb'] *= 0.
#: CMB spectra entering the QE weights (the spectra multplying the inverse-variance filtered maps in the QE legs)

libdir_pixphas = os.path.join(TEMP, 'pix_phas_nside%s'%nside)
pix_phas = phas.pix_lib_phas(libdir_pixphas, 3, (hp.nside2npix(nside),))
#: Noise simulation T, Q, U random phases instance.

klms_dir = '/mnt/d/THESIS/LENSING MAPS/'
libdir_sims = os.path.join(TEMP, 'delensed_sims')
sims  = planck2018_sims.smica_dx12_custom(libdir=libdir_sims, include_noise=False, delens=1., klm_folder=klms_dir)

# --- We turn to the inverse-variance filtering library. In this file we use trivial isotropic filtering,
#     (independent T and Pol. filtering)
ftl = utils.cli(cl_len['tt'][:lmax_ivf + 1] + (nlev_t / 60. / 180. * np.pi / transf) ** 2)
fel = utils.cli(cl_len['ee'][:lmax_ivf + 1] + (nlev_p / 60. / 180. * np.pi / transf) ** 2)
fbl = utils.cli(cl_len['bb'][:lmax_ivf + 1] + (nlev_p / 60. / 180. * np.pi / transf) ** 2)
ftl[:lmin_ivf] *= 0.
fel[:lmin_ivf] *= 0.
fbl[:lmin_ivf] *= 0.
#: Inverse CMB co-variance in T, E and B (neglecting TE coupling).

ivfs = filt_simple.library_fullsky_sepTP(os.path.join(TEMP, 'ivfs'), sims, nside, transf, cl_len, ftl, fel, fbl, cache=True)

#---- QE libraries instances. For the MCN0, RDN0, MC-correction etc calculation, we need in general three of them,
# Shuffling dictionary used for qlms_ss. This remaps idx -> idx + 1 by blocks of 60 up to 300:
ss_dict = { k : v for k, v in zip( np.concatenate( [ range(i*60, (i+1)*60) for i in range(0,5) ] ),
                    np.concatenate( [ np.roll( range(i*60, (i+1)*60), -1 ) for i in range(0,5) ] ) ) }

ivfs_s = filt_util.library_shuffle(ivfs, ss_dict)
#: This is a filtering instance shuffling simulation indices according to 'ss_dict'.


qlms_dd = qest.library_sepTP(os.path.join(TEMP, 'qlms_dd'), ivfs, ivfs,   cl_len['te'], nside, lmax_qlm=lmax_qlm)
qlms_ss = qest.library_sepTP(os.path.join(TEMP, 'qlms_ss'), ivfs, ivfs_s, cl_len['te'], nside, lmax_qlm=lmax_qlm)

#---- QE spectra libraries instances:
# This takes power spectra of the QE maps from the QE libraries, after subtracting a mean-field.
# Only qcls_dd needs a mean-field subtraction.

mc_sims_bias = np.arange(60) #: The mean-field will be calculated from these simulations.
mc_sims_var  = np.arange(60, 300) #: The covariance matrix will be calculated from these simulations

mc_sims_mf_dd = mc_sims_bias
mc_sims_mf_ss = np.array([]) #:By construction, only qcls_dd needs a mean-field subtraction.

qcls_dd = qecl.library(os.path.join(TEMP, 'qcls_dd'), qlms_dd, qlms_dd, mc_sims_mf_dd)
qcls_ss = qecl.library(os.path.join(TEMP, 'qcls_ss'), qlms_ss, qlms_ss, mc_sims_mf_ss)
qcls_pt = ptcls(os.path.join(TEMP, 'qcls_pt'), qlms_dd, ivfs, mc_sims_mf=mc_sims_mf_dd)

#---- semi-analytical Gaussian lensing bias library:
nhl_dd = nhl.nhl_lib_simple(os.path.join(TEMP, 'nhl_dd'), ivfs, cl_weight, lmax_qlm)

#---- N1 lensing bias library:
libdir_n1_dd = os.path.join(TEMP, 'n1_ffp10')
n1_dd = n1.library_n1(libdir_n1_dd,cl_len['tt'],cl_len['te'],cl_len['ee'])

#---- QE response calculation library:
qresp_dd = qresp.resp_lib_simple(os.path.join(TEMP, 'qresp'), lmax_ivf, cl_weight, cl_len,
                                 {'t': ivfs.get_ftl(), 'e':ivfs.get_fel(), 'b':ivfs.get_fbl()}, lmax_qlm)
