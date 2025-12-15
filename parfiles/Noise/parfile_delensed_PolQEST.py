import os
import healpy as hp
import numpy as np
import env_config
import plancklens
from plancklens.filt import filt_cinv, filt_util
from plancklens import utils
from plancklens import qest, qecl, qresp
from plancklens import nhl
from plancklens.n1 import n1
from plancklens.sims import planck2018_sims, cmbs, phas, maps, utils as maps_utils

assert 'PLENS' in os.environ.keys(), 'Set env. variable PLENS to a writeable folder'
TEMP =  os.path.join(os.environ['PLENS'], 'noisy_ivfs_delensed_polQEST')
cls_path = os.path.join(os.path.dirname(os.path.abspath(plancklens.__file__)), 'data', 'cls')

lmax_ivf = 2048
lmin_ivf = 100
lmax_qlm = 4096
nside = 2048
nlev_t = 35.
nlev_p = 55.
nsims = 300

transf = hp.gauss_beam(5. / 60. / 180. * np.pi, lmax=lmax_ivf) * hp.pixwin(nside)[:lmax_ivf + 1]
cl_unl = utils.camb_clfile(os.path.join(cls_path, 'FFP10_wdipole_lenspotentialCls.dat'))
cl_len = utils.camb_clfile(os.path.join(cls_path, 'FFP10_wdipole_lensedCls.dat'))
cl_weight = utils.camb_clfile(os.path.join(cls_path, 'FFP10_wdipole_lensedCls.dat'))
cl_weight['bb'] *= 0.

# Masks
Tmaskpaths = [os.path.join(os.environ["PARAMS"]  ,"mask.fits.gz")]
libdir_cinvt = os.path.join(TEMP, 'cinv_t')
libdir_cinvp = os.path.join(TEMP, 'cinv_p')
libdir_ivfs  = os.path.join(TEMP, 'ivfs')
libdir_dclphas = os.path.join(TEMP, 'dcl_phas')
libdir_sims = os.path.join(TEMP, "sims_delensed")

dcl_phas = phas.lib_phas(libdir_dclphas, 3, lmax_ivf)
dcl = np.loadtxt(os.path.join(os.environ["PARAMS"]  ,"dcl_sim"))[:, :lmax_ivf+1]* utils.cli(transf)**2
dcl_dat = np.loadtxt(os.path.join(os.environ["PARAMS"]  ,"dcl_dat"))[:, :lmax_ivf+1]* utils.cli(transf)**2

sims_dcl_sim = maps.cmb_maps_noisefree(cmbs.sims_cmb_unl({'tt':dcl[0], 'ee':dcl[1], 'bb':dcl[2]}, dcl_phas), transf)
sims_dcl_dat = maps.cmb_maps_noisefree(cmbs.sims_cmb_unl({'tt':dcl_dat[0], 'ee':dcl_dat[1], 'bb':dcl_dat[2]}, dcl_phas), transf)
sims_dcl = maps_utils.sim_lib_add_dat([sims_dcl_sim, sims_dcl_dat])


import parfiles.Noise.parfile as plm_par
libdir_sims = os.path.join(TEMP, 'delensed_sims')
sims  = planck2018_sims.smica_dx12_custom(libdir=libdir_sims, 
                                          include_noise=True, delens=1.,
                                          sims_dcl=sims_dcl,
                                          plm_parfile=plm_par, kPhi="p_p")


ninv_t = [np.array([3. / nlev_t ** 2])] + Tmaskpaths
cinv_t = filt_cinv.cinv_t(libdir_cinvt, lmax_ivf,nside, cl_len, transf, ninv_t,
                        marge_monopole=True, marge_dipole=True, marge_maps=[])

ninv_p = [[np.array([3. / nlev_p ** 2])] + Tmaskpaths]
cinv_p = filt_cinv.cinv_p(libdir_cinvp, lmax_ivf, nside, cl_len, transf, ninv_p)

ivfs_raw    = filt_cinv.library_cinv_sepTP(libdir_ivfs, sims, cinv_t, cinv_p, cl_len)
ftl = np.ones(lmax_ivf + 1, dtype=float) * (np.arange(lmax_ivf + 1) >= lmin_ivf) # rescaling or cuts. Here just a lmin cut
fel = np.ones(lmax_ivf + 1, dtype=float) * (np.arange(lmax_ivf + 1) >= lmin_ivf)
fbl = np.ones(lmax_ivf + 1, dtype=float) * (np.arange(lmax_ivf + 1) >= lmin_ivf)
ivfs   = filt_util.library_ftl(ivfs_raw, lmax_ivf, ftl, fel, fbl)

# This remaps idx -> idx + 1 by blocks of 60 up to 300:
ss_dict = { k : v for k, v in zip( np.concatenate( [ range(i*60, (i+1)*60) for i in range(0,5) ] ),
                    np.concatenate( [ np.roll( range(i*60, (i+1)*60), -1 ) for i in range(0,5) ] ) ) }
# This remap all sim. indices to the data maps
ds_dict = { k : -1 for k in range(300)}

ivfs_d = filt_util.library_shuffle(ivfs, ds_dict)
ivfs_s = filt_util.library_shuffle(ivfs, ss_dict)

libdir_qlmsdd = os.path.join(TEMP, 'qlms_dd')
libdir_qlmsds = os.path.join(TEMP, 'qlms_ds')
libdir_qlmsss = os.path.join(TEMP, 'qlms_ss')
qlms_dd = qest.library_sepTP(libdir_qlmsdd, ivfs, ivfs,   cl_len['te'], nside, lmax_qlm=lmax_qlm)
qlms_ds = qest.library_sepTP(libdir_qlmsds, ivfs, ivfs_d, cl_len['te'], nside, lmax_qlm=lmax_qlm)
qlms_ss = qest.library_sepTP(libdir_qlmsss, ivfs, ivfs_s, cl_len['te'], nside, lmax_qlm=lmax_qlm)

mc_sims_bias = np.arange(60, dtype=int)
mc_sims_var  = np.arange(60, 300, dtype=int)

mc_sims_mf_dd = mc_sims_bias
mc_sims_mf_ds = np.array([])
mc_sims_mf_ss = np.array([])

libdir_qcls_dd = os.path.join(TEMP, 'qcls_dd')
libdir_qcls_ds = os.path.join(TEMP, 'qcls_ds')
libdir_qcls_ss = os.path.join(TEMP, 'qcls_ss')
qcls_dd = qecl.library(libdir_qcls_dd, qlms_dd, qlms_dd, mc_sims_mf_dd)
qcls_ds = qecl.library(libdir_qcls_ds, qlms_ds, qlms_ds, mc_sims_mf_ds)
qcls_ss = qecl.library(libdir_qcls_ss, qlms_ss, qlms_ss, mc_sims_mf_ss)


libdir_nhl_dd = os.path.join(TEMP, 'nhl_dd')
nhl_dd = nhl.nhl_lib_simple(libdir_nhl_dd, ivfs, cl_weight, lmax_qlm)

libdir_n1_dd = os.path.join(TEMP, 'n1_ffp10')
n1_dd = n1.library_n1(libdir_n1_dd,cl_len['tt'],cl_len['te'],cl_len['ee'])

libdir_resp_dd = os.path.join(TEMP, 'qresp')
qresp_dd = qresp.resp_lib_simple(libdir_resp_dd, lmax_ivf, cl_weight, cl_len,
                                 {'t': ivfs.get_ftl(), 'e':ivfs.get_fel(), 'b':ivfs.get_fbl()}, lmax_qlm)


from qecl_pt import library_phiT as ptcls
nl_TT =  (nlev_t / 60. / 180. * np.pi / transf) ** 2
qcls_pt = ptcls(os.path.join(TEMP, 'qcls_pt'), qlms_dd, ivfs_raw, mc_sims_mf=mc_sims_mf_dd)
qcls_pt_ss = ptcls(os.path.join(TEMP, 'qcls_pt_ss'), qlms_dd, ivfs_s, mc_sims_mf=mc_sims_mf_dd)