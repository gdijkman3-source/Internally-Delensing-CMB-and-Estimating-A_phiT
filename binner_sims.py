"""
binner_sims.py

This module provides classes to compute and bin cross-spectra and auto-spectra for lensing and temperature fields:

Classes:
  ffp10_binner_phiT:
    Builds and bins the phi-T cross-spectrum using Planck-style weights, handles fiducial and inverse-variance filters,
    Wiener filtering, Monte Carlo noise, and error estimation across simulations.

  binner_sims:
    Subclass of the FFP10 binner for phi-phi auto-spectra, adding methods to compute Wiener filters, Monte Carlo N0 bias,
    and lensing amplitude estimates for simulations and data.

Usage:
    from binner_sims import ffp10_binner_phiT, binner_sims
    binner = ffp10_binner_phiT(kphi, parfile, binning_type)
    cl = binner.get_cL_PHI_T()  # returns binned cross-spectrum

"""

import os
os.environ['PLENS'] = "/mnt/d/THESIS/PLENS"
os.environ['INPUT'] = "/mnt/d/THESIS/SIMS"
import numpy as np
import plancklens.bandpowers as bp
from plancklens import utils
import healpy as hp


import os
import numpy as np
from plancklens import utils
from plancklens.bandpowers import get_blbubc, ffp10_binner


class ffp10_binner_phiT:
    """Binner class for the phi-T cross-spectrum.
       Uses the Planck-style inverse variance weight:
         V_L^-1 ~ (2L+1)*f_sky/2 * R_L^2 / (CL^TT + NL^TT).
    """
    def __init__(self, kphi, parfile, btype):
        """
        Args:
          kphi : Phi estimator key, e.g. kphi='p_p', kphi='ptt' or similar.
          parfile  : A parameter file object that provides:
                     - .qresp_dd.get_response(kphi, 'p') for the lensing response R_L
                     - Something to retrieve TT and TT noise (or you can load from CAMB files).
          btype    : Bin edges descriptor (the same 'consext8', 'agr2', etc.).
          ksource  : By default 'pt' or 'phiT'; only used for naming or pass-through.
        """
        
        self.kphi = kphi
        self.parfile = parfile

        lmax = 2048

        # Load from parfile
        clpt_fid = parfile.cl_unl['pt'][:lmax+1]
        cltt_fid  = parfile.cl_unl['pt'][:lmax+1]
        nltt      = parfile.nl_TT[:lmax+1]
        R_L = parfile.qresp_dd.get_response(kphi, 'p')[:lmax+1]
        fsky = getattr(parfile.qcls_dd, 'fsky1234', 1.0)

        # Build the "inverse variance" weight array:
        #    Planck style: V_L^-1 ~ (2L+1) * fsky / 2 * R_L^2 / (cltt_fid[L] + nltt[L])
        ells = np.arange(lmax+1, dtype=float)
        vlpt_inv = (2.*ells + 1.) * fsky / 2.
        vlpt_inv *= R_L**2
        vlpt_inv /= (cltt_fid + nltt)


        # Define bin edges from btype:
        bin_lmins, bin_lmaxs, bin_centers = get_blbubc(btype)
        nbins = len(bin_centers)

        # Now define "vlpt_den" = sum of (fiducial^2 * vlpt_inv) over each bin:
        vlpt_den = []
        for lmin, lmax in zip(bin_lmins, bin_lmaxs):
            vlpt_den.append(
                np.sum( (clpt_fid[lmin:lmax+1])**2 * vlpt_inv[lmin:lmax+1] )
            )
        vlpt_den = np.array(vlpt_den)

        fid_bandpowers = np.ones(nbins, dtype=float)

        # Window function for each bin:
        #    Weighted by clpt_fid * vlpt_inv, normalized so that sum in bin = 1.
        def _get_bil(i, L):
            ret = fid_bandpowers[i] / vlpt_den[i] * vlpt_inv[L] * clpt_fid[L]
            # restrict to bin range:
            ret *= (L >= bin_lmins[i]) & (L <= bin_lmaxs[i])
            return ret

        # 10) Compute an "average L" in each bin (optional):
        lav = np.zeros(nbins)
        for i, (lmin, lmax) in enumerate(zip(bin_lmins, bin_lmaxs)):
            Lrange = np.arange(lmin, lmax+1, dtype=int)
            w_lav  = 1./(Lrange**2 * (Lrange+1.)**2)  # or whatever weighting you prefer
            numerator   = np.sum(Lrange * w_lav * _get_bil(i, Lrange))
            denominator = np.sum(w_lav * _get_bil(i, Lrange))
            lav[i]      = numerator / denominator

        # "Renormalize" the fiducial bandpowers at the average L:
        #     So that each bin is anchored to the fiducial cl at that bin's L_av.
        #     (Same logic as the original code.)
        updated_fid = np.interp(lav, ells, clpt_fid)
        for i in range(nbins):
            fid_bandpowers[i] = updated_fid[i]

        # store everything:
        self.bin_lmins = bin_lmins
        self.bin_lmaxs = bin_lmaxs
        self.bin_lavs  = lav
        self.nbins     = nbins
        self.fid_bandpowers = fid_bandpowers
        self.vlpt_inv  = vlpt_inv
        self.vlpt_den  = vlpt_den
        self.clpt_fid  = clpt_fid
        self.cl_pp = parfile.cl_unl['pp']
        self.cltt_fid  = cltt_fid
        self.nltt      = nltt
        self.R_L       = R_L
        self.lmax      = lmax
        self.fsky      = fsky
        self.N_PT = None
        self.lmin = None
        self.lmaxA = None
        bin_tmp = ffp10_binner(self.kphi, self.kphi, self.parfile, btype)
        self.n1 = bin_tmp.get_n1(unnormed=True)
        self.bmmc = None


    def _get_bil(self, i, L):
        """Returns the bin i window at multipole(s) L."""
        ret = (self.fid_bandpowers[i] / self.vlpt_den[i]) * self.vlpt_inv[L] * self.clpt_fid[L]
        ret *= (L >= self.bin_lmins[i]) & (L <= self.bin_lmaxs[i])
        return ret
    
    def fiducial_cl_scaled(self, binned=True):
        ell = np.arange(self.lmax + 1)**3
        clPT=self.clpt_fid[:self.lmax+1] * 1e2 * ell
        if binned: 
            return self._get_binnedcl(clPT)
        else:
            return clPT

    def _get_binnedcl(self, cl_pt):
        """Bins an externally provided phi-T cross-spectrum cl_pt[L]. 
           e.g. the measured cross-power or from a simulation.
        """
        assert len(cl_pt) > self.bin_lmaxs[-1], "cl_pt array is too short."
        ret = np.zeros(self.nbins, dtype=float)
        for i, (lmin, lmax) in enumerate(zip(self.bin_lmins, self.bin_lmaxs)):
            Lrange = np.arange(lmin, lmax+1)
            ret[i] = np.sum( self._get_bil(i, Lrange) * cl_pt[lmin:lmax+1] )
        return ret
    
    
    def get_WF(self, N0=False):
        """Returns Wiener Filter based on MCN0 lensing bias

        """

        ss = self.parfile.qcls_ss.get_sim_stats_qcl(self.kphi, self.parfile.mc_sims_var, k2=self.kphi).mean()[:self.lmax+1]
        qc_resp = self.parfile.qresp_dd.get_response(self.kphi, 'p')[:self.lmax+1]**2
        mcn0 = 2.*ss
        mcn0 *= utils.cli(qc_resp)
        cl_fid = self.cl_pp[:self.lmax+1]
        return cl_fid*utils.cli(cl_fid + mcn0)
    
    def get_cL_PHI_T(self, mc_sims = None, binned = True, input=False, scaled=False, error=False):
        if mc_sims is None: mc_sims = self.parfile.mc_sims_var

        if not input:
            if self.parfile.qcls_pt.WF is None: self.parfile.qcls_pt.WF = self.get_WF()[:self.lmax + 1]

            if not error:
                clPT = self.parfile.qcls_pt.get_sim_stats_qcl(self.kphi, mc_sims, lmax=self.lmax).mean()[:self.lmax + 1]
            else:
                clPT = self.parfile.qcls_pt.get_sim_stats_qcl(self.kphi, mc_sims, lmax=self.lmax)


            resp = self.parfile.qresp_dd.get_response(self.kphi, 'p')[:self.lmax + 1]
            clPT *= utils.cli(resp)

        else:
            clPT = self.parfile.qcls_pt.get_sim_stats_qcl('input', mc_sims, lmax=self.lmax).mean()[:self.lmax + 1]

        if scaled:
            ell = np.arange(self.lmax + 1)**3
            clPT *= 1e2 * ell

        if binned:
            return self._get_binnedcl(clPT)
        else:
            return clPT
        
    def get_cL_PHI_T_with_error(self,
                           mc_sims=None,
                           binned=True,
                           input=False,
                           scaled=False,
                           cross_maps=False):
        """
        Compute the mean phi–T cross‐spectrum (binned or not) together with 
        Monte Carlo error bars (1σ on the mean) across sims.

        Returns
        -------
        cl_bar : array
            The mean cross‐spectrum (length = nbins if binned else lmax+1).
        cl_err : array
            The 1σ error on the mean in each bin (or multipole).
        """
        import numpy as np
        from plancklens.utils import stats, cli

        # default sim list
        if mc_sims is None:
            mc_sims = self.parfile.mc_sims_var

        # figure out output dimension
        if binned:
            # run a dummy through your binning to get nbins
            dummy = np.zeros(self.lmax + 1)
            nbins = self._get_binnedcl(dummy).size
            stats_qcl = stats(nbins, docov=True)
        else:
            stats_qcl = stats(self.lmax + 1, docov=True)

        resp = cli(self.parfile.qresp_dd.get_response(self.kphi, 'p')[:self.lmax + 1])

        # loop over sims
        for idx in mc_sims:
            # get raw (unbinned) cl_PT for this sim
            if cross_maps:
                raw = self.parfile.qcls_pt_ss.get_sim_qcl(self.kphi, idx, lmax=self.lmax)
            else:
                raw = self.parfile.qcls_pt.get_sim_qcl(self.kphi, idx, lmax=self.lmax)

            # apply response filter if this is not the 'input' case
            if not input:
                raw = raw * resp

            # apply any ℓ³ scaling
            if scaled:
                ell = np.arange(self.lmax + 1)**3
                raw = raw * (1e2 * ell)

            # bin if requested
            if binned:
                vec = self._get_binnedcl(raw)
            else:
                vec = raw

            # accumulate into our stats object
            stats_qcl.add(vec)

        # compute mean and 1σ error on the mean
        cl_bar = stats_qcl.mean()
        cl_err = stats_qcl.sigmas_on_mean()

        return cl_bar, cl_err


class binner_sims(bp.ffp10_binner):
    def __init__(self, k1, k2, parfile, btype):
        super(binner_sims, self).__init__(k1, k2, parfile=parfile, btype=btype)

        lmax_pt = 2048

        self.cl_pp = self.parfile.cl_unl['pp']
        self.cl_pt = self.parfile.cl_unl['pt'][:lmax_pt + 1]
        self.cl_tt = self.parfile.cl_unl['tt'][:lmax_pt + 1]


    def get_WF(self, binned = False, N0=False):
        """Returns Wiener Filter based on MCN0 lensing bias

        """

        ss = self.parfile.qcls_ss.get_sim_stats_qcl(self.k1, self.parfile.mc_sims_var, k2=self.k2).mean()
        qc_resp = self.parfile.qresp_dd.get_response(self.k1, self.ksource) * self.parfile.qresp_dd.get_response(self.k2, self.ksource)
        mcn0 = 2.*ss
        lmax = mcn0.shape[0]
        if N0: mcn0 += self.get_n1(unnormed=True)[:lmax]
        mcn0 *= utils.cli(qc_resp)
        cl_fid = self.cl_pp[:lmax]

        if binned:
            return self._get_binnedcl(cl_fid*utils.cli(cl_fid + mcn0))
        else:
            return cl_fid*utils.cli(cl_fid + mcn0)
        
    def get_mcn0(self, mc_sims = None, unbinned = False):
        """Returns Monte-Carlo N0 lensing bias.

        """
        if mc_sims is None:
            mc_sims = self.parfile.mc_sims_var

        ss = self.parfile.qcls_ss.get_sim_stats_qcl(self.k1, mc_sims, k2=self.k2).mean()
        qc_resp = self.parfile.qresp_dd.get_response(self.k1, self.ksource) * self.parfile.qresp_dd.get_response(self.k2, self.ksource)
        
        if unbinned:
            return utils.cli(qc_resp) * (2. * ss)
        else:
            return self._get_binnedcl(utils.cli(qc_resp) * (2. * ss))
    

    def get_sims_lensing_amplitude(self, mc_sims=None):
        """Computes the lensing amplitude for all sims averaged

        """

        if mc_sims is None:
            mc_sims = self.parfile.mc_sims_var

        # First get spectrum, bin it, and correct for N0 and N1
        binned_input = self.get_sims_bandpowers(mc_sims) -  self.get_mcn0(mc_sims) - self.get_n1()

        # Calculate lensing amplitude by normalizing to the fiducial band-powers
        fid_bandpowers = self.get_fid_bandpowers()
        amplitude = np.sum((binned_input-fid_bandpowers)/fid_bandpowers)/self.nbins+1

        return amplitude
    
    def get_single_sim_lensing_amplitude(self, idx):
        """Computes the lensing amplitude for a single sim

        """

        # First get spectrum, bin it, and correct for N0 and N1
        binned_input = self.get_single_sim_bandpowers(idx) - self.get_mcn0() - self.get_n1()

        # Calculate lensing amplitude by normalizing to the fiducial band-powers
        fid_bandpowers = self.get_fid_bandpowers()
        amplitude = np.sum((binned_input-fid_bandpowers)/fid_bandpowers)/self.nbins+1

        return amplitude
    
    def get_dat_lensing_amplitude(self):
        """Computes the lensing amplitude for the data

        """

        # First get spectrum, bin it, and correct for N0 and N1
        binned_input = self.get_dat_bandpowers() - self.get_rdn0() - self.get_n1() - self.get_ps_corr()

        # Calculate lensing amplitude by normalizing to the fiducial band-powers
        fid_bandpowers = self.get_fid_bandpowers()
        amplitude = np.sum((binned_input-fid_bandpowers)/fid_bandpowers)/self.nbins+1

        return amplitude

    def get_sims_clpp(self, mc_sims = None):
        if mc_sims is None:
            mc_sims = self.parfile.mc_sims_var
        dd = self.parfile.qcls_dd.get_sim_stats_qcl(self.k1, mc_sims, k2=self.k2).mean()
        qc_resp = self.parfile.qresp_dd.get_response(self.k1, self.ksource) * self.parfile.qresp_dd.get_response(self.k2, self.ksource)
        return utils.cli(qc_resp) * dd

    def get_sims_bandpowers(self, mc_sims = None, unbinned = False):
        """Returns average of sims in bandpower

        """
        if mc_sims is None:
            mc_sims = self.parfile.mc_sims_var
        
        dd = self.parfile.qcls_dd.get_sim_stats_qcl(self.k1, mc_sims, k2=self.k2).mean()
        qc_resp = self.parfile.qresp_dd.get_response(self.k1, self.ksource) * self.parfile.qresp_dd.get_response(self.k2, self.ksource)
        if unbinned:
            return utils.cli(qc_resp) * dd
        else:
            return self._get_binnedcl(utils.cli(qc_resp) * dd)
        
    def get_sims_bandpowers_with_error(self, mc_sims = None, unbinned = False):
        """Returns average of sims in bandpower

        """
        if mc_sims is None:
            mc_sims = self.parfile.mc_sims_var
        
        dd = self.parfile.qcls_dd.get_sim_stats_qcl(self.k1, mc_sims, k2=self.k2).mean()
        qc_resp = self.parfile.qresp_dd.get_response(self.k1, self.ksource) * self.parfile.qresp_dd.get_response(self.k2, self.ksource)
        if unbinned:
            return utils.cli(qc_resp) * dd
        else:
            return self._get_binnedcl(utils.cli(qc_resp) * dd)
        

    def get_sims_bandpowers_with_error(self, mc_sims=None, unbinned=False):
        """
        Compute the mean Phi–Phi bandpowers (binned or unbinned) and 1σ errors
        on the mean from a suite of FFP10 sims.
        
        Returns
        -------
        band_mean : ndarray
            Mean bandpowers (length = nbins if binned else ℓmax+1).
        band_err : ndarray
            1σ error on the mean in each bin (or ℓ).
        """
        from tqdm import tqdm
        import numpy as np
        from plancklens.utils import stats, cli

        # default sims list
        if mc_sims is None:
            mc_sims = self.parfile.mc_sims_var

        # build the combined response vector once
        resp1 = self.parfile.qresp_dd.get_response(self.k1, self.ksource)
        resp2 = self.parfile.qresp_dd.get_response(self.k2, self.ksource)
        qc_resp = resp1 * resp2           # length = ℓmax+1

        # determine output dimension
        if unbinned:
            size = qc_resp.size
        else:
            # feed zeros through your binning to learn # of bins
            dummy = np.zeros_like(qc_resp)
            size = self._get_binnedcl(dummy).size

        # accumulator with covariance tracking
        stats_bp = stats(size, docov=True)

        # loop over sims
        print("Binner sims, calculating SIGMA...")
        for idx in tqdm(mc_sims):
            # raw auto-spectrum for this sim
            dd_sim = self.parfile.qcls_dd.get_sim_qcl(self.k1, idx, k2=self.k2)

            # apply the ℓ-filter (cli) and response
            vec = cli(qc_resp) * dd_sim

            # bin if requested
            if not unbinned:
                vec = self._get_binnedcl(vec)

            # accumulate
            stats_bp.add(vec)

        # ensemble mean + 1σ error on the mean
        band_mean = stats_bp.mean()
        band_err  = stats_bp.sigmas_on_mean()

        return band_mean, band_err




    def get_single_sim_bandpowers(self, idx):
        """Returns bandpowers of a single simulation

        """
        dd = self.parfile.qcls_dd.get_sim_qcl(self.k1, idx, self.k2)
        qc_resp = self.parfile.qresp_dd.get_response(self.k1, self.ksource) * self.parfile.qresp_dd.get_response(self.k2, self.ksource)
        return self._get_binnedcl(utils.cli(qc_resp) * dd)