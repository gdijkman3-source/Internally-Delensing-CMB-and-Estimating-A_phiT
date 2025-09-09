from __future__ import print_function
import os
import healpy as hp
import numpy as np
import pickle as pk

"""
qecl_pt.py

This module defines the library_phiT class to compute and cache phi-T cross-spectra,
providing methods to generate and manage quadrature lensing maps (QLMs), subtract mean-field,
compute cross-spectra, and gather statistics over simulations.

Classes:
  library_phiT: Build, cache, and retrieve cross-spectra between lensing phi estimators and temperature maps.

Functions:
  All class methods include docstrings describing their behavior and usage.
"""

from plancklens.helpers import mpi, sql
from plancklens import utils

class library_phiT(object):
    r"""Library that builds phi-T cross-spectra from:
         - A lensing QE instance (for phi)
         - Temperature maps or simulations (for T)
         - Possibly a phi mean-field set to subtract from phi.

       This is analogous to your original library for QE-QE cross-spectra,
       but specialized to phi x T.

       The cross-spectrum is formed as
         (1 / [(2L+1) fsky]) * sum_m [ (phi_{Lm} - MF_{Lm}) * T_{Lm}^* ].

       The code caches results in 'lib_dir' similarly to your original approach.
    """

    def __init__(self, lib_dir, qe_phi, tmaps, mc_sims_mf, ftl=None):
        """
        Args:
          lib_dir: Output directory for caching cross-spectra files
          qe_phi:  A lensing QE instance for phi (like qeA in your code).
                   Must implement:
                     - get_sim_qlm(k, idx, lmax) -> returns alm of phi QE for simulation 'idx'
                     - get_sim_qlm_mf(k, mc_sims, lmax) -> returns the mean-field alm
                     - get_mask(a) -> returns the sky mask if needed
                     - .keys -> The allowed anisotropy keys (like 'p_p', 'p_tp', etc.)
                     - get_lmax_qlm(k) -> returns max l for the QE
          tmaps:   An object or library for temperature maps. Must provide:
                     - get_sim_tlm(idx, lmax) -> returns T_{lm} for sim index
                   If you want to handle the "data" case, you might define idx=-1 for data.
          mc_sims_mf: List of simulation indices for building / subtracting the phi MF.
                      Typically even indices for the first leg (phi).
                      (We do not do a second leg MF, since T isn't a QE.)
        """
        self.lib_dir = lib_dir
        self.qe_phi  = qe_phi
        self.tmaps   = tmaps
        self.mc_sims_mf = mc_sims_mf
        if ftl is None: 
            self.ftl=1 
        else: 
            self.ftl = ftl

        if mpi.rank == 0:
            if not os.path.exists(lib_dir):
                os.makedirs(lib_dir)
        mpi.barrier()

        # The code below is an example of storing sky fractions fskies in a .dat file,
        # just like your original library. We'll replicate that approach for uniformity.
        fsname = os.path.join(lib_dir, 'fskies_phiT.dat')
        hname  = os.path.join(lib_dir, 'qcl_sim_hash_phiT.pk')
        self.MF = None
        self.ells = None
        self.WF = None

        if mpi.rank == 0:
            if not os.path.exists(fsname):
                print('Caching sky fractions (phi-T cross) ...')
                # Suppose 'get_mask(1)' is your main mask for the phi QE
                # For T, you might have a separate mask, e.g. tmaps.get_mask()
                mask_phi = self.qe_phi.get_mask(1)
                mask_t   = self.tmaps.get_mask() if hasattr(self.tmaps, 'get_mask') else np.ones_like(mask_phi)
                assert mask_phi.shape == mask_t.shape, "Masks not the same resolution or shape!"
                fsky_phiT = np.mean(mask_phi * mask_t)

                with open(fsname, 'w') as f:
                    f.write('1234 %.6f\n' % fsky_phiT)

            if not os.path.exists(hname):
                pk.dump(self.hashdict(), open(hname, 'wb'), protocol=2)
        mpi.barrier()

        # Check if the library is consistent with stored hash:
        utils.hash_check(pk.load(open(hname, 'rb')), self.hashdict(), fn=hname)

        # Minimal approach: we store our cross-spectra in an sqlite db as in original:
        self.npdb = sql.npdb(os.path.join(lib_dir, 'cldb_phiT.db'))

        # Load the single fsky number:
        with open(fsname) as f:
            line = f.readline()
            (key, val) = line.split()
            self.fsky1234 = float(val)   # the fraction of sky used in phi x T

    def hashdict(self):
        """Dictionary to ensure that the library is consistent with the expected pipeline."""
        return {
            'qe_phi'      : self.qe_phi.hashdict(),
            'tmaps_hash'  : self._tmaps_hash(),
            'mc_sims_mf'  : utils.mchash(self.mc_sims_mf)
        }

    def _tmaps_hash(self):
        """Tries to build a suitable hash for the T maps object, e.g. if it has a .hashdict() method."""
        if hasattr(self.tmaps, 'hashdict'):
            return self.tmaps.hashdict()
        return {'tmaps': 'no hashdict'}

    def get_lmaxqcl(self, kPhi):
        """Max L for the phi QE at key kPhi."""
        return self.qe_phi.get_lmax_qlm(kPhi)

    def load_sim_qcl(self, kPhi, idx, lmax=None):
        """Same as get_sim_qcl but does not trigger calculation if not present."""
        return self.get_sim_qcl(kPhi, idx, lmax=lmax, calc=False)

    def get_sim_qcl(self, kPhi, idx, lmax=None, recache=False, calc=True):
        """Returns phi-T cross-spectrum for simulation 'idx'.

           If idx == -1, you could interpret that as "data" (like original code).
           Otherwise, idx is a simulation index.

           Args:
             kPhi: QE anisotropy key for the phi QE
             idx : simulation index (int)
             lmax: optional maximum multipole for the returned cross-spectrum
             calc: whether to compute it if not in the cache

           Returns:
             cross-spectrum array (length lmax_out + 1)
        """

        if kPhi == "input":
            inputKlm = True
            lmax_qcl = lmax
            lmax_out = lmax
        else:
            assert kPhi in self.qe_phi.keys, (kPhi, self.qe_phi.keys)
            lmax_qcl = min(self.get_lmaxqcl(kPhi), lmax) if lmax is not None else self.get_lmaxqcl(kPhi)
            
            try:
                lmax_ivfs = self.tmaps.cinv_t.lmax
            except:
                lmax_ivfs = 2048
            
            lmax_qcl = min(lmax_qcl, lmax_ivfs)
            # print("QECL PT: lmax qcl = %i"%lmax_qcl)
            lmax_out = min(lmax, lmax_qcl) if lmax is not None else lmax_qcl
            # print("QECL PT: lmax out = %i"%lmax_out)
            assert lmax_out <= lmax_qcl
            inputKlm = False

        if idx >= 0:
            fname = os.path.join(self.lib_dir,
                     'sim_qcl_phiT_kPhi%s_lmax%s_%04d_%s.dat' % (kPhi, lmax_qcl, idx, utils.mchash(self.mc_sims_mf)))
        else:
            assert idx == -1  # data case
            fname = os.path.join(self.lib_dir,
                     'dat_qcl_phiT_kPhi%s_lmax%s_%s.dat' % (kPhi, lmax_qcl, utils.mchash(self.mc_sims_mf)))

        if calc and (self.npdb.get(fname) is None or recache):

            if inputKlm:
                klmfilepath = os.path.join(os.environ["KFIELD"], "klm_%03d.fits"%idx)
                qlm_phi = utils.alm_copy(hp.read_alm(klmfilepath), lmax)

                if self.ells is None:
                    self.ells = np.arange(lmax+1, dtype=np.float64)
                    self.ells *= self.ells + 1
                    self.ells /= 2
                    self.ells = utils.cli(self.ells)

                hp.almxfl(qlm_phi, self.ells, inplace=True)

            else:
                qlm_phi = self.qe_phi.get_sim_qlm(kPhi, idx, lmax=lmax_qcl)
                # 2) subtract mean-field from phi:
                if self.MF is None:
                    self.MF = self.qe_phi.get_sim_qlm_mf(kPhi, self.mc_sims_mf, lmax=lmax_qcl)
                qlm_phi -= self.MF

            # print("using ftl")
            tlm = utils.alm_copy(self.tmaps.get_sim_tmliklm(idx), lmax_qcl)
            # hp.almxfl(tlm, self.ftl,inplace=True)

            cl_x = hp.alm2cl(qlm_phi, alms2=tlm)
            cl_x /= self.fsky1234   # it's (1 / [(2L+1)*fsky]) sum_m [phi_{Lm} T_{Lm}^*]
            #    (In reality, hp.alm2cl does (1/(2L+1)) sum_m, so we still divide by fsky
            #     to match the original library's convention.)

            if recache and self.npdb.get(fname) is not None:
                self.npdb.remove(fname)
            self.npdb.add(fname, cl_x)
            del qlm_phi, tlm

        cl_out = self.npdb.get(fname)
        if cl_out is None:
            raise ValueError("Cross-spectrum not found in DB and calc=False: " + fname)
        return cl_out[:lmax_out + 1]

    def get_sim_stats_qcl(self, kPhi, mc_sims, recache=False, lmax=None):
        """Returns average cross-spectrum for a list of sim indices."""
        if lmax is None:
            lmax = min(self.get_lmaxqcl(kPhi), self.tmaps.cinv_t.lmax)
        tfname = os.path.join(self.lib_dir,
                  'sim_qcl_stats_phiT_%s_%s.pk' % (kPhi, utils.mchash(mc_sims)))
        if not os.path.exists(tfname) or lmax > pk.load(open(tfname, 'rb'))[1] or recache:
            stats_qcl = utils.stats(lmax + 1, docov=False)
            for i, idx in utils.enumerate_progress(mc_sims, label='sim_stats phi-T (kPhi)=' + str(kPhi)):
                stats_qcl.add(self.get_sim_qcl(kPhi, idx, lmax=lmax))
            pk.dump((stats_qcl, lmax), open(tfname, 'wb'), protocol=2)
        return pk.load(open(tfname, 'rb'))[0]


