import numpy as np
import matplotlib.pyplot as plt

from orphics import maps, cosmology, io, pixcov, mpi, stats
from falafel import qe, utils as futils
from pixell import enmap, curvedsky as cs, lensing as plensing, enplot
import healpy as hp
#from healpy.fitsfunc import read_alm
import pytempura
import cmb_ps
import pymaster as nmt
from scipy.interpolate import interp1d

import time

PATH_TO_SCRATCH = "/global/cscratch1/sd/jaejoonk/"
ESTS = ['TT']
RESOLUTION = np.deg2rad(0.5/60.)
full_shape, full_wcs = enmap.fullsky_geometry(res=RESOLUTION)
t1 = time.time()

# constants
LMIN = 600
LMAX = 4000
MLMAX = 6000

BEAM_FWHM = 1.5
NOISE_T = 10.
NSIDE = 1024

# let's convert our lensed alms to a map
PATH_TO_SCRATCH = "/global/cscratch1/sd/jaejoonk/"
KAP_FILENAME = "websky/kap.fits"
ALM_FILENAME = "websky/lensed_alm.fits"
OVERDENSITY_FILENAME = PATH_TO_SCRATCH + "galaxy_delta_hp_2.fits"
MASK_FILENAME = PATH_TO_SCRATCH + "dr6_lensing_mask_healpix_analysis.fits"

RAW_SPECTRA_FILENAME = "ps_websky_overdensity_spectra.png"
CROSS_SPECTRA_FILENAME = "ps_websky_overdensity_cross_vs_auto.png"
RELATIVE_DIFF_FILENAME = "ps_websky_overdensity_relative_diff.png"

# returns kappa alms, input alms are 1d in shape
def lensing_reconstruction(alms, est=ESTS[0], lmin=LMIN, lmax=LMAX, mlmax=MLMAX):
    # try reconstructing?
    ucls, tcls = cmb_ps.get_theory_dicts_white_noise_websky(BEAM_FWHM, NOISE_T, grad=True, lmax=mlmax)
    Xdats = futils.isotropic_filter([alms, alms*0., alms*0.], tcls, lmin=lmin, lmax=lmax)
    px = qe.pixelization(shape=full_shape, wcs=full_wcs)

    recon_alms = qe.qe_all(px, ucls, mlmax=MLMAX,
                           fTalm=Xdats[0], fEalm=Xdats[1], fBalm=Xdats[2],
                           estimators=ESTS, xfTalm=None, xfEalm=None, xfBalm=None)

    # normalize using tempura
    Al = pytempura.get_norms(ESTS,ucls,tcls,LMIN,LMAX,k_ellmax=MLMAX,no_corr=False)

    # return normalization as well
    return plensing.phi_to_kappa(hp.almxfl(recon_alms[est][0].astype(np.complex128), Al[est][0])), Al[est][0]

def cross_correlate_kappa(raw_spectra_filename=RAW_SPECTRA_FILENAME,
                          cross_spectra_filename=CROSS_SPECTRA_FILENAME,
                          relative_diff_filename=RELATIVE_DIFF_FILENAME):
    ikalm = futils.change_alm_lmax(hp.map2alm(hp.read_map(KAP_FILENAME)), MLMAX)
    icls = hp.alm2cl(ikalm,ikalm)
    ells = np.arange(len(icls))

    lensed_alm = futils.change_alm_lmax(hp.read_alm(ALM_FILENAME), lmax=MLMAX)
    galaxy_map = hp.read_map(OVERDENSITY_FILENAME)
    overdensity_alm = hp.map2alm(galaxy_map, lmax=MLMAX)

    # plot cross spectra vs auto spectra
    norm_recon_alms, Al_l = lensing_reconstruction(lensed_alm)

    # use this if first bin object is doing linear bins
    bin_edges = np.geomspace(2,LMAX,30)
    binner = stats.bin1D(bin_edges)
    bin = lambda x: binner.bin(ells,x)

    pl = io.Plotter('CL', figsize=(16,12))
    pl2 = io.Plotter('rCL',xyscale='loglin',figsize=(16,12))
    pl3 = io.Plotter('rCL',xyscale='loglin',figsize=(16,12))

    ocls = hp.alm2cl(overdensity_alm, overdensity_alm)
    rcls = hp.alm2cl(norm_recon_alms, norm_recon_alms)

    rxi_cls = hp.alm2cl(norm_recon_alms, ikalm)
    rxo_cls = hp.alm2cl(norm_recon_alms, overdensity_alm)
    ixo_cls = hp.alm2cl(ikalm, overdensity_alm)

    pl.add(ells,(ells*(ells+1.)/2.)**2. * Al_l, ls='--', label="noise PS (per mode)")
    pl.add(ells,rxo_cls,label='recon x galaxy')
    pl.add(ells,rxi_cls,label='recon x input kappa')
    pl.add(ells,icls,label='input kappa x input kappa')
    pl.add(ells,ocls,label='galaxy x galaxy')

    pl2.add(*bin(ixo_cls),marker='o',label="input x galaxy Clkg")
    pl2.add(*bin(rxo_cls),marker='o',label="recon x galaxy Clkg")
    pl2._ax.set_xlabel(r'$L$', fontsize=32)
    pl2._ax.set_ylabel(r'$C_L^{\kappa_i g}$', fontsize=24)
    pl2._ax.legend(fontsize=30)
    pl2.hline(y=0)
    #pl2._ax.set_ylim(-0.3,0.3)

    pl3.add(*bin((ixo_cls - rxo_cls) / rxo_cls), marker='o', label="input kappa vs recon")
    pl3._ax.set_xlabel(r'$L$', fontsize=32)
    pl3._ax.set_ylabel(r'$(C_L^{\kappa g} - C_L^{\hat{\kappa}_{\rm recon} g}) / C_L^{\hat{\kappa}_{\rm recon} g}$', fontsize=24)
    pl3._ax.legend(fontsize=30)
    pl3.hline(y=0)
    pl3._ax.set_ylim(-0.4,0.4)

    pl.done(raw_spectra_filename)
    pl2.done(cross_spectra_filename)
    pl3.done(relative_diff_filename)

    print("Time elapsed: %0.5f seconds" % (time.time() - t1))

def cross_correlate_kappa_nmt(mask=None, ells_per_bp=4,
                              raw_spectra_filename=RAW_SPECTRA_FILENAME,
                              cross_spectra_filename=CROSS_SPECTRA_FILENAME,
                              relative_diff_filename=RELATIVE_DIFF_FILENAME):

    # overdensity map
    galaxy_map = hp.pixelfunc.ud_grade(hp.read_map(OVERDENSITY_FILENAME), NSIDE)
    print(f"Overdensity nside: {hp.get_nside(galaxy_map)}")
    mask_galaxy = np.ones(shape=galaxy_map.shape) if mask is None else mask
    f_galaxy = nmt.NmtField(mask_galaxy, [galaxy_map])
    b_g = nmt.NmtBin.from_nside_linear(hp.get_nside(galaxy_map), ells_per_bp)
    
    # input kappa map
    kappa_map = hp.pixelfunc.ud_grade(hp.read_map(KAP_FILENAME), hp.get_nside(galaxy_map))
    print(f"Kappa nside: {hp.get_nside(kappa_map)}")
    b = nmt.NmtBin.from_nside_linear(hp.get_nside(kappa_map), ells_per_bp)
    mask_kappa = np.ones(shape=kappa_map.shape) if mask is None else mask
    f_kappa = nmt.NmtField(mask_kappa, [kappa_map])

    icls = nmt.compute_full_master(f_kappa, f_kappa, b)
    ells = b.get_effective_ells()

    # lensed alms
    lensed_alm = futils.change_alm_lmax(hp.read_alm(ALM_FILENAME), lmax=MLMAX)
    # plot cross spectra vs auto spectra
    norm_recon_alms, Al_l = lensing_reconstruction(lensed_alm)
    norm_recon_map = cs.alm2map_healpix(norm_recon_alms, nside=hp.get_nside(galaxy_map))
    print(f"Lensed alms nside: {hp.get_nside(norm_recon_map)}")
    mask_recon = np.ones(shape=norm_recon_map.shape) if mask is None else mask
    f_recon = nmt.NmtField(mask_recon, [norm_recon_map])
    b_r = nmt.NmtBin.from_nside_linear(hp.get_nside(norm_recon_map), ells_per_bp)

    # use this if first bin object is doing linear bins
    bin_edges = np.geomspace(2,LMAX,20)
    binner = stats.bin1D(bin_edges)
    bin = lambda x: binner.bin(ells,x)

    pl = io.Plotter('CL', figsize=(16,12))
    pl2 = io.Plotter('rCL',xyscale='loglin',figsize=(16,12))
    pl3 = io.Plotter('rCL',xyscale='loglin',figsize=(16,12))

    ocls = nmt.compute_full_master(f_galaxy, f_galaxy, b_g)
    #ocls = hp.alm2cl(overdensity_alm, overdensity_alm)
    rcls = nmt.compute_full_master(f_recon, f_recon, b_r)
    #rcls = hp.alm2cl(norm_recon_alms, norm_recon_alms)

    rxi_cls = nmt.compute_full_master(f_recon, f_kappa, b_r)
    rxo_cls = nmt.compute_full_master(f_recon, f_galaxy, b_g)
    ixo_cls = nmt.compute_full_master(f_kappa, f_galaxy, b_g)

    #rxi_cls = hp.alm2cl(norm_recon_alms, ikalm)
    #rxo_cls = hp.alm2cl(norm_recon_alms, overdensity_alm)
    #ixo_cls = hp.alm2cl(ikalm, overdensity_alm)
    #Al_fn = interp1d(np.arange(len(Al_l)), Al_l)
    #pl.add(ells,(ells*(ells+1.)/2.)**2. * Al_fn(ells), ls='--', label="noise PS (per mode)")

    pl.add(ells,rxo_cls[0],label='recon x galaxy')
    pl.add(ells,rxi_cls[0],label='recon x input kappa')
    pl.add(ells,icls[0],label='input kappa x input kappa')
    pl.add(ells,ocls[0],label='galaxy x galaxy')

    pl2.add(ells,ixo_cls[0], marker='o',label="input x galaxy Clkg")
    pl2.add(ells,rxo_cls[0], marker='o',label="recon x galaxy Clkg")
    pl2._ax.set_xlabel(r'$L$', fontsize=32)
    pl2._ax.set_ylabel(r'$C_L^{\kappa_i g}$', fontsize=24)
    pl2._ax.legend(fontsize=30)
    pl2.hline(y=0)
    #pl2._ax.set_ylim(-0.3,0.3)

    pl3.add(*bin((rxo_cls[0] - ixo_cls[0]) / ixo_cls[0]), marker='o', label="C(recon,g) vs C(input,g)")
    pl3.add(*bin((rxi_cls[0] - icls[0]) / icls[0]), marker='o', label="C(recon,input) vs C(input,input)")
    pl3._ax.set_xlabel(r'$L$', fontsize=32)
    pl3._ax.set_ylabel(r'$ \Delta C_L^{\kappa g} / C_L^{\kappa g}$', fontsize=24)
    pl3._ax.legend(fontsize=30)
    pl3.hline(y=0)
    pl3._ax.set_ylim(-0.2,0.2)

    pl.done(raw_spectra_filename)
    pl2.done(cross_spectra_filename)
    pl3.done(relative_diff_filename)

    print("Time elapsed: %0.5f seconds" % (time.time() - t1))

if __name__ == '__main__':
    #cross_correlate_kappa()
    mask = hp.pixelfunc.ud_grade(hp.read_map(MASK_FILENAME), NSIDE)
    cross_correlate_kappa_nmt(mask=None,
                              raw_spectra_filename="ps_websky_overdensity_nmt_spectra.png",
                              cross_spectra_filename= "ps_websky_overdensity_nmt_cross_vs_auto.png",
                              relative_diff_filename= "ps_websky_overdensity_nmt_relative_diff.png")