import numpy as np
import matplotlib.pyplot as plt

from orphics import maps, cosmology, io, pixcov, mpi, stats
from falafel import qe, utils as futils
from pixell import enmap, curvedsky as cs, lensing as plensing, enplot
import healpy as hp
from healpy.fitsfunc import read_alm
import pytempura
from mpi4py import MPI

#import websky_lensing_reconstruction as wlrecon
import cmb_ps

import time

ESTS = ['TT']
RESOLUTION = np.deg2rad(0.5/60.)
COMM = MPI.COMM_WORLD
rank = COMM.Get_rank() 
full_shape, full_wcs = enmap.fullsky_geometry(res=RESOLUTION)
t1 = time.time()

# constants
LMIN = 600
LMAX = 4000
MLMAX = 6000

BEAM_FWHM = 1.5
NOISE_T = 10.

# let's convert our lensed alms to a map
PATH_TO_SCRATCH = "/global/cscratch1/sd/jaejoonk/"
#KAP_FILENAME = PATH_TO_SCRATCH + "sehgal/tSZ_skymap_healpix_nopell_Nside4096_y_tSZrescale0p75.fits"
KAP_FILENAME = "websky/kap.fits"
ALM_FILENAME = "websky/lensed_alm.fits"
FILTERED_MAP_FILENAME = PATH_TO_SCRATCH + "optimal_filtered_websky_empty.fits"

CLTT_OUTPUT_NAME = f"ps_cltt_optimal_websky_empty.png"

ikalm = futils.change_alm_lmax(hp.map2alm(hp.read_map(KAP_FILENAME)), MLMAX)
icls = hp.alm2cl(ikalm,ikalm)
ells = np.arange(len(icls))

# binning
#bin_edges = np.geomspace(2,MLMAX,30)
bin_edges = np.arange(2,LMAX,30)
binner = stats.bin1D(bin_edges)
bin = lambda x: binner.bin(np.arange(LMAX+1),x)

# convolve beam before cltt
BEAM_FN = lambda ells: maps.gauss_beam(ells, BEAM_FWHM)

# from inpainted map
filtered_map = enmap.read_map(FILTERED_MAP_FILENAME)
filtered_alm = cs.map2alm(filtered_map, lmax=LMAX)

unfiltered_alm = futils.change_alm_lmax(read_alm(ALM_FILENAME), lmax=LMAX)

filtered_alm = cs.almxfl(filtered_alm, BEAM_FN)
unfiltered_alm = cs.almxfl(unfiltered_alm, BEAM_FN)

white_noise_map = maps.white_noise(shape=full_shape, wcs=full_wcs, noise_muK_arcmin=NOISE_T)
unfiltered_alm = cs.map2alm(cs.alm2map(unfiltered_alm,
                                       enmap.empty(full_shape, full_wcs, dtype=np.float32)) \
                            + white_noise_map, lmax=LMAX)

ucls, tcls = cmb_ps.get_theory_dicts_white_noise_websky(BEAM_FWHM, NOISE_T, grad=False, lmax=MLMAX)
ucls['TT'] = ucls['TT'][:LMAX+1]
tcls['TT'] = tcls['TT'][:LMAX+1]

iso_alm = futils.isotropic_filter([unfiltered_alm, unfiltered_alm*0., unfiltered_alm*0.],
                                  tcls, LMIN, LMAX)[0]

# plot cltt
pl_tt = io.Plotter('rCL',xyscale='linlin',figsize=(12,12))
pl_tt2 = io.Plotter('rCL',xyscale='linlin',figsize=(12,12))
unfiltered_cls = hp.alm2cl(iso_alm.astype(np.cdouble),
                           iso_alm.astype(np.cdouble),
                           lmax=LMAX)
filtered_cls = hp.alm2cl(filtered_alm.astype(np.cdouble),
                         filtered_alm.astype(np.cdouble),   
                         lmax=LMAX)

pl_tt.add(*bin((filtered_cls - unfiltered_cls) / unfiltered_cls), marker='o', label="optimal filter vs isotropic filter")
pl_tt._ax.set_ylabel(r'$(C_L^{T_{OF} T_{OF}} - C_L^{TT}) /  C_L^{TT}$', fontsize=20)
pl_tt._ax.set_xlabel(r'$L$', fontsize=20)
pl_tt._ax.legend(fontsize=30)
pl_tt.hline(y=0)
pl_tt.done(CLTT_OUTPUT_NAME)

pl_tt2.add(*bin(filtered_cls), marker='o', label="optimal filtered Cltt")
pl_tt2.add(*bin(unfiltered_cls), marker='o', label="isotropic filtered Cltt")
pl_tt2._ax.set_xlabel(r'$L$', fontsize=20)
pl_tt2._ax.set_ylabel(r'$C_L$', fontsize=20)
pl_tt2._ax.legend(fontsize=30)
pl_tt2.done(f"ps_raw_cltt_optimal_websky_empty_{ESTS[0]}.png")

# try reconstructing?
INV_BEAM_FN = lambda ells: 1./maps.gauss_beam(ells, BEAM_FWHM)
# deconvolve beam before reconstruction
f_alm = cs.almxfl(filtered_alm, INV_BEAM_FN)
iso_alm = cs.almxfl(iso_alm, INV_BEAM_FN)

px = qe.pixelization(shape=full_shape, wcs=full_wcs)

frecon_alms = qe.qe_all(px, ucls, mlmax=MLMAX,
                        fTalm=f_alm, fEalm=f_alm*0., fBalm=f_alm*0.,
                        estimators=ESTS,
                        xfTalm=None, xfEalm=None, xfBalm=None)

recon_alms = qe.qe_all(px, ucls, mlmax=MLMAX,
                       fTalm=iso_alm, fEalm=iso_alm*0., fBalm=iso_alm*0.,
                       estimators=ESTS,
                       xfTalm=None, xfEalm=None, xfBalm=None)

# normalize using tempura
Al = pytempura.get_norms(ESTS,ucls,tcls,LMIN,LMAX,k_ellmax=MLMAX,no_corr=False)

# plot cross spectra vs auto spectra
norm_recon_alms = {}
norm_frecon_alms = {}

# use this if first bin object is doing linear bins
bin_edges2 = np.geomspace(2,LMAX,40)
binner2 = stats.bin1D(bin_edges2)
bin2 = lambda x: binner2.bin(ells,x)

for est in ESTS:
    pl = io.Plotter('CL', figsize=(16,12))
    pl2 = io.Plotter('rCL',xyscale='loglin',figsize=(16,12))
    pl3 = io.Plotter('rCL', xyscale='loglin',figsize=(16,12))
    
    # convolve reconstruction w/ normalization
    norm_recon_alms[est] = plensing.phi_to_kappa(hp.almxfl(recon_alms[est][0].astype(np.complex128), Al[est][0]))
    norm_frecon_alms[est] = plensing.phi_to_kappa(hp.almxfl(frecon_alms[est][0].astype(np.complex128), Al[est][0]))

    frcls = hp.alm2cl(norm_frecon_alms[est], norm_frecon_alms[est])
    rcls = hp.alm2cl(norm_recon_alms[est], norm_recon_alms[est])

    rxi_cls = hp.alm2cl(norm_recon_alms[est], ikalm)
    fxi_cls = hp.alm2cl(norm_frecon_alms[est], ikalm)

    pl.add(ells,(ells*(ells+1.)/2.)**2. * Al[est][0],ls='--', label="noise PS (per mode)")
    pl.add(ells,fxi_cls,label='optimal filter recon x input')
    pl.add(ells,rxi_cls,label='isotropic filter recon x input')
    pl.add(ells,icls,label='input x input')

    pl2.add(*bin2((fxi_cls-icls)/icls),marker='o',label="optimal filter recon x input Clkk")
    pl2.add(*bin2((rxi_cls-icls)/icls),marker='o',label="isotropic filter recon x input Clkk")
    pl3.add(*bin2((fxi_cls-rxi_cls)/(rxi_cls-icls)),marker='o',label="optimal filter vs isotropic filter")
    #pl2.add(*bin2((filtered_cls-unfiltered_cls)/unfiltered_cls), marker='o', label="OF ClTT vs IF ClTT")
    pl2._ax.set_xlabel(r'$L$', fontsize=20)
    pl2._ax.legend(fontsize=30)
    
    pl3._ax.set_ylabel(r'Relative difference of $C_L^{\hat{\kappa} \kappa} / C_L^{\kappa \kappa} - 1$', fontsize=16)
    #pl2._ax.legend()
    pl2.hline(y=0)
    pl3.hline(y=0)
    pl2._ax.set_ylim(-0.25,0.25)
    pl3._ax.set_ylim(-0.1,0.1)

    pl.done(f'ps_websky_optimal_filtering_websky_halos_{est}.png')
    pl2.done(f'ps_websky_optimal_filtering_websky_halos_cross_vs_auto_diff_{est}.png')
    pl3.done(f'ps_websky_optimal_filtering_websky_halos_relative_{est}.png')

print("Time elapsed: %0.5f seconds" % (time.time() - t1))
