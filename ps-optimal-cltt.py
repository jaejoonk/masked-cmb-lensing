import numpy as np
import matplotlib.pyplot as plt

from orphics import maps, io, stats
from falafel import qe, utils as futils
from pixell import enmap, curvedsky as cs, lensing as plensing, enplot
import healpy as hp
from healpy.fitsfunc import read_alm

#import websky_lensing_reconstruction as wlrecon

import pytempura

import cmb_ps
import time

ESTS = ['TT']
RESOLUTION = np.deg2rad(0.5/60.)
full_shape, full_wcs = enmap.fullsky_geometry(res=RESOLUTION)

# constants
LMIN = 600
LMAX = 6000
MLMAX = 7000
HOLE_RADIUS = np.deg2rad(6./60.)

BEAM_FWHM = 1.5
NOISE_T = 10.

SEED = 42

# let's convert our lensed alms to a map
PATH_TO_SCRATCH = "/global/cscratch1/sd/jaejoonk/"

ALM_FILENAME = "websky/lensed_alm.fits"
KAP_FILENAME = "websky/kap.fits"

#LESS_FILTERED_MAP_FILENAME = PATH_TO_SCRATCH + "optimal_filtered_websky_random_0.5.fits"
EMPTY_FILTERED_MAP_FILENAME = PATH_TO_SCRATCH + "optimal_filtered_websky_empty_1.0_6000.fits"
FILTERED_MAP_FILENAME = PATH_TO_SCRATCH + "optimal_filtered_websky_random_1.0_6000_nocmbalmzero.fits"
INPAINTED_MAP_FILENAME = PATH_TO_SCRATCH + "inpainted_map_websky_random_fake.fits"
SNR_COORDS_FILENAME = "coords-snr-5-fake-10259.txt"

CLTT_OUTPUT_NAME = f"ps_cltt_optimal_comparison_1.0_6000_nozerocmbalm_{ESTS[0]}.png"
CLTT_RAW_OUTPUT_NAME = f"ps_raw_cltt_optimal_comparison_1.0_6000_nozerocmbalm_{ESTS[0]}.png"

ikalm = futils.change_alm_lmax(hp.map2alm(hp.read_map(KAP_FILENAME)), MLMAX)
icls = hp.alm2cl(ikalm,ikalm)
ells = np.arange(LMAX+1)
ells2 = np.arange(len(icls))

t1 = time.time()

def masked_coords(coords, size=HOLE_RADIUS):
    return enmap.distance_from(full_shape, full_wcs, coords.T, rmax=size) >= size

empty_map = enmap.empty(full_shape, full_wcs, dtype=np.float32)
BEAM_FN = lambda ells: maps.gauss_beam(ells, BEAM_FWHM)
INV_BEAM_FN = lambda ells: 1./maps.gauss_beam(ells, BEAM_FWHM)
white_noise_map = maps.white_noise(shape=full_shape, wcs=full_wcs, noise_muK_arcmin=NOISE_T, seed=SEED)
# define websky theory cls
ucls, tcls = cmb_ps.get_theory_dicts_white_noise_websky(BEAM_FWHM, NOISE_T, grad=False, lmax=MLMAX)
ucls['TT'] = ucls['TT'][:LMAX+1]
tcls['TT'] = tcls['TT'][:LMAX+1]
iso_filt = lambda a: futils.isotropic_filter([a, a*0., a*0.], tcls, LMIN, LMAX)[0]

# binning
bin_edges = np.arange(LMIN,LMAX,40)
binner = stats.bin1D(bin_edges)
bin = lambda x: binner.bin(np.arange(LMAX+1),x)

##################################
# baseline, no holes + isotropic
##################################
empty_alm = futils.change_alm_lmax(read_alm(ALM_FILENAME), lmax=LMAX)
# convolve beam
empty_alm = cs.almxfl(empty_alm, BEAM_FN) 
# add noise
empty_alm = cs.map2alm(cs.alm2map(empty_alm,
                                  enmap.empty(full_shape, full_wcs, dtype=np.float32)) \
                       + white_noise_map, lmax=LMAX)
# isotropic filtering
empty_alm = iso_filt(empty_alm)

############################
# optimal filtering + holes
############################
filtered_map = enmap.read_map(FILTERED_MAP_FILENAME)
optimal_alm = cs.map2alm(filtered_map, lmax=LMAX)
# convolve beam for optimal filtering
optimal_alm = cs.almxfl(optimal_alm, BEAM_FN)

#less_filtered_map = enmap.read_map(LESS_FILTERED_MAP_FILENAME)
#optimal_half_alm = cs.map2alm(less_filtered_map, lmax=LMAX)
#optimal_half_alm = cs.almxfl(optimal_half_alm, BEAM_FN)

###############################
# optimal filtering + no holes
###############################
empty_filtered_map = enmap.read_map(EMPTY_FILTERED_MAP_FILENAME)
empty_optimal_alm = cs.map2alm(empty_filtered_map, lmax=LMAX)
# convolve beam for optimal filtering
empty_optimal_alm = cs.almxfl(empty_optimal_alm, BEAM_FN)

###############################
# zero'd out holes + isotropic
###############################
d = np.loadtxt(SNR_COORDS_FILENAME)
coords = np.column_stack((d[:,0], d[:,1]))
# mask out holes
hole_only_map = cs.alm2map(futils.change_alm_lmax(read_alm(ALM_FILENAME), lmax=LMAX),
                           enmap.empty(full_shape, full_wcs, dtype=np.float32))
hole_only_map[~masked_coords(coords)] = 0.
hole_only_alm = cs.map2alm(hole_only_map, lmax=LMAX)
# convolve beam
hole_only_alm = cs.almxfl(hole_only_alm, BEAM_FN)
# add noise
hole_only_alm = cs.map2alm(cs.alm2map(hole_only_alm,
                                      enmap.empty(full_shape, full_wcs, dtype=np.float32)) \
                           + white_noise_map, lmax=LMAX)
# isotropic filtering
hole_only_alm = iso_filt(hole_only_alm)


#########################
# inpainting + isotropic
#########################
inpainted_map = enmap.read_map(INPAINTED_MAP_FILENAME)
inpainted_alm = cs.map2alm(inpainted_map, lmax=LMAX)
# isotropic filtering
inpainted_alm = iso_filt(inpainted_alm)

###########
# plotting
###########

# theory curve
# theory = ucls['TT'] * BEAM_FN(np.arange(LMAX+1))**2 / (tcls['TT'] ** 2)

# plot cltt
pl_tt = io.Plotter('rCL',xyscale='loglin',figsize=(12,12))
pl_tt2 = io.Plotter('rCL',xyscale='loglog',figsize=(12,12))
empty_cls = hp.alm2cl(empty_alm.astype(np.cdouble),
                      empty_alm.astype(np.cdouble),
                      lmax=LMAX)
optimal_cls = hp.alm2cl(optimal_alm.astype(np.cdouble),
                        optimal_alm.astype(np.cdouble),   
                        lmax=LMAX)
hole_only_cls = hp.alm2cl(hole_only_alm.astype(np.cdouble),
                          hole_only_alm.astype(np.cdouble),
                          lmax=LMAX)
inpainted_cls = hp.alm2cl(inpainted_alm.astype(np.cdouble),
                          inpainted_alm.astype(np.cdouble),
                          lmax=LMAX)
empty_optimal_cls = hp.alm2cl(empty_optimal_alm.astype(np.cdouble),
                              empty_optimal_alm.astype(np.cdouble),   
                              lmax=LMAX)
#optimal_half_cls = hp.alm2cl(optimal_half_alm.astype(np.cdouble),
#                             optimal_half_alm.astype(np.cdouble),
#                             lmax=LMAX)

pl_tt.add(*bin((empty_optimal_cls - empty_cls) / empty_cls), marker='o', label="optimal filtering (no holes) vs no masking")
#pl_tt.add(*bin((optimal_half_cls - empty_cls) / empty_cls), marker='o', label="optimal filtering (half holes) vs no masking")
pl_tt.add(*bin((optimal_cls - empty_cls) / empty_cls), marker='o', label="optimal filtering vs no masking")
pl_tt.add(*bin((hole_only_cls - empty_cls) / empty_cls), marker='o', label="masking + isotropic vs no masking")
pl_tt.add(*bin((inpainted_cls - empty_cls) / empty_cls), marker='o', label="inpaint + isotropic vs no masking")
pl_tt._ax.set_ylabel(r'$(C_L^{TT} - C_L^{T_{\rm no mask} T_{\rm no mask}}) /  C_L^{T_{\rm no mask} T_{\rm no mask}}$', fontsize=24)
pl_tt._ax.set_xlabel(r'$L$', fontsize=30)
pl_tt._ax.legend(fontsize=40)
pl_tt.hline(y=0)
fsky = (10259 * np.pi * (6./60.)**2) / (4 * np.pi * (180./np.pi)**2)
pl_tt.hline(y=fsky)
pl_tt._ax.set_ylim(-0.05,0.1)
pl_tt.done(CLTT_OUTPUT_NAME)

pl_tt2.add(*bin(empty_optimal_cls / ells**2), marker='o', label="optimal filter w/ no holes")
pl_tt2.add(*bin(empty_cls / ells**2), marker='o', label="no masking + isotropic filter")
pl_tt2.add(*bin(optimal_cls / ells**2), marker='o', label="optimal filter")
pl_tt2.add(*bin(hole_only_cls / ells**2), marker='o', label="masking + isotropic filter")
pl_tt2.add(*bin(inpainted_cls / ells**2), marker='o', label="inpainting + isotropic filter")
#pl_tt2.add(np.arange(LMIN,LMAX+1), theory[LMIN:], ls='--', label="expected theory")
pl_tt2._ax.set_xlabel(r'$L$', fontsize=30)
pl_tt2._ax.set_ylabel(r'$L^{-2} \, C_L^{\hat{T} \hat{T}}$', fontsize=24)
pl_tt2._ax.legend(fontsize=40)
pl_tt2.done(CLTT_RAW_OUTPUT_NAME)

# deconvolve beam from each alm
inpainted_alm = cs.almxfl(inpainted_alm, INV_BEAM_FN)
empty_optimal_alm = cs.almxfl(empty_optimal_alm, INV_BEAM_FN)
optimal_alm = cs.almxfl(optimal_alm, INV_BEAM_FN)
empty_alm = cs.almxfl(empty_alm, INV_BEAM_FN)

px = qe.pixelization(shape=full_shape, wcs=full_wcs)

qe_recon = lambda a: qe.qe_all(px, ucls, mlmax=MLMAX, fTalm=a, fEalm=a*0., fBalm=a*0.,
                               estimators=ESTS, xfTalm=None, xfEalm=None, xfBalm=None)

inpainted_alm_recon = qe_recon(inpainted_alm)
empty_optimal_alm_recon = qe_recon(empty_optimal_alm)
optimal_alm_recon = qe_recon(optimal_alm)
empty_alm_recon = qe_recon(empty_alm)

# normalize using tempura
Al = pytempura.get_norms(ESTS,ucls,tcls,LMIN,LMAX,k_ellmax=MLMAX,no_corr=False)

# plot cross spectra vs auto spectra
inpainted_alm_recon_norm = {}
empty_optimal_alm_recon_norm = {}
optimal_alm_recon_norm = {}
empty_alm_recon_norm = {}

# use this if first bin object is doing linear bins
bin_edges2 = np.geomspace(2,LMAX,40)
binner2 = stats.bin1D(bin_edges2)
bin2 = lambda x: binner2.bin(ells2,x)

for est in ESTS:
    normalize = lambda ralms: plensing.phi_to_kappa(hp.almxfl(ralms[est][0].astype(np.complex128),
                                                              Al[est][0]))

    pl = io.Plotter('CL', figsize=(16,12))
    pl2 = io.Plotter('rCL',xyscale='loglin',figsize=(16,12))
    #pl3 = io.Plotter('rCL', xyscale='loglin',figsize=(16,12))
    
    # convolve reconstruction w/ normalization
    inpainted_alm_recon_norm[est] = normalize(inpainted_alm_recon)
    empty_optimal_alm_recon_norm[est] = normalize(empty_optimal_alm_recon)
    optimal_alm_recon_norm[est] = normalize(optimal_alm_recon)
    empty_alm_recon_norm[est] = normalize(empty_alm_recon)

    inp_a_cls = hp.alm2cl(inpainted_alm_recon_norm[est], inpainted_alm_recon_norm[est])
    opt_empt_a_cls = hp.alm2cl(empty_optimal_alm_recon_norm[est], empty_optimal_alm_recon_norm[est])
    opt_a_cls = hp.alm2cl(optimal_alm_recon_norm[est], optimal_alm_recon_norm[est])
    rcls = hp.alm2cl(empty_alm_recon_norm[est], empty_alm_recon_norm[est])

    rxi_cls = hp.alm2cl(empty_alm_recon_norm[est], ikalm)
    inpxi_cls = hp.alm2cl(inpainted_alm_recon_norm[est], ikalm)
    optemptxi_cls = hp.alm2cl(empty_optimal_alm_recon_norm[est], ikalm)
    optxi_cls = hp.alm2cl(optimal_alm_recon_norm[est], ikalm)

    pl.add(ells2,(ells2*(ells2+1.)/2.)**2. * Al[est][0],ls='--', label="noise PS (per mode)")
    pl.add(ells2,inpxi_cls,label='inpainting + isotropic recon x input')
    pl.add(ells2,rxi_cls,label='baseline recon x input')
    pl.add(ells2,optxi_cls,label='optimal recon x input')
    pl.add(ells2,optemptxi_cls,label='optimal no holes recon x input')

    pl2.add(*bin2((inpxi_cls-icls)/icls),marker='o',label="inpainting + isotropic recon x input Clkk")
    pl2.add(*bin2((rxi_cls-icls)/icls),marker='o',label="baseline recon x input Clkk")
    pl2.add(*bin2((optxi_cls-icls)/icls),marker='o',label="optimal recon x input Clkk")
    pl2.add(*bin2((optemptxi_cls-icls)/icls),marker='o',label="optimal no holes recon x input Clkk")
    ##pl2.add(*bin2((filtered_cls-unfiltered_cls)/unfiltered_cls), marker='o', label="OF ClTT vs IF ClTT")
    pl2._ax.set_xlabel(r'$L$', fontsize=20)
    pl2._ax.legend(fontsize=30)
    
   #pl3._ax.set_ylabel(r'Relative difference of $C_L^{\hat{\kappa} \kappa} / C_L^{\kappa \kappa} - 1$', fontsize=16)
    #pl2._ax.legend()
    pl2.hline(y=0)
    #pl3.hline(y=0)
    pl2._ax.set_ylim(-0.25,0.25)
   #pl3._ax.set_ylim(-0.1,0.1)

    pl.done(f'ps_websky_optimal_filtering_websky_{est}.png')
    pl2.done(f'ps_websky_optimal_filtering_websky_cross_vs_auto_diff_{est}.png')
    #pl3.done(f'ps_websky_optimal_filtering_websky_halos_relative_{est}.png')

print("Time elapsed: %0.5f seconds" % (time.time() - t1))
