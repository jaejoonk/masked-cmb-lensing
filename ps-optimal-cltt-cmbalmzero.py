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
RESOLUTION = np.deg2rad(1.0/60.)
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
FILTERED_MAP_ZERO_FILENAME = PATH_TO_SCRATCH + "optimal_filtered_websky_random_1.0_6000_cmbalmzero.fits"
FILTERED_MAP_NO_ZERO_FILENAME = PATH_TO_SCRATCH + "optimal_filtered_websky_random_1.0_6000_nocmbalmzero.fits"
INPAINTED_MAP_FILENAME = PATH_TO_SCRATCH + "inpainted_map_websky_random_fake.fits"
SNR_COORDS_FILENAME = "coords-snr-5-fake-10259.txt"

CLTT_OUTPUT_NAME = f"ps_cltt_optimal_comparison_1.0_6000_zero_comparison_{ESTS[0]}.png"
CLTT_RAW_OUTPUT_NAME = f"ps_raw_cltt_optimal_comparison_1.0_6000_zero_comparison_{ESTS[0]}.png"

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
filtered_map = enmap.read_map(FILTERED_MAP_NO_ZERO_FILENAME)
optimal_alm = cs.map2alm(filtered_map, lmax=LMAX)
# convolve beam for optimal filtering
optimal_alm = cs.almxfl(optimal_alm, BEAM_FN)

filtered_zero_map = enmap.read_map(FILTERED_MAP_ZERO_FILENAME)
optimal_zero_alm = cs.map2alm(filtered_zero_map, lmax=LMAX)
# convolve beam for optimal filtering
optimal_zero_alm = cs.almxfl(optimal_zero_alm, BEAM_FN)

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
optimal_zero_cls = hp.alm2cl(optimal_zero_alm.astype(np.cdouble),
                             optimal_zero_alm.astype(np.cdouble),   
                             lmax=LMAX)
hole_only_cls = hp.alm2cl(hole_only_alm.astype(np.cdouble),
                          hole_only_alm.astype(np.cdouble),
                          lmax=LMAX)
inpainted_cls = hp.alm2cl(inpainted_alm.astype(np.cdouble),
                          inpainted_alm.astype(np.cdouble),
                          lmax=LMAX)

pl_tt.add(*bin((optimal_zero_cls - empty_cls) / empty_cls), marker='o', label="optimal filtering (zero'd CMB) vs no masking")
pl_tt.add(*bin((optimal_cls - empty_cls) / empty_cls), marker='o', label="optimal filtering (nonzero'd CMB) vs no masking")
pl_tt.add(*bin((hole_only_cls - empty_cls) / empty_cls), marker='o', label="masking (zero'd kappa) vs no masking")
pl_tt.add(*bin((inpainted_cls - empty_cls) / empty_cls), marker='o', label="inpainting vs no masking")
pl_tt._ax.set_ylabel(r'$(C_L^{TT} - C_L^{T_{\rm{no mask}} T_{\rm{no mask}}}) /  C_L^{T_{\rm{no mask}} T_{\rm{no mask}}}$', fontsize=24)
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
