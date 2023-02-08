# output a bunch of thumbnail pics on some of the coordinates
import websky_lensing_reconstruction as josh_wlrecon
import numpy as np
from pixell import enmap, curvedsky as cs
import healpy as hp
from orphics import maps, io
import cmb_ps
from falafel import utils as futils

COORDS_FILENAME = "coords-snr-5-fake-10259.txt"
ALM_FILENAME = "websky/lensed_alm.fits"
OPTIMAL_FILENAME = "../sims/" + "optimal_filtered_6000_cmbalmzero.fits"

OUTPUT_NAME = "optimal_vs_unfiltered_cmbalmzero_6000"

NUM_COORDS = 10
RESOLUTION = np.deg2rad(1.0/60.) # 0.5 arcmin by default
RADIUS = 30 * RESOLUTION

LMAX = 6000
MLMAX = 7000

FULL_SHAPE, FULL_WCS = enmap.fullsky_geometry(res=RESOLUTION)
NOISE_T = 10.
BEAM_FWHM = 1.5
SEED = 42
noise_map = maps.white_noise(shape=FULL_SHAPE, wcs=FULL_WCS, noise_muK_arcmin=NOISE_T, seed=SEED)

# plot the thumbnails before and after inpainting
coords = np.loadtxt(COORDS_FILENAME)
sampled = coords[np.random.choice(len(coords), NUM_COORDS, replace=False)]

# load 1 / (C_l + N_l/b^2) * d / b_l
optimal_map = enmap.read_map(OPTIMAL_FILENAME)
# multiply by beam to get d / (C_l + N_l/b^2) = (b_l^2 C_l + N_l) / tcls
optimal_alms = cs.almxfl(cs.map2alm(optimal_map, lmax=LMAX),
                                    lambda ells: maps.gauss_beam(ells, BEAM_FWHM))
_, tcls = cmb_ps.get_theory_dicts_white_noise_websky(BEAM_FWHM, NOISE_T, grad=True, lmax=LMAX)

# multiply by tcls to get b_l^2 C_l + N_l
optimal_map = cs.alm2map(cs.almxfl(optimal_alms, tcls['TT']),
                         enmap.empty(shape=FULL_SHAPE, wcs=FULL_WCS))

# load d = b_l a_lm
unfiltered_alms = cs.almxfl(hp.read_alm(ALM_FILENAME), lambda ells: maps.gauss_beam(ells, BEAM_FWHM))
# square and add noise to get b_l^2 C_l + N_l
unfiltered_map = cs.alm2map(unfiltered_alms, enmap.empty(shape=FULL_SHAPE, wcs=FULL_WCS)) + noise_map

# change to lmax
unfiltered_map = cs.alm2map(cs.map2alm(unfiltered_map, lmax=LMAX), enmap.empty(shape=FULL_SHAPE, wcs=FULL_WCS))

josh_wlrecon.optimal_vs_iso_map(unfiltered_map, optimal_map,
                                sampled, title=OUTPUT_NAME,
                                radius=RADIUS, res=RESOLUTION)

