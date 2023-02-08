import numpy as np
import matplotlib.pyplot as plt

from orphics import maps, cosmology, io, pixcov, mpi, stats
from falafel import qe, utils as futils
from pixell import enmap, reproject, curvedsky as cs, lensing as plensing
import healpy as hp
import pytempura
from optweight import filters
import cmb_ps

import websky_lensing_reconstruction as wlrecon

import time, string, os

ESTS = ['TT']
RESOLUTION = np.deg2rad(1.0/60.)

# let's convert our lensed alms to a map
PATH_TO_SCRATCH = "/home3/jaejoonk/sims/"

ALM_FILENAME = "websky/lensed_alm.fits"
#MAP_FILENAME = PATH_TO_SCRATCH + "maps/cmb_night_pa5_f150_8way_coadd_map.fits"
#IVAR_FILENAME = PATH_TO_SCRATCH + "maps/cmb_night_pa5_f150_8way_coadd_ivar.fits"
#MASK_FILENAME = PATH_TO_SCRATCH + "act_mask_20220316_GAL060_rms_70.00_d2sk.fits"

SNR_COORDS_FILENAME = "coords-snr-5-fake-10259.txt"

HOLE_RADIUS = np.deg2rad(6.0/60.)
NOISE_T = 10. # muK arcmin

LMAX = 10000
MLMAX = 10500

SEED = 37

FULL_SHAPE, FULL_WCS = enmap.fullsky_geometry(res=RESOLUTION)
IVAR = maps.ivar(shape=FULL_SHAPE, wcs=FULL_WCS, noise_muK_arcmin=NOISE_T)
# IVAR = enmap.downgrade(enmap.read_map(IVAR_FILENAME), 2, op=np.sum)

BEAM_FWHM = 1.5 # arcmin
BEAM_FN = lambda ells: maps.gauss_beam(ells, BEAM_FWHM)
SNR = 5

NITER = 100

FILTERED_MAP_NAME = PATH_TO_SCRATCH + f"optimal_filtered_10000_cmbalmzero.fits"

# projects onto full sky geometry to agree with ivar map
def masked_coords(coords, size=HOLE_RADIUS):
    return enmap.distance_from(FULL_SHAPE, FULL_WCS, coords.T, rmax=size) >= size


def optimal_filter(coords, alm_filename=ALM_FILENAME,
                   output_filename=FILTERED_MAP_NAME,
                   lmax=LMAX, beam_fwhm=BEAM_FWHM):

    start = time.time()
    print(f"Setting up stuff and writing eventually to {output_filename}...")

    # create into sigurd style map d = a_lm * b_l + n_l
    alms = hp.fitsfunc.read_alm(alm_filename, hdu=(1,2,3))[0]
    alms = futils.change_alm_lmax(cs.almxfl(alms, BEAM_FN), lmax)
    alms = cs.map2alm(cs.alm2map(alms, enmap.empty(FULL_SHAPE, FULL_WCS, dtype=np.float32)) \
                      + maps.white_noise(shape=FULL_SHAPE, wcs=FULL_WCS, noise_muK_arcmin=NOISE_T, seed=SEED),
                      lmax=lmax)
    # stamp out holes
    mask_bool = masked_coords(coords)
    # thanks Mat!
    IVAR[~mask_bool] = 0.
    # set cmb alms to zero at coordinates too
    cmb_map = cs.alm2map(alms, enmap.empty(FULL_SHAPE, FULL_WCS, dtype=np.float32))
    cmb_map[~mask_bool] = 0.
    alms = cs.map2alm(cmb_map, lmax=lmax).astype(np.complex128)
    
    # use this line if testing on websky sims
    theory_cls, _ = cmb_ps.get_theory_dicts_white_noise_websky(BEAM_FWHM, NOISE_T, grad=False, lmax=MLMAX)
    #theory_cls, _ = futils.get_theory_dicts(lmax=lmax, grad=False)
    theory_cls['TT'] = theory_cls['TT'][:lmax+1]
    b_ells = BEAM_FN(np.arange(lmax+1))

    print(f"Done (time elapsed: {time.time() - start:0.5f} seconds)")

    print(f"Running optimal filtering...")
    alm_dict = filters.cg_pix_filter(alms, theory_cls, b_ells, lmax,
                                     icov_pix=IVAR, cov_noise_ell=None,
                                     niter=NITER, verbose=True)

    print(f"Done (time elapsed: {time.time() - start:0.5f} seconds)")

    print(f"Writing output map to {output_filename}:")
    omap = cs.alm2map(alm_dict['ialm'], enmap.empty(shape=FULL_SHAPE, wcs=FULL_WCS))
    enmap.write_map(output_filename, omap)
    print(f"Done (time elapsed: {time.time() - start:0.5f} seconds)")

def do_all(saved=False, folder_name=None, from_alms=True):
    d = np.loadtxt(SNR_COORDS_FILENAME)
    c = np.column_stack(([d[:,0], d[:,1]]))
    
    optimal_filter(c)

if __name__ == '__main__':
    do_all()
