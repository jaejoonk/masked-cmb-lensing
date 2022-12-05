import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from orphics import maps, cosmology, io, pixcov, mpi, stats
from falafel import qe, utils as futils
from pixell import enmap, reproject, curvedsky as cs, lensing as plensing
import healpy as hp
import pytempura
from optweight import filters

import websky_lensing_reconstruction as wlrecon

import time, string, os

ESTS = ['TT']
RESOLUTION = np.deg2rad(0.5/60.)
#COMM = MPI.COMM_WORLD
#rank = COMM.Get_rank() 

# let's convert our lensed alms to a map
PATH_TO_SCRATCH = "/global/cscratch1/sd/jaejoonk/"

ALM_FILENAME = "websky/lensed_alm.fits"

#MAP_FILENAME = PATH_TO_SCRATCH + "maps/cmb_night_pa5_f150_8way_coadd_map.fits"
#IVAR_FILENAME = PATH_TO_SCRATCH + "maps/cmb_night_pa5_f150_8way_coadd_ivar.fits"
#MASK_FILENAME = PATH_TO_SCRATCH + "act_mask_20220316_GAL060_rms_70.00_d2sk.fits"

COORDS_FILENAME = "output_halos_10259.txt"
COORDS_OUTPUT = "coords-10259.txt"
COORDS = 10259

SNR_COORDS_FILENAME = "coords-snr-5.txt"

MIN_MASS = 1.0 # 1e14
MAX_MASS = 100.0 # 1e14

HOLE_RADIUS = np.deg2rad(6.0/60.)
NOISE_T = 10. # muK arcmin

LMAX = 3000
MLMAX = 6000  

FULL_SHAPE, FULL_WCS = enmap.fullsky_geometry(res=RESOLUTION)
IVAR = maps.ivar(shape=FULL_SHAPE, wcs=FULL_WCS, noise_muK_arcmin=NOISE_T)
# IVAR = enmap.downgrade(enmap.read_map(IVAR_FILENAME), 2, op=np.sum)

OUTPUT_DIR = "/global/cscratch1/sd/jaejoonk/inpaint_geos/"
BEAM_FWHM = 1.5 # arcmin
BEAM_SIG = BEAM_FWHM / (8 * np.log(2))**0.5 
BEAM_FN = lambda ells: maps.gauss_beam(ells, BEAM_FWHM)
SNR = 5

CONTEXT_FRACTION = 2./3.

FILTERED_MAP_NAME = f"optimal-filtered-websky-map.fits"

# projects onto full sky geometry to agree with ivar map
def masked_coords(coords, size=HOLE_RADIUS):
    return enmap.distance_from(FULL_SHAPE, FULL_WCS, coords.T, rmax=size) >= size

def optimal_filter(coords, alm_filename=ALM_FILENAME,
                   output_filename=FILTERED_MAP_NAME,
                   res=RESOLUTION, lmax=LMAX,
                   mlmax=MLMAX, beam_fwhm=BEAM_FWHM):

    start = time.time()
    print(f"Setting up stuff...")
    
    alms = hp.fitsfunc.read_alm(alm_filename, hdu=(1,2,3))[0]
    alms = cs.almxfl(alms, BEAM_FN)
    # stamp out holes
    imap = cs.alm2map(alms, enmap.empty(shape=FULL_SHAPE, wcs=FULL_WCS))
    mask_bool = masked_coords(coords)
    alms = cs.map2alm(imap * mask_bool, lmax=mlmax)
    
    theory_cls, _ = futils.get_theory_dicts(lmax=mlmax)
    ells = np.arange(len(theory_cls['TT']))
    b_ells = BEAM_FN(ells)
    
    print(f"Done (time elapsed: {time.time() - start:0.5f} seconds)")

    print(f"Running optimal filtering...")
    alm_dict = filters.cg_pix_filter(alms, theory_cls, b_ells, lmax,
                                     icov_pix=IVAR, mask_bool=mask_bool,
                                     cov_noise_ell=None, verbose=True)

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