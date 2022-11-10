import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from orphics import maps, cosmology, io, pixcov, mpi, stats
from falafel import qe, utils as futils
from pixell import enmap, curvedsky as cs, lensing as plensing
import healpy as hp
import pytempura
from mpi4py import MPI

import websky_lensing_reconstruction as josh_wlrecon
import websky_stack_and_visualize as josh_wstack
import cmb_ps

import time

ESTS = ['TT']
RESOLUTION = np.deg2rad(0.5/60.)
COMM = MPI.COMM_WORLD
rank = COMM.Get_rank() 

# let's convert our lensed alms to a map
KAP_FILENAME = "websky/kap.fits"
ALM_FILENAME = "websky/lensed_alm.fits"
COORDS_FILENAME = "output_halos.txt"
COORDS_OUTPUT = "random-coords.txt"
COORDS = 12000
#SNR_COORDS_FILENAME = "fake-snr-5-coords-dec-limited.txt"
SNR_COORDS_FILENAME = "coords-snr-5.txt"

shape, wcs = enmap.fullsky_geometry(res=RESOLUTION)
omap = enmap.empty(shape, wcs, dtype=np.float32)

MIN_MASS = 3.0 # 1e14
MAX_MASS = 20.0 # 1e14

HOLE_RADIUS = np.deg2rad(6.0/60.)
NOISE_T = 10. # muK arcmin
IVAR = maps.ivar(shape=shape, wcs=wcs, noise_muK_arcmin=NOISE_T)
OUTPUT_DIR = "/global/cscratch1/sd/jaejoonk/inpaint_geos/"
BEAM_FWHM = 1.5 # arcmin
BEAM_SIG = BEAM_FWHM / (8 * np.log(2))**0.5 
LMAX = 3000
THEORY_FN = cosmology.default_theory().lCl
## probably wrong, but gaussian centered at l = 0 and sigma derived from beam fwhm   
BEAM_FN = lambda ells: maps.gauss_beam(ells, BEAM_FWHM)
SNR = 5

def mass_gen_coords(min_mass=MIN_MASS, max_mass=MAX_MASS, ncoords=COORDS,
                    coords_filename=COORDS_FILENAME, coords_output=COORDS_OUTPUT):
    if rank == 0:
        ra, dec = josh_wstack.read_coords_from_file(coords_filename,
                  lowlim=min_mass, highlim=max_mass)
        try:
            indices = np.random.choice(len(ra), ncoords, replace=False)
        except ValueError: # too many ncoords in the catalog
            indices = len(ra)
        coords = np.array([[dec[i], ra[i]] for i in range(COORDS)]) 

        np.savetxt(coords_output, coords) 
        return coords

    else:
        coords = None
    
    print(f"Generated coordinates between {min_mass}e14 ~ {max_mass}e14, wrote to {coords_output}.")

def save_geometries(coords, hole_rad=HOLE_RADIUS, ivar=IVAR, output_dir=OUTPUT_DIR,
                    theory_fn=THEORY_FN, beam_fn=BEAM_FN, comm=COMM):

    t1 = time.time()
    pixcov.inpaint_uncorrelated_save_geometries(coords, hole_rad, ivar, output_dir,
                                                theory_fn=theory_fn, beam_fn=beam_fn,
                                                pol=False, comm=comm)

    t2 = time.time() 
    if rank == 0: print(f"Done saving geometries after {t2-t1:0.5f} seconds!")

def inpainting(output_dir=OUTPUT_DIR, alm_filename=ALM_FILENAME, res=RESOLUTION, lmax=LMAX,
               beam_fwhm=BEAM_FWHM, ifsnr=True, snr=5, min_mass=MIN_MASS, max_mass=MAX_MASS):
    ## reconvolve beam?
    lensed_map = josh_wlrecon.almfile_to_map(alm_filename=alm_filename, res=res)
    lensed_alms_plus_beam = cs.almxfl(cs.map2alm(lensed_map, lmax=lmax), BEAM_FN)
    lensed_map = cs.alm2map(lensed_alms_plus_beam, enmap.empty(shape, wcs, dtype=np.float32)) 
    # add a noise profile
    lensed_map += maps.white_noise(shape=shape, wcs=wcs, noise_muK_arcmin=NOISE_T)
    #io.hplot(lensed_map, "pre_inpainted_map_view.png")

    t1 = time.time()
    inpainted_map = pixcov.inpaint_uncorrelated_from_saved_geometries(lensed_map, output_dir)
    t2 = time.time()

    COMM.Barrier()
    # don't need parallel processes anymore?
    if rank != 0: exit()

    print(f"Time for inpainting from geometries: {t2-t1:0.5f} seconds")

    ## deconvolve beam
    INV_BEAM_FN = lambda ells: 1./maps.gauss_beam(ells, beam_fwhm)
    inpainted_alm = cs.almxfl(cs.map2alm(inpainted_map, lmax=lmax), INV_BEAM_FN)
    inpainted_map = cs.alm2map(inpainted_alm, enmap.empty(shape, wcs, dtype=np.float32)) 

    # SAVE MAP + alms
    if snr: enmap.write_map(f"inpainted_map_ivar_SNR_{snr}.fits", inpainted_map, fmt="fits")
    else: enmap.write_map(f"inpainted_map_ivar_{MIN_MASS:.1f}_to_{MAX_MASS:.1f}.fits", inpainted_map, fmt="fits")
    print(f"Saved map.")


def do_all(ifsnr = True):
    if not ifsnr: c = mass_gen_coords()
    else:
        d = np.loadtxt(SNR_COORDS_FILENAME)
        c = np.column_stack((d[:,0], d[:,1]))

    #save_geometries(c)
    #COMM.Barrier()
    inpainting()

if __name__ == '__main__':
    do_all()
