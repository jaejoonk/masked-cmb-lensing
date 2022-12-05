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

import time, string, os

ESTS = ['TT']
RESOLUTION = np.deg2rad(0.5/60.)
COMM = MPI.COMM_WORLD
rank = COMM.Get_rank() 

# let's convert our lensed alms to a map
PATH_TO_SCRATCH = "/global/cscratch1/sd/jaejoonk/"
#ALM_FILENAME = "websky/lensed_alm.fits"

# ALM_FILENAME = PATH_TO_SCRATCH + "maps/lensed_cmb_alm_websky_cmb1999_lmax6000.fits"
ALM_FILENAME = "websky/lensed_alm.fits"

MAP_FILENAME = PATH_TO_SCRATCH + "maps/cmb_night_pa5_f150_8way_coadd_map.fits"
IVAR_FILENAME = PATH_TO_SCRATCH + "maps/cmb_night_pa5_f150_8way_coadd_ivar.fits"
MASK_FILENAME = PATH_TO_SCRATCH + "act_mask_20220316_GAL060_rms_70.00_d2sk.fits"

COORDS_FILENAME = "output_halos_10259.txt"
COORDS_OUTPUT = "coords-10259.txt"
COORDS = 10259

SNR_COORDS_FILENAME = "coords-snr-2-mask-fake-10259.txt"

MIN_MASS = 1.0 # 1e14
MAX_MASS = 100.0 # 1e14

HOLE_RADIUS = np.deg2rad(6.0/60.)
NOISE_T = 10. # muK arcmin

FULL_SHAPE, FULL_WCS = enmap.fullsky_geometry(res=RESOLUTION)
IVAR = maps.ivar(shape=FULL_SHAPE, wcs=FULL_WCS, noise_muK_arcmin=NOISE_T)
# IVAR = enmap.downgrade(enmap.read_map(IVAR_FILENAME), 2, op=np.sum)

MASK = enmap.read_map(MASK_FILENAME)
OUTPUT_DIR = "/global/cscratch1/sd/jaejoonk/inpaint_geos/"
BEAM_FWHM = 1.5 # arcmin
BEAM_SIG = BEAM_FWHM / (8 * np.log(2))**0.5 
LMAX = 3000
MLMAX = 6000
THEORY_FN = cosmology.default_theory().lCl
## probably wrong, but gaussian centered at l = 0 and sigma derived from beam fwhm   
BEAM_FN = lambda ells: maps.gauss_beam(ells, BEAM_FWHM)
SNR = 5

CONTEXT_FRACTION = 2./3.

UNINPAINTED_MAP_NAME = f"uninpainted_map_websky_fake3.fits"
INPAINTED_MAP_NAME = f"inpainted_map_websky_fake3.fits"

# random 8 letter name
FOLDER_SIZE = 8 
if rank == 0:
    FOLDER_ARRAY = np.random.choice(np.array(list(string.ascii_lowercase)), size=FOLDER_SIZE)
    FOLDER_DICT = {}
    for i in range(len(FOLDER_ARRAY)): FOLDER_DICT[i] = FOLDER_ARRAY[i]
else:
    FOLDER_DICT = None

FOLDER_DICT = COMM.bcast(FOLDER_DICT, root=0)
FOLDER_NAME = "".join([FOLDER_DICT[i] for i in range(len(FOLDER_DICT.keys()))]) + "/"

def mass_gen_coords(min_mass=MIN_MASS, max_mass=MAX_MASS, ncoords=COORDS,
                    coords_filename=COORDS_FILENAME, coords_output=COORDS_OUTPUT):
    ra, dec = josh_wstack.read_coords_from_file(coords_filename,
                lowlim=min_mass, highlim=max_mass)
    try:
        indices = np.random.choice(len(ra), ncoords, replace=False)
        coords = np.array([[dec[i], ra[i]] for i in indices])
    except ValueError: # too many ncoords in the catalog
        coords = np.array([[dec[i], ra[i]] for i in range(len(ra))]) 
    np.savetxt(coords_output, coords) 
    
    if rank == 0:
        print(f"Generated coordinates between {min_mass}e14 ~ {max_mass}e14, wrote to {coords_output}.")

    return coords
    
def save_geometries(coords, hole_rad=HOLE_RADIUS, ivar=IVAR, output_dir=OUTPUT_DIR,
                    theory_fn=THEORY_FN, beam_fn=BEAM_FN, comm=COMM):
    if rank == 0:
        print(f"Saving geometries now...")
        os.mkdir(output_dir + FOLDER_NAME)
        print(f"Created directory {FOLDER_NAME}.")

    comm.Barrier()
    t1 = time.time()
    pixcov.inpaint_uncorrelated_save_geometries(coords, hole_rad, ivar, output_dir + FOLDER_NAME,
                                                theory_fn=theory_fn, beam_fn=beam_fn,
                                                pol=False, context_fraction=CONTEXT_FRACTION, comm=comm)

    t2 = time.time() 
    if rank == 0: print(f"Done saving geometries after {t2-t1:0.5f} seconds!")

def inpainting(output_dir=OUTPUT_DIR, map_filename=MAP_FILENAME,
               alm_filename=ALM_FILENAME, mask_filename=MASK_FILENAME,
               res=RESOLUTION, mlmax=MLMAX, beam_fwhm=BEAM_FWHM,
               ifsnr=True, snr=5, min_mass=MIN_MASS, max_mass=MAX_MASS):
    ## reconvolve beam?
    if rank == 0:
        print(f"Reading geometries from {output_dir}.")
        # only for sehgal
        if alm_filename is not None:
            lensed_map = cs.alm2map(hp.sphtfunc.map2alm(hp.read_map(alm_filename)),
                                    enmap.empty(FULL_SHAPE, FULL_WCS, dtype=np.float32))
            lensed_map = josh_wlrecon.almfile_to_map(alm_filename=alm_filename, res=res)
        else:
            lensed_map = enmap.read_map(map_filename)[0]
            #io.hplot(lensed_map, PATH_TO_SCRATCH + "pre_inpainted_pre_mask_data_map", downgrade=2)
        
        lensed_map = josh_wlrecon.apply_mask(lensed_map, MASK, lmax=mlmax)
        #io.hplot(lensed_map, PATH_TO_SCRATCH + "pre_inpainted_post_mask_data_map")
        lensed_alms_plus_beam = cs.almxfl(cs.map2alm(lensed_map, lmax=mlmax, tweak=True), BEAM_FN)
        lensed_map = cs.alm2map(lensed_alms_plus_beam,
                                enmap.empty(lensed_map.shape, lensed_map.wcs, dtype=np.float32)) 
        # add a noise profile
        lensed_map += maps.white_noise(shape=lensed_map.shape,
                                       wcs=lensed_map.wcs,
                                       noise_muK_arcmin=NOISE_T)
        #io.hplot(lensed_map, PATH_TO_SCRATCH + "pre_inpainted_post_mask_noisy_data_map")
        enmap.write_map(UNINPAINTED_MAP_NAME, lensed_map, fmt="fits")

        print("Saved uninpainted map to disk.")
    #io.hplot(lensed_map, "pre_inpainted_map_view.png")
    else:
        lensed_map = enmap.zeros(MASK.shape, wcs=MASK.wcs, dtype=np.float32)
    
    COMM.Barrier()
    COMM.Bcast(lensed_map, root=0)

    t1 = time.time()
    inpainted_map = pixcov.inpaint_uncorrelated_from_saved_geometries(lensed_map, output_dir,
                                                                      norand=False, zeroout=False)
    t2 = time.time()

    COMM.Barrier()
    # don't need parallel processes anymore?
    if rank != 0: exit()

    enmap.write_map(INPAINTED_MAP_NAME, inpainted_map, fmt="fits")
    print(f"Time for inpainting from geometries: {t2-t1:0.5f} seconds")

    ## deconvolve beam
    #INV_BEAM_FN = lambda ells: 1./maps.gauss_beam(ells, beam_fwhm)
    #inpainted_alm = cs.almxfl(cs.map2alm(inpainted_map, lmax=lmax), INV_BEAM_FN)
    #inpainted_map = cs.alm2map(inpainted_alm, enmap.empty(shape, wcs, dtype=np.float32)) 

    # SAVE MAP + alms
    #if snr: enmap.write_map(f"inpainted_map_ivar_SNR_{snr}.fits", inpainted_map, fmt="fits")
    #else: enmap.write_map(f"inpainted_map_ivar_{MIN_MASS:.1f}_to_{MAX_MASS:.1f}.fits", inpainted_map, fmt="fits")
    #print(f"Saved map.")

def inpainting_from_alms(output_dir=OUTPUT_DIR, alm_filename=ALM_FILENAME,
               res=RESOLUTION, mlmax=MLMAX, beam_fwhm=BEAM_FWHM,
               ifsnr=True, snr=5, min_mass=MIN_MASS, max_mass=MAX_MASS):
    ## reconvolve beam?
    if rank == 0:
        print(f"Reading geometries from {output_dir}.")
        print("Inpainting from alms.")
  
        lensed_map = josh_wlrecon.almfile_to_map(alm_filename=alm_filename, res=res)
        # convolve beam
        lensed_alms_plus_beam = cs.almxfl(cs.map2alm(lensed_map, lmax=mlmax, tweak=True), BEAM_FN)
        lensed_map = cs.alm2map(lensed_alms_plus_beam,
                                enmap.empty(FULL_SHAPE, FULL_WCS, dtype=np.float32)) 
        # add a noise profile
        lensed_map += maps.white_noise(shape=FULL_SHAPE,
                                       wcs=FULL_WCS,
                                       noise_muK_arcmin=NOISE_T)
        enmap.write_map(UNINPAINTED_MAP_NAME, lensed_map, fmt="fits")

        print("Saved uninpainted map to disk.")
    else:
        lensed_map = enmap.zeros(FULL_SHAPE, wcs=FULL_WCS, dtype=np.float32)
    
    COMM.Barrier()
    COMM.Bcast(lensed_map, root=0)

    t1 = time.time()
    inpainted_map = pixcov.inpaint_uncorrelated_from_saved_geometries(lensed_map, output_dir,
                                                                      norand=False, zeroout=False)
    t2 = time.time()

    COMM.Barrier()
    # don't need parallel processes anymore?
    if rank != 0: exit()

    enmap.write_map(INPAINTED_MAP_NAME, inpainted_map, fmt="fits")
    print(f"Time for inpainting from geometries: {t2-t1:0.5f} seconds")

def do_all(saved=False, ifsnr = True, folder_name=None, from_alms=True):
    if not ifsnr: c = mass_gen_coords()
    else:
        d = np.loadtxt(SNR_COORDS_FILENAME)
        c = np.column_stack((d[:,0], d[:,1]))

    if not saved: save_geometries(c)
    COMM.Barrier()
    
    if from_alms: inpainting_from_alms(
                      output_dir=(OUTPUT_DIR + \
                          (FOLDER_NAME if folder_name is None else folder_name)
                      ))
    else: inpainting(
              output_dir=(OUTPUT_DIR + \
                  (FOLDER_NAME if folder_name is None else folder_name)
              ))

if __name__ == '__main__':
    do_all(saved=True, folder_name="rcgctrya/", from_alms=True)
