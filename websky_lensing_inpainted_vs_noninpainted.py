###############################################
# imports
###############################################
import time, os

from pixell import enmap,utils as putils,reproject,enplot,curvedsky as cs
from pixell.curvedsky import map2alm

import numpy as np

import healpy as hp
from healpy.fitsfunc import read_alm,read_map

import symlens as s
from symlens import utils as sutils

from falafel import qe, utils
import pytempura 

from mpi4py import MPI

# my own files
import websky_lensing_reconstruction as wlrecon
import websky_stack_and_visualize as websky_stack

import argparse
###############################################
# constants
###############################################
DEBUG = True

PATH_TO_SCRATCH = "/global/cscratch1/sd/jaejoonk/"
PATH_TO_FALAFEL = "/home/joshua/research/falafel"
MASK_FILENAME = PATH_TO_SCRATCH + "act_mask_20220316_GAL060_rms_70.00_d2sk.fits"
KAP_FILENAME = "websky/kap.fits"
#KAP_FILENAME = PATH_TO_SCRATCH + "maps/kappa_alm_lmax6000.fits"
ALM_FILENAME = "websky/lensed_alm.fits"
MAP_FILENAME = "inpainted_fake_map_6000.fits"
MAP2_FILENAME = "uninpainted_fake_map_6000.fits"
HALOS_FILENAME = PATH_TO_SCRATCH + "halos.pksc"
COORDS_FILENAME = "output_halos.txt"
NCOORDS = 10000
OTHER_COORDS = None
NBINS = 20
LWIDTH = 50

RESOLUTION = np.deg2rad(0.5 / 60.)
STACK_RES = np.deg2rad(0.5 / 60.)
RADIUS = STACK_RES * 20. # 10 arcmin
SYM_RES = np.deg2rad(0.5 / 60.)
SYM_SHAPE = (2000,2000)
RAD = np.deg2rad(0.5)
OMEGAM_H2 = 0.1428 # planck 2018 vi paper
RHO = 2.775e11 * OMEGAM_H2
MIN_MASS = 1. # 1e14 solar masses
MAX_MASS = 6. # 1e14 solar masses

LMIN = 600
LMAX = 3000
GLMAX = 2000
MLMAX = 6000

BEAM_FWHM = 1.5 # arcmin
NOISE_T = 10. # noise stdev in uK-acrmin
ESTS = ['TT']

COMM = MPI.COMM_WORLD

def oneprint(s):
    if COMM.Get_rank() == 0: print(s)

###############################################
# Arguments
###############################################

parser = argparse.ArgumentParser()
parser.add_argument("--lmin", type=int, default=300, help="minimum l multipole to reconstruct from")
parser.add_argument("--lmax", type=int, default=6000, help="maximum l multipole to reconstruct from")
parser.add_argument("--res", type=float, default=0.5, help="resolution of maps (in arcmin)")
parser.add_argument("--ncoords", type=int, default=5000, help="number of random clusters to stack on")
parser.add_argument("--coords", type=str, default="", help="coords file")
parser.add_argument("--verbose", action="store_true", help="output debug / verbose text")
args = parser.parse_args()

if args.lmin:
    LMIN = args.lmin
    oneprint(f"lmin set to {LMIN}.")
if args.lmax:
    LMAX = args.lmax
    oneprint(f"lmax set to {LMAX}.")
if args.res:
    RESOLUTION = np.deg2rad(args.res / 60.) 
    STACK_RES = np.deg2rad(args.res / 60.)
    SYM_RES = np.deg2rad(args.res / 60.)
    oneprint(f"resolution set to {args.res} arcmin.")
if args.ncoords:
    NCOORDS = args.ncoords
    oneprint(f"N_coords set to {NCOORDS}.")
if args.coords:
    OTHER_COORDS = np.loadtxt(args.coords)
    oneprint(f"Coords file set to {args.coords}")
    if NCOORDS > len(OTHER_COORDS[:,0]):
        NCOORDS = len(OTHER_COORDS[:,0])
        oneprint(f"Not enough coordinates in provided data file, so N_coords changed to {NCOORDS}.")

oneprint(f"Using alms from inpainted map {MAP_FILENAME}.")

if args.verbose: DEBUG = args.verbose

minstr, maxstr = "SNR", "5"
OUTPUT_STACKS_FILENAME = f"stacks-inp-vs-noninp-fake-kappa-{minstr}to{maxstr}-{LMAX}.png"
OUTPUT_RPROFILE_FILENAME = f"rbin-profiles-fake-kappa-{minstr}to{maxstr}-{LMAX}.png"
OUTPUT_RRPROFILE_FILENAME = f"rbin-profiles-diff-fake-kappa-{minstr}to{maxstr}-{LMAX}.png"

###############################################
# Lensing reconstruction
###############################################

def full_procedure(debug=DEBUG):
    if COMM.Get_rank() == 0:
        t1 = time.time()

        ialms = cs.map2alm(enmap.read_map(MAP_FILENAME), lmax=MLMAX)
        ualms = cs.map2alm(enmap.read_map(MAP2_FILENAME), lmax=MLMAX)
        kap_map = wlrecon.kapfile_to_map(kap_filename=KAP_FILENAME, mlmax=MLMAX,
                                         res=RESOLUTION)
        #s, w = enmap.fullsky_geometry(res=RESOLUTION)
        #kap_map = cs.alm2map(hp.read_alm(KAP_FILENAME),
        #                     enmap.empty(shape=s, wcs=w, dtype=np.float32))
        kap_map = kap_map - kap_map.mean()
        
        if debug:
            print("Opened kap file and created map. Total time elapsed: %0.5f seconds" % (time.time() - t1))
        
        ucls, tcls, fialms   = wlrecon.alms_inverse_filter(ialms, lmin=LMIN, lmax=LMAX,
                                                           beam_fwhm = BEAM_FWHM, noise_t = NOISE_T)
        _, _, fualms = wlrecon.alms_inverse_filter(ualms, lmin=LMIN, lmax=LMAX,
                                                   beam_fwhm = BEAM_FWHM, noise_t = NOISE_T)

        if debug:
            print("Inverse filtered alms. Total time elapsed: %0.5f seconds" % (time.time() - t1))
        
        irecon_alms = wlrecon.falafel_qe(ucls, fialms, mlmax=MLMAX, ests=ESTS, res=RESOLUTION)
        urecon_alms = wlrecon.falafel_qe(ucls, fualms, mlmax=MLMAX, ests=ESTS, res=RESOLUTION)

        if debug:
            print("Performed standard QE reconstruction. Total time elapsed: %0.5f seconds" % (time.time() - t1))
        
        Al_temp = wlrecon.tempura_norm(ESTS, ucls, tcls, lmin=LMIN, lmax=LMAX, k_ellmax=LMAX)[0]

        if debug:
            print("Computed tempura's lensing normalization for uncut QE. Total time elapsed: %0.5f seconds" % (time.time() - t1))

        sym_shape, sym_wcs = enmap.geometry(res=SYM_RES, pos=[0,0], shape=SYM_SHAPE, proj='plain')
        s_norms = wlrecon.get_s_norms(ESTS, ucls, tcls, LMIN, LMAX, sym_shape, sym_wcs)
        kells = np.arange(Al_temp.shape[0])
        
        Al_sym = wlrecon.s_norms_formatter(s_norms[ESTS[0]], kells, sym_shape, sym_wcs, LMIN, LMAX, LWIDTH)
        
        if debug:
            print("Computed symlens's lensing normalization for cut + uncut QE. Total time elapsed: %0.5f seconds" % (time.time() - t1))

        kells_factor = 1. / 2.

        isymlens_map = wlrecon.mapper(kells_factor * Al_sym, irecon_alms, res=RESOLUTION, lmin=LMIN, lmax=LMAX)
        # tempura_map = josh_wlrecon.mapper(Al_temp * kells_factor, recon_alms, res=RESOLUTION)
        usymlens_map = wlrecon.mapper(kells_factor * Al_sym, urecon_alms, res=RESOLUTION, lmin=LMIN, lmax=LMAX)

        if debug:
            print("Created kappa maps for cut + uncut QE. Total time elapsed: %0.5f seconds" % (time.time() - t1))

    else:
        map_shape, map_wcs = enmap.fullsky_geometry(res=RESOLUTION)
        isymlens_map = enmap.zeros(map_shape, wcs=map_wcs, dtype=np.float32)
        usymlens_map = enmap.zeros(map_shape, wcs=map_wcs, dtype=np.float32)
        kap_map = enmap.zeros(map_shape, wcs=map_wcs, dtype=np.float32)

    if args.coords != "":
        decs, ras = OTHER_COORDS[:,0], OTHER_COORDS[:,1]
    else: decs, ras = wlrecon.gen_coords(coords_filename=COORDS_FILENAME, Ncoords=NCOORDS,
                                              lowlim=MIN_MASS, highlim=MAX_MASS)

    COMM.Barrier()
    COMM.Bcast(isymlens_map, root=0)
    COMM.Bcast(usymlens_map, root=0)
    COMM.Bcast(kap_map, root=0)

    #mask_map = enmap.read_map(MASK_FILENAME)
    #isymlens_map = wlrecon.apply_mask(isymlens_map, mask_map)
    #usymlens_map = wlrecon.apply_mask(usymlens_map, mask_map)
    #kap_map = wlrecon.apply_mask(kap_map, mask_map)
   
    errs, avgd_maps = wlrecon.stack_and_plot_maps([isymlens_map, usymlens_map, kap_map],
                                                   ras, decs, Ncoords = NCOORDS,
                                                   labels=["Inpainted stack from sQE + symlens",
                                                           "Non-inpainted stack from sQE + symlens",
                                                           "Stack from input kappa"],
                                                   output_filename = OUTPUT_STACKS_FILENAME,
                                                   radius=RADIUS, res=STACK_RES, error_bars=True,
                                                   fontsize=30, Nbins=NBINS)
    if COMM.Get_rank() == 0:
        if debug:
            print("Stacked and averaged kappa maps, saved to %s.\nTotal time elapsed: %0.5f seconds" % (OUTPUT_STACKS_FILENAME,
                                                                                                        time.time() - t1))
        titleend_text = "(SNR > 5)"
    
        rs, profiles = wlrecon.radial_profiles(avgd_maps, error_bars=errs,
                                           labels=["Inpainted symlens + sQE",
                                                   "Non-inpainted symlens + sQE",
                                                   "kap.fits"], 
                                           output_filename=OUTPUT_RPROFILE_FILENAME,
                                           radius=RADIUS, res=STACK_RES, Nbins = 2*NBINS,
                                           titleend=titleend_text)

        for i in range(len(profiles)):
            np.savetxt(f"profiles_{i}.txt", np.column_stack((rs, profiles[i])))
        print("Saved profiles to disk.")
        profiles2 = wlrecon.radial_profile_ratio(avgd_maps[:-1], avgd_maps[-1], error_bars=errs,
                                                 labels=["Inpainted symlens + sQE",
                                                         "Non-inpainted symlens + sQE",
                                                         "kap.fits"],
                                                 output_filename=OUTPUT_RRPROFILE_FILENAME,
                                                 radius=RADIUS, res=STACK_RES, Nbins=2*NBINS,
                                                 titleend=titleend_text)

        t2 = time.time()
        if debug:
            print("Plotted radial profiles, saved to %s.\nTotal time elapsed: %0.5f seconds | %0.5f minutes" \
                % (OUTPUT_RPROFILE_FILENAME, t2 - t1, (t2 - t1) / 60.))       
        print("** COMPLETE! **")      
                
if __name__ == '__main__':
    full_procedure()
