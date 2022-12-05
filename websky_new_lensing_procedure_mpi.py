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
import websky_lensing_reconstruction as josh_wlrecon
import websky_stack_and_visualize as josh_websky

import argparse
###############################################
# constants
###############################################
DEBUG = True

PATH_TO_SCRATCH = "/global/cscratch1/sd/jaejoonk/"
PATH_TO_FALAFEL = "/home/joshua/research/falafel"
KAP_FILENAME = "websky/kap.fits"
ALM_FILENAME = "websky/lensed_alm.fits"
MAP_FILENAME = PATH_TO_SCRATCH + "maps/inpainted_map_websky_fake3.fits"
HALOS_FILENAME = PATH_TO_SCRATCH + "halos.pksc"
COORDS_FILENAME = "output_halos.txt"
NCOORDS = 10000
OTHER_COORDS = None
NBINS = 20
LWIDTH = 50

RESOLUTION = np.deg2rad(0.5 / 60.)
STACK_RES = np.deg2rad(0.5 / 60.)
RADIUS = STACK_RES * 10. # 10 arcmin
SYM_RES = np.deg2rad(0.5 / 60.)
SYM_SHAPE = (2000,2000)
RAD = np.deg2rad(0.5)
OMEGAM_H2 = 0.1428 # planck 2018 vi paper
RHO = 2.775e11 * OMEGAM_H2
MIN_MASS = 1. # 1e14 solar masses
MAX_MASS = 6. # 1e14 solar masses

LMIN = 300
LMAX = 3000
GLMAX = 2000
MLMAX = 4000

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
parser.add_argument("--snr", action="store_true", help="if using a snr cut catalog or not")
parser.add_argument("--minmass", type=float, default=1., help="minimum mass of clusters to stack on (x 1e14)")
parser.add_argument("--maxmass", type=float, default=40., help="maximum mass of clusters to stack on (x 1e14)")
parser.add_argument("--lmin", type=int, default=300, help="minimum l multipole to reconstruct from")
parser.add_argument("--lmax", type=int, default=6000, help="maximum l multipole to reconstruct from")
parser.add_argument("--glmax", type=int, default=2000, help="gradient cut max l multipole")
parser.add_argument("--res", type=float, default=0.5, help="resolution of maps (in arcmin)")
parser.add_argument("--ncoords", type=int, default=5000, help="number of random clusters to stack on")
parser.add_argument("--coords", type=str, default="", help="coords file")
parser.add_argument("--asmap", action="store_true", help="if input alms are provided as a map")
parser.add_argument("--verbose", action="store_true", help="output debug / verbose text")
args = parser.parse_args()

if args.snr:
    oneprint(f"Using SNR cut clusters")
if args.minmass and not args.snr:
    MIN_MASS = args.minmass
    oneprint(f"Minimum mass set to {MIN_MASS}e14 M_sun.")
if args.maxmass and not args.snr: 
    MAX_MASS = args.maxmass
    oneprint(f"Maximum mass set to {MAX_MASS}e14 M_sun.")
if args.lmin:
    LMIN = args.lmin
    oneprint(f"lmin set to {LMIN}.")
if args.lmax:
    LMAX = args.lmax
    oneprint(f"lmax set to {LMAX}.")
if args.glmax:
    GLMAX = args.glmax
    oneprint(f"glmax set to {GLMAX}.")
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

oneprint(f"Using alms from inpainted map {MAP_FILENAME}.") \
    if args.asmap else oneprint(f"Using alms from {ALM_FILENAME}.")

if args.verbose: DEBUG = args.verbose

minstr, maxstr = int(MIN_MASS), int(MAX_MASS)
if args.snr: minstr, maxstr = "SNR", "5"
inpaintstr = "inpaint-" if args.asmap else ""
OUTPUT_STACKS_FILENAME = f"websky-fake3-{inpaintstr}kappa-{minstr}to{maxstr}-{LMAX}-qe-stacks-mpi.png"
OUTPUT_RPROFILE_FILENAME = f"websky-fake3-{inpaintstr}kappa-{minstr}to{maxstr}-{LMAX}-rbin-profiles-err-mpi.png"
OUTPUT_RRPROFILE_FILENAME = f"websky-fake3-{inpaintstr}kappa-{minstr}to{maxstr}-{LMAX}-rbin-profiles-diff-mpi.png"

###############################################
# Lensing reconstruction
###############################################

def full_procedure(debug=DEBUG):
    if COMM.Get_rank() == 0:
        t1 = time.time()

        if args.asmap:
            alms = cs.map2alm(enmap.read_map(MAP_FILENAME), lmax=MLMAX)
        else: alms = utils.change_alm_lmax(josh_wlrecon.almfile_to_alms(alm_filename=ALM_FILENAME),
                                        lmax=MLMAX)
            
        kap_map = josh_wlrecon.kapfile_to_map(kap_filename=KAP_FILENAME, mlmax=MLMAX,
                                            res=RESOLUTION)
        kap_map = kap_map - kap_map.mean()
        
        if debug:
            print("Opened kap file and created map. Total time elapsed: %0.5f seconds" % (time.time() - t1))
        
        ucls, tcls, falm    = josh_wlrecon.alms_inverse_filter(alms, lmin=LMIN, lmax=LMAX,
                                                               beam_fwhm = BEAM_FWHM, noise_t = NOISE_T)
        xucls, xtcls, xfalm = josh_wlrecon.alms_inverse_filter(alms, lmin=LMIN, lmax=GLMAX,
                                                               beam_fwhm = BEAM_FWHM, noise_t = NOISE_T)

        if debug:
            print("Inverse filtered alms. Total time elapsed: %0.5f seconds" % (time.time() - t1))
        
        recon_alms     = josh_wlrecon.falafel_qe(ucls, falm, mlmax=MLMAX, ests=ESTS, res=RESOLUTION)
        cut_recon_alms = josh_wlrecon.falafel_qe(xucls, falm, xfalm=xfalm, mlmax=MLMAX, ests=ESTS, res=RESOLUTION)

        if debug:
            print("Performed grad-cut + standard QE reconstruction. Total time elapsed: %0.5f seconds" % (time.time() - t1))
        
        Al_temp = josh_wlrecon.tempura_norm(ESTS, ucls, tcls, lmin=LMIN, lmax=LMAX, k_ellmax=LMAX)[0]

        if debug:
            print("Computed tempura's lensing normalization for uncut QE. Total time elapsed: %0.5f seconds" % (time.time() - t1))

        sym_shape, sym_wcs = enmap.geometry(res=SYM_RES, pos=[0,0], shape=SYM_SHAPE, proj='plain')
        s_norms =     josh_wlrecon.get_s_norms(ESTS, ucls, tcls, LMIN, LMAX, sym_shape, sym_wcs)
        cut_s_norms = josh_wlrecon.get_s_norms(ESTS, ucls, tcls, LMIN, LMAX, sym_shape, sym_wcs,
                                               GLMIN=LMIN, GLMAX=GLMAX)

        kells = np.arange(Al_temp.shape[0])
        
        Al_sym =      josh_wlrecon.s_norms_formatter(s_norms[ESTS[0]], kells, sym_shape, sym_wcs,
                                                     1, LMAX, LWIDTH)
        Al_cut_sym =  josh_wlrecon.s_norms_formatter(cut_s_norms[ESTS[0]], kells, sym_shape, sym_wcs,
                                                     1, LMAX, LWIDTH)

        
        if debug:
            print("Computed symlens's lensing normalization for cut + uncut QE. Total time elapsed: %0.5f seconds" % (time.time() - t1))

        kells_factor = 1. / 2.

        symlens_map = josh_wlrecon.mapper(kells_factor * Al_sym, recon_alms, res=RESOLUTION, lmin=1, lmax=LMAX)
        # tempura_map = josh_wlrecon.mapper(Al_temp * kells_factor, recon_alms, res=RESOLUTION)
        cut_symlens_map = josh_wlrecon.mapper(kells_factor * Al_cut_sym, cut_recon_alms, res=RESOLUTION, lmin=1, lmax=LMAX)

        if debug:
            print("Created kappa maps for cut + uncut QE. Total time elapsed: %0.5f seconds" % (time.time() - t1))

    else:
        map_shape, map_wcs = enmap.fullsky_geometry(res=RESOLUTION)
        symlens_map = enmap.zeros(map_shape, wcs=map_wcs, dtype=np.float32)
        cut_symlens_map = enmap.zeros(map_shape, wcs=map_wcs, dtype=np.float32)
        kap_map = enmap.zeros(map_shape, wcs=map_wcs, dtype=np.float32)

    if args.coords != "":
        decs, ras = OTHER_COORDS[:,0], OTHER_COORDS[:,1]
    else: decs, ras = josh_wlrecon.gen_coords(coords_filename=COORDS_FILENAME, Ncoords=NCOORDS,
                                              lowlim=MIN_MASS, highlim=MAX_MASS)

    COMM.Barrier()
    COMM.Bcast(symlens_map, root=0)
    COMM.Bcast(cut_symlens_map, root=0)
    COMM.Bcast(kap_map, root=0)
   
    errs, avgd_maps = josh_wlrecon.stack_and_plot_maps([symlens_map, cut_symlens_map, kap_map],
                                                        ras, decs, Ncoords = NCOORDS,
                                                        labels=["Stack from falafel QE + symlens",
                                                                "Stack from gradient cut falafel QE + symlens",
                                                                "Stack from kap.fits"],
                                                        output_filename = OUTPUT_STACKS_FILENAME,
                                                        radius=RADIUS, res=STACK_RES, error_bars=True,
                                                        fontsize=30, Nbins=NBINS)
    if COMM.Get_rank() == 0:
        if debug:
            print("Stacked and averaged kappa maps, saved to %s.\nTotal time elapsed: %0.5f seconds" % (OUTPUT_STACKS_FILENAME,
                                                                                                        time.time() - t1))
        if args.snr:
            titleend_text = "(SNR > 5)"
        else:
            titleend_text = f"(glmax={GLMAX}, lmax={LMAX}, {MIN_MASS*1e14:.2e} ~ {MAX_MASS*1e14:.2e} M_sun)"
    
        profiles = josh_wlrecon.radial_profiles(avgd_maps, error_bars=errs, labels=["symlens + sQE",
                                                "grad. cut symlens + sQE", "kap.fits"], 
                                                output_filename=OUTPUT_RPROFILE_FILENAME,
                                                radius=RADIUS, res=STACK_RES, Nbins = NBINS,
                                                titleend=titleend_text)
        profiles2 = josh_wlrecon.radial_profile_ratio(avgd_maps[:-1], avgd_maps[-1], error_bars=errs,
                                                      labels=["symlens + sQE", "grad. cut symlens + sQE"],
                                                      output_filename=OUTPUT_RRPROFILE_FILENAME,
                                                      radius=RADIUS, res=STACK_RES, Nbins=NBINS,
                                                      titleend=titleend_text)

        t2 = time.time()
        if debug:
            print("Plotted radial profiles, saved to %s.\nTotal time elapsed: %0.5f seconds | %0.5f minutes" \
                % (OUTPUT_RPROFILE_FILENAME, t2 - t1, (t2 - t1) / 60.))       
        print("** COMPLETE! **")      
                
if __name__ == '__main__':
    full_procedure()
