###############################################
# imports
###############################################
import time, os

from pixell import enmap,utils as putils,reproject,enplot
from pixell.lensing import phi_to_kappa
from pixell.reproject import healpix2map,thumbnails
from pixell.curvedsky import alm2map,map2alm,almxfl
from pixell.pointsrcs import radial_bin

import numpy as np
import matplotlib.pyplot as plt

import healpy as hp
from healpy.fitsfunc import read_alm,read_map

import symlens as s
from symlens import utils as sutils

# utils require orphics + pyfisher + enlib
from falafel import qe, utils
import pytempura 

from orphics import maps, cosmology, io, stats, pixcov

# my own files
import websky_stack_and_visualize as josh_websky
import websky_lensing_reconstruction as josh_wlrecon
# from gcut_simple import get_theory_dicts_white_noise_websky 

###############################################
# constants
###############################################
DEBUG = True

PATH_TO_FALAFEL = "/home/joshua/research/falafel"
KAP_FILENAME = "websky/kap.fits"
#KSZ_FILENAME = "websky/ksz.fits"
ALM_FILENAME = "websky/lensed_alm.fits"
HALOS_FILENAME = "$SCRATCH/halos.pksc"
COORDS_FILENAME = "output_halos.txt"
OUTPUT_STACKS_FILENAME = "new-data-6to40-qe-stacks.png"
OUTPUT_RPROFILE_FILENAME = "new-kappa-6to40-binnedrprofiles.png"
OUTPUT_RRPROFILE_FILENAME = "new-kappa-6to40-binnedrprofiles-diff.png"
NCOORDS = 10000
NBINS = 20
LWIDTH = 50

RESOLUTION = np.deg2rad(0.5 / 60.)
STACK_RES = np.deg2rad(0.5 / 60.)
RADIUS = STACK_RES * 10. # 10 arcmin
SYM_RES = np.deg2rad(0.5 / 60.)
SYM_SHAPE = (4000,4000)
RAD = np.deg2rad(0.5)
OMEGAM_H2 = 0.1428 # planck 2018 vi paper
RHO = 2.775e11 * OMEGAM_H2
MIN_MASS = 6. # 1e14 solar masses
MAX_MASS = 40. # 1e14 solar masses

LMIN = 300
LMAX = 6000
GLMAX = 2000
MLMAX = 6500
BEAM_FWHM = 1.5 # arcmin
NOISE_T = 10. # noise stdev in uK-acrmin
ESTS = ['TT']

###############################################
# Lensing reconstruction
###############################################

def full_procedure(debug=DEBUG):
    t1 = time.time()

    alms = utils.change_alm_lmax(josh_wlrecon.almfile_to_alms(alm_filename=ALM_FILENAME),
                                 lmax=MLMAX)
            
    kap_map = josh_wlrecon.kapfile_to_map(kap_filename=KAP_FILENAME, lmax=LMAX,
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
    cut_s_norms = josh_wlrecon.get_s_norms(ESTS, xucls, xtcls, LMIN, LMAX, sym_shape, sym_wcs,
                                           GLMIN=LMIN, GLMAX=GLMAX, grad_cut=True)

    kells = np.arange(Al_temp.shape[0])
    
    Al_sym =      josh_wlrecon.s_norms_formatter(s_norms[ESTS[0]], kells, sym_shape, sym_wcs,
                                                 1, LMAX, LWIDTH)
    Al_cut_sym =  josh_wlrecon.s_norms_formatter(cut_s_norms[ESTS[0]], kells, sym_shape, sym_wcs,
                                                 1, LMAX, LWIDTH)
    
    if debug:
        print("Computed symlens's lensing normalization for cut + uncut QE. Total time elapsed: %0.5f seconds" % (time.time() - t1))

    kells_factor = 1. / 2.

    symlens_map = josh_wlrecon.mapper(kells_factor * Al_sym, recon_alms, res=RESOLUTION, lmin=0, lmax=LMAX)
    # tempura_map = josh_wlrecon.mapper(Al_temp * kells_factor, recon_alms, res=RESOLUTION)
    cut_symlens_map = josh_wlrecon.mapper(kells_factor * Al_cut_sym, cut_recon_alms, res=RESOLUTION, lmin=0, lmax=LMAX)

    if debug:
        print("Created kappa maps for cut + uncut QE. Total time elapsed: %0.5f seconds" % (time.time() - t1))

    ras, decs = josh_wlrecon.gen_coords(coords_filename=COORDS_FILENAME, Ncoords=NCOORDS,
                                        lowlim=MIN_MASS, highlim=MAX_MASS)
    avgd_maps = josh_wlrecon.stack_and_plot_maps([symlens_map, cut_symlens_map, kap_map],
                                                  ras, decs, 
                                                  Ncoords = NCOORDS,
                                                  labels=["Stack from falafel QE + symlens",
                                                          "Stack from gradient cut falafel QE + symlens",
                                                          "Stack from kap.fits"],
                                                  output_filename = OUTPUT_STACKS_FILENAME)
    
    if debug:
        print("Stacked and averaged kappa maps, saved to %s.\nTotal time elapsed: %0.5f seconds" % (OUTPUT_STACKS_FILENAME,
                                                                                                    time.time() - t1))

    profiles = josh_wlrecon.radial_profiles(avgd_maps, labels=["symlens + sQE",
                                            "grad. cut symlens + sQE", "kap.fits"], 
                                            output_filename=OUTPUT_RPROFILE_FILENAME,
                                            radius=RADIUS, res=STACK_RES, Nbins = NBINS)
    profiles2 = josh_wlrecon.radial_profile_ratio(avgd_maps[:-1], avgd_maps[-1], labels=["symlens + sQE", "grad. cut symlens + sQE"],
                                                  output_filename=OUTPUT_RRPROFILE_FILENAME,
                                                  radius=RADIUS, res=STACK_RES, Nbins=NBINS)
    if debug:
        print("Plotted radial profiles, saved to %s.\nTotal time elapsed: %0.5f seconds" % (OUTPUT_RPROFILE_FILENAME,
                                                                                            time.time() - t1))       
    print("** COMPLETE! **")      
                

if __name__ == '__main__':
    full_procedure()
