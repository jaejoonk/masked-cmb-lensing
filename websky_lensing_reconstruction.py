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

from orphics import cosmology, io, stats, pixcov

# my own file
import websky_stack_and_visualize as josh_websky

###############################################
# constants
###############################################

PATH_TO_FALAFEL = "/home/joshua/research/falafel"
KAP_FILENAME = "kap.fits"
KSZ_FILENAME = "ksz.fits"
ALM_FILENAME = "lensed_alm.fits"
HALOS_FILENAME = "halos_10x10.pksc"
NCOORDS = 1000

RESOLUTION = np.deg2rad(1.5 / 60.)
STACK_RES = np.deg2rad(1.0 / 60.)
RAD = np.deg2rad(0.5)
OMEGAM_H2 = 0.1428 # planck 2018 vi paper
RHO = 2.775e11 * OMEGAM_H2
MASS_CUTOFF = 4.0 # 1e14 solar masses

LMIN = 300
LMAX = 6000
BEAM_FWHM = 1.5 # arcmin
NOISE_T = 10. # noise stdev in uK-acrmin
ESTS = ['TT']

###############################################
# helper functions
###############################################

# should be compatible with a standard falafel installation
def import_theory_root(path = PATH_TO_FALAFEL):
    config = io.config_from_yaml(PATH_TO_FALAFEL + "/input/config.yml")
    thloc = PATH_TO_FALAFEL + "/data/" + config['theory_root']

    theory = cosmology.loadTheorySpectraFromCAMB(thloc,get_dimensionless=False)
    return thloc, theory

# open and read alm file, return full sky map with appropriate dimensions
def almfile_to_map(alm_filename = ALM_FILENAME, res = RESOLUTION):
    alm_hp = read_alm(alm_filename)
    shape, wcs = enmap.fullsky_geometry(res=res)
    
    # create empty map to overlay our map
    omap = enmap.empty(shape, wcs, dtype=np.float32)
    alm_px = alm2map(alm_hp, omap)
    return alm_px

# inverse filter the alms onto [lmin, lmax] and introduce 1/f noise
def alm_inverse_filter(alm_map, lmin = LMIN, lmax = LMAX,
                       beam_fwhm = BEAM_FWHM, noise_t = NOISE_T):
    alms = map2alm(alm_map, lmax=lmax)
    ucls, tcls = utils.get_theory_dicts_white_noise(beam_fwhm, noise_t)
    fTalm = utils.isotropic_filter([alms, alms*0., alms*0.], tcls,
                                    lmin, lmax, ignore_te=True)[0]
    
    return ucls, tcls, fTalm

# run the quadratic estimator from falafel
# first index regular alms, second index gradient alms
def falafel_qe(ucls, fTalm, lmax=LMAX, ests=ESTS):
    shape, wcs = enmap.fullsky_geometry(res=res)
    px = qe.pixelization(shape=shape, wcs=wcs)

    recon_alms = qe.qe_all(px, ucls, fTalm=fTalm, fEalm=fTalm*0.,
                           fBalm=fTalm*0., mlmax=lmax, estimators=ests)
    return recon_alms[ests[0]]

# get normalizations from tempura
# first index regular alms, second index gradient alms
def tempura_norm(ests, ucls, tcls,
                 lmin=LMIN, lmax=LMAX, k_ellmax=LMAX):
    return pytempura.get_norms(ests, ucls, tcls,
                               lmin, lmax, k_ellmax, no_corr=False)[ests[0]]

# get normalizations from symlens
def symlens_norm(sym_lmin=SYM_LMIN, sym_lmax=SYM_LMAX,
                 sym_glmin=SYM_LMIN, sym_glmax=SYM_LMAX):


    sym_shape, sym_wcs = enmap.geometry(res=RESOLUTION, pos=[0,0],
                                        shape=(1000,1000), proj='plain') 

    kmask = sutils.mask_kspace(sym_shape, sym_wcs, lmin=sym_lmin, lmax=sym_lmax)
    g_kmask = sutils.mask_kspace(sym_shape, sym_wcs, lmin=sym_glmin, lmax=sym_glmax)

    # generate a feed_dict
    feed_dict = {}
    modlmap = enmap.modlmap(sym_shape, sym_wcs)

    # from falafel/utils.py
    thloc, th = import_theory_root()
    # ells,gt,ge,gb,gte = np.loadtxt(f"{thloc}_camb_1.0.12_grads.dat",
    #                                unpack=True,usecols=[0,1,2,3,4])

    nells = (NOISE_T*np.pi/180./60.)**2. / sutils.gauss_beam(BEAM_FWHM,modlmap)**2.
    
    feed_dict['uC_T_T'] = th.lCl('TT',modlmap)
    feed_dict['tC_T_T'] = feed_dict['uC_T_T'] + nells

    # generate normalization
    return s.A_l(sym_shape,sym_wcs,feed_dict,"hdv","TT",xmask=g_kmask,ymask=kmask)

# run full lensing reconstruction from falafel + tempura
# and return a lensed map
def reconstruct(alm_filename=ALM_FILENAME, res=RESOLUTION,
                lmin=LMIN, lmax=LMAX, beam_fwhm=BEAM_FWHM,
                noise_t = NOISE_T, ests=ESTS, k_ellmax=LMAX, grad=False):

    shape, wcs = enmap.fullsky_geometry(res=res)
    GRAD = 1 if grad else 0

    alm_map = almfile_to_map(alm_filename, res)
    ucls, tcls, fTalm = alm_inverse_filter(alm_map, lmin, lmax,
                                           beam_fwhm, noise_t)
    recon_alms = falafel_qe(ucls, fTalm, lmax, ests)
    kappa_norms = tempura_norm(ests, ucls, tcls, lmin, lmax, k_ellmax)

    kappa_alms = phi_to_kappa(almxfl(recon_alms[GRAD].astype(np.complex128),
                                     kappa_norms[GRAD]))
    
    return alm2map(kappa_alms, enmap.empty(shape, wcs, dtype=np.float32))

# read and open kappa map and filter up to lmax
def read_kappa_map(kap_filename=KAP_FILENAME, res=RESOLUTION):
    shape, wcs = enmap.fullsky_geometry(res=res)
    kap_px = josh_websky.px_to_car("../" + kap_filename, res=res)

    return alm2map(map2alm(kap_px, lmax=lmax),
                   enmap.empty(shape, wcs, dtype=np.float32))

# stack recon maps 
def stack_recon_maps(kappa_map, kapfile_map, 
                    halos_filename=HALOS_FILENAME, res=STACK_RES,
                     output_filename="recon.png"):
    ra, dec = josh_websky.catalog_to_coords(filename=halos_filename)


    # stack QE recon'd maps 
    stack_map, avg_map = josh_websky.stack_average_random(kappa_map, ra, dec, Ncoords=100,
                                                        radius=10*RES, res=RES)

    # stack kap.fits convergence map
    stack_kap, avg_kap = josh_websky.stack_average_random(kap_map, ra, dec, Ncoords=100,
                                                        radius=10*RES, res=RES)                                                     

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 24))

    im1 = axes[0].imshow(avg_map, cmap='jet')
    axes[0].set_title("Stack from falafel QE", fontsize=15)
    im2 = axes[1].imshow(avg_kap, cmap='jet')
    axes[1].set_title("Stack from kap.fits", fontsize=15)

    fig.subplots_adjust(right=0.85)
    fig.colorbar(im1, ax = axes[0], fraction=0.046, pad=0.04)
    fig.colorbar(im2, ax = axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(output_filename)

# symlens normalization

###############################################
# own radial binning function
###############################################

# works for rectangles too!
# returns map where values represent arcmin distance from the center of the map
def distance_map(imap, res):
    [xdim, ydim] = imap.shape
    xvec, yvec = np.ones(xdim), np.ones(ydim)
    # create an array of size N, centered at 0 and incremented by pixel count
    xinds, yinds = (np.arange(xdim) + 0.5 - xdim/2.), (np.arange(ydim) + 0.5 - ydim/2.)
    # compute the outer product matrix: X[i, j] = onesvec[i] * inds[j] for i,j 
    # in range(N), which is just N rows copies of inds - for the x dimension
    x = np.outer(xvec, yinds)
    y = np.outer(xinds, yvec)
    r = np.sqrt(x**2 + y**2)
    
    return r * res

# gonna write my own function
def radial_sum_own(imap, res, bins, weights=None):
    dmap = distance_map(imap, res)
    if weights is not None: imap = imap * weights
    result = []
    
    for in_bin, out_bin in zip(bins, bins[1:]):
        # indices
        (xcoords, ycoords) = np.where(np.logical_and(dmap >= in_bin, dmap < out_bin))
        coords = [(xcoords[i], ycoords[i]) for i in range(len(xcoords))]
        # sum of values within the annulus of distance
        result.append(sum([imap[x,y] for (x,y) in coords]))
        
    return result

# currently buggy, working on using this for modlmap templates
def radial_lsum_own(imap, shape, wcs, bins, weights=None):
    dmap = enmap.modlmap(shape, wcs)
    if weights is not None: imap = imap * weights
    result = []
    
    for in_bin, out_bin in zip(bins, bins[1:]):
        # indices
        (xcoords, ycoords) = np.where(np.logical_and(dmap >= in_bin, dmap < out_bin))
        coords = [(xcoords[i], ycoords[i]) for i in range(len(xcoords))]
        # sum of values within the annulus of distance
        result.append(sum([imap[x,y] for (x,y) in coords]))
        
    return result

# binning and adding then dividing, probably buggy
def radial_avg_own(imap, res, bins, weights=None):
    
    numerator = radial_sum_own(imap, res, bins, weights)
    map_ones = 1. + enmap.empty(imap.shape, imap.wcs, dtype=np.float32)
    denominator = radial_sum_own(map_ones, res, bins, None)
    assert len(numerator) == len(denominator), "unequal sizes of numerator + denominator for averaging"
    
    return [numerator[i] / denominator[i] for i in range(len(numerator))]

# binning and adding
def radial_sum_own(imap, res, bins, weights=None):
    dmap = distance_map(imap, res)
    if weights is not None: imap = imap * weights
    result = []
    
    for in_bin, out_bin in zip(bins, bins[1:]):
        # indices
        (xcoords, ycoords) = np.where(np.logical_and(dmap >= in_bin, dmap < out_bin))
        coords = [(xcoords[i], ycoords[i]) for i in range(len(xcoords))]
        # sum of values within the annulus of distance
        result.append(sum([imap[x,y] for (x,y) in coords]) / len(coords))
        
    return result

