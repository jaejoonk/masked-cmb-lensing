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

# my own file(s)
import websky_stack_and_visualize as josh_websky
import cmb_ps

###############################################
# constants
###############################################

PATH_TO_FALAFEL = "/home/joshua/research/falafel"
KAP_FILENAME = "websky/kap.fits"
#KSZ_FILENAME = "websky/ksz.fits"
ALM_FILENAME = "websky/lensed_alm.fits"
HALOS_FILENAME = "$SCRATCH/halos.pksc"
COORDS_FILENAME = "output_halos.txt"
NCOORDS = 10000

RESOLUTION = np.deg2rad(1.5 / 60.)
STACK_RES = np.deg2rad(1.0 / 60.)
RAD = np.deg2rad(0.5)
OMEGAM_H2 = 0.1428 # planck 2018 vi paper
RHO = 2.775e11 * OMEGAM_H2
MASS_CUTOFF = 4.0 # 1e14 solar masses

LMIN = 300
SYM_LMIN = LMIN
LMAX = 6000
SYM_LMAX = LMAX
MLMAX = 8000
BEAM_FWHM = 1.5 # arcmin
NOISE_T = 10. # noise stdev in uK-acrmin
ESTS = ['TT']

###############################################
# helper functions
###############################################

# should be compatible with a standard falafel installation
def import_theory_root(path = PATH_TO_FALAFEL):
    # extract theory configuration from config.yml
    config = io.config_from_yaml(PATH_TO_FALAFEL + "/input/config.yml")
    # location of theory config information
    thloc = PATH_TO_FALAFEL + "/data/" + config['theory_root']

    # load power spectra using CAMB, according to the theory config info
    # function is from orphics.cosmology
    theory = cosmology.loadTheorySpectraFromCAMB(thloc,get_dimensionless=False)
    return thloc, theory

# open and read alm file, return full sky map with appropriate dimensions
def almfile_to_map(alm_filename = ALM_FILENAME, res = RESOLUTION):
    # healpy.fitsfunc.read_alm to create a healpy object for the alm input file
    alm_hp = read_alm(alm_filename)
    # create a pixell shape + wcs for the full sky geometry with an input resolution
    shape, wcs = enmap.fullsky_geometry(res=res)
    
    # create empty map 
    omap = enmap.empty(shape, wcs, dtype=np.float32)
    # pixell.curvedsky.alm2map to overlay our alm object into the full sky map
    alm_px = alm2map(alm_hp, omap)
    return alm_px

# do same as above but only return the alm array object
def almfile_to_alms(alm_filename = ALM_FILENAME):
    return read_alm(alm_filename, hdu=(1,2,3))

# filter convergence file to an lmax and return map
def kapfile_to_map(kap_filename = KAP_FILENAME, lmax = LMAX, res = RESOLUTION):
    # create a pixell shape + wcs for the full sky geometry with an input resolution
    shape, wcs = enmap.fullsky_geometry(res=res)
    # use my function to convert the input kappa file into a map object with the provided resolution
    kap_px = josh_websky.px_to_car(kap_filename, res=res)
    
    # convert input map into alms to filter to a maximum ell, and then convert back to a map
    # with the full sky geometry w/ given resolution
    kap_alms = maps.change_alm_lmax(map2alm(kap_px, lmax=lmax), MLMAX)
    return alm2map(kap_alms, enmap.empty(shape, wcs, dtype=np.float32))

# inverse filter the alms onto [lmin, lmax] and introduce 1/f noise
def alm_inverse_filter(alm_map, lmin = LMIN, lmax = LMAX,
                       beam_fwhm = BEAM_FWHM, noise_t = NOISE_T, grad = True):
    # extract alms from the map, up to lmax
    alms = map2alm(alm_map, lmax=lmax)
    # falafel.utils to get ucls and tcls
    # this function calls cosmology.loadTheorySpectraFromCAMB
    ucls, tcls = cmb_ps.get_theory_dicts_white_noise_websky(beam_fwhm, noise_t, grad=grad)
    # returns [fTalm, fEalm, fBalm]
    fTalm = utils.isotropic_filter([alms, alms*0., alms*0.], tcls, lmin, lmax, ignore_te=True)[0]
    # beam?
    # fTalm = almxfl(fTalm, lambda ells: 1./sutils.gauss_beam(ells, beam_fwhm))
    
    return ucls, tcls, fTalm

def alms_inverse_filter(alms, lmin = LMIN, lmax = LMAX, 
                        beam_fwhm = BEAM_FWHM, noise_t = NOISE_T, grad=True):
    ucls, tcls = cmb_ps.get_theory_dicts_white_noise_websky(beam_fwhm, noise_t, grad=grad, lmax=lmax)
    falms = utils.isotropic_filter(alms, tcls, lmin, lmax)
    return ucls, tcls, falms

# run the quadratic estimator from falafel
# first index regular alms, second index gradient alms
def falafel_qe(ucls, falm, xfalm = None, mlmax=MLMAX, ests=ESTS, res=RESOLUTION):
    shape, wcs = enmap.fullsky_geometry(res=res)
    # create a pixelization object
    px = qe.pixelization(shape=shape, wcs=wcs)
    print(len(falm))
    if len(falm) != 3:
       alms = [falm, falm*0., falm*0.,]
       xalms = [None, None, None] if xfalm is None else [xfalm, xfalm*0., xfalm*0.]
    else:
       alms = falm
       xalms = [None] * 3 if xfalm is None else xfalm 
    
    # run the quadratic estimator using the theory ucls
    recon_alms = qe.qe_all(px, ucls, mlmax=mlmax,
                           fTalm=alms[0], fEalm=alms[1], fBalm=alms[2],
                           estimators=ests,
                           xfTalm=xalms[0], xfEalm=xalms[1], xfBalm=xalms[2])
    return recon_alms

# get normalizations from tempura
# first index regular alms, second index gradient alms
def tempura_norm(ests, ucls, tcls,
                 lmin=LMIN, lmax=LMAX, k_ellmax=LMAX):
    return pytempura.get_norms(ests, ucls, tcls,
                               lmin, lmax, k_ellmax, no_corr=False)[ests[0]]

# get normalizations from symlens, standard way
# unused function
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
    #thloc, th = import_theory_root()
    # ells,gt,ge,gb,gte = np.loadtxt(f"{thloc}_camb_1.0.12_grads.dat",
    #                                unpack=True,usecols=[0,1,2,3,4])

    #nells = (NOISE_T*np.pi/180./60.)**2. / sutils.gauss_beam(BEAM_FWHM,modlmap)**2.
    
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

    return alm2map(map2alm(kap_px, lmax=LMAX),
                   enmap.empty(shape, wcs, dtype=np.float32))

# generating maps by performing A_l * phi and converting to kappa
# alms are the output values from falafel QE
def mapper(norms, alms, res=RESOLUTION, lmin=LMIN, lmax=LMAX):
    shape, wcs = enmap.fullsky_geometry(res=res)
    phi_product = almxfl(alms['TT'][0].astype(np.complex128), 
                         np.array([0. if (i < lmin or i > lmax)
                                  else norms[i] for i in range(len(norms))]))
    return alm2map(phi_product, enmap.empty(shape, wcs, dtype=np.float32))

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
    dmap = distance_map(imap, res)
    if weights is not None: imap = imap * weights
    result = []
    
    for in_bin, out_bin in zip(bins, bins[1:]):
        # indices
        (xcoords, ycoords) = np.where(np.logical_and(dmap >= in_bin, dmap < out_bin))
        coords = [(xcoords[i], ycoords[i]) for i in range(len(xcoords))]
        # sum of values within the annulus of distance
        if len(coords) == 0: result.append(0.)
        else: result.append(sum([imap[x,y] for (x,y) in coords]) / len(coords))
        
    return result

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

###############################################
# own symlens wrappers
###############################################
def get_s_norms(ests, ucls, tcls, lmin, lmax, shape, wcs,
                GLMIN = None, GLMAX = None, grad_cut=False):
    feed_dict = {}
    modlmap = enmap.modlmap(shape, wcs)
    kmask = sutils.mask_kspace(shape, wcs, lmin=lmin, lmax=lmax)
    g_kmask = sutils.mask_kspace(shape, wcs,
                                 lmin=(lmin if GLMIN == None else GLMIN),
                                 lmax=(lmax if GLMAX == None else GLMAX))
    
    results = {}
    for est in ests:
        ells = np.arange(len(ucls[est]))
        feed_dict['uC_T_T'] = sutils.interp(ells, ucls[est])(modlmap)
        feed_dict['tC_T_T'] = sutils.interp(ells, tcls[est])(modlmap)
        
        norms = s.A_l(shape, wcs, feed_dict, "hdv", est, xmask=g_kmask, ymask=kmask)
        results[est] = norms
    
    return results

def s_norms_formatter(s_norms, kells, shape, wcs, lmin, lmax, lwidth):
    # return formatted norm for each estimator if multiple are provided
    if isinstance(s_norms, dict):
        result = {}
        for est in s_norms.keys():
            result[est] = s_norms_formatter(s_norms[est],kells,shape,wcs,lmin,lmax,lwidth)
        return result

    # binning 
    modlmap = enmap.modlmap(shape, wcs)
    Lrange = np.arange(lmin, lmax, lwidth)
    binner = stats.bin2D(modlmap, Lrange)

    # double check this!!
    #l_factor = modlmap * (modlmap + 1) / 4.
    centers, binned_norms = binner.bin(np.array(s_norms))
    
    # generate norms object by interpolating
    #lfactor = 1. / (kells * (kells + 1))
    Al = maps.interp(centers, binned_norms, kind='cubic')(kells)
    return Al

def s_norms_formatter_to_temp(s_norms, kells, shape, wcs, lmin, lmax, lwidth):
    Al = s_norms_formatter(s_norms, kells, shape, wcs, lmin, lmax, lwidth)
    lfactor = 1. / (kells * (kells + 1))
    
    if isinstance(Al, dict):
        result = {}
        for est in Al.keys():
            result[est] = Al[est] * lfactor
        return result
    else: return np.array(Al) * lfactor

###############################################
# Mean field calculations
###############################################

## Incorrectly generate a uniform distribution of ra,dec around spherical projection
def wrong_ra_dec(N):
    return np.random.uniform(0, 2*np.pi, N), np.random.uniform(-np.pi/2, np.pi/2, N)

## Generate a uniform distribution of ra,dec around a spherical projection
def random_ra_dec(N, zero=1e-4):
    xyz = []
    while len(xyz) < N:
        [x,y,z] = np.random.normal(size=3)
        # for rounding errors
        if (x**2 + y**2 + z**2)**0.5 > zero: xyz.append([x,y,z])
    colat, ra = hp.vec2ang(np.array(xyz))
    return ra, np.pi/2 - colat

###############################################
# Stacker and plotter functions
###############################################

def gen_coords(coords_filename=COORDS_FILENAME, Ncoords=NCOORDS,
               lowlim=None, highlim=MASS_CUTOFF):
    ras, decs = josh_websky.read_coords_from_file(coords_filename,
                                                  lowlim=lowlim, highlim=highlim)
    return ras, decs


# Stacks the input maps for Ncoords # of random coordinates, and then plots the 
# (by default) averaged stack. Returns the output stacked (or averaged) maps, 
# satisfying the same order as the input.
def stack_and_plot_maps(maps, ras, decs, Ncoords=NCOORDS, labels=None,
                        output_filename="plot.png", radius=10*RESOLUTION,
                        res=RESOLUTION, figscale=16, fontsize=13,
                        stacked_maps=False):
    struct = []
    # check if random # of coordinates asked for exceeds # of data points
    if len(ras) < Ncoords: Ncoords = len(ras) 
    for imap in maps:
        stacked_map, avg_map = josh_websky.stack_average_random(imap, ras, decs,
                                                                Ncoords=Ncoords,
                                                                radius=radius,
                                                                res = res)
        struct.append((stacked_map, avg_map))
    
    figscale_x = figscale * 3 // 4 * len(struct)
    fig, axes = plt.subplots(nrows=1, ncols=len(struct), figsize=(figscale_x, figscale))
    
    ims = []
    for i in range(len(struct)):
        im = axes[i].imshow(struct[i][0 if stacked_maps else 1], cmap='jet')
        ims.append(im)
        if labels is not None and i < len(labels):
            axes[i].set_title(labels[i], fontsize=fontsize)
    
    fig.subplots_adjust(right=0.85)
    for i in range(len(ims)):
        fig.colorbar(ims[i], ax = axes[i], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(output_filename)
    plt.clf()

    return [elem[0 if stacked_maps else 1] for elem in struct]

# Plots the binned radial profile for a bunch of maps centered at a signal, which
# for a convergence map should be the kappa convergence value as a function of radians 
# from the center. Returns kappa(radians) for each map, satisfying the same order as the input.
def radial_profiles(signal_maps, labels=None, 
                    output_filename="binned_radial_profiles.png",
                    radius=10*RESOLUTION, res=RESOLUTION, Nbins = 20, figsize=10):

    
    radius_bins = np.linspace(0, radius, Nbins)
    radius_centers = 0.5 * (radius_bins[1:] + radius_bins[:-1])
    binned_profiles = []
    for imap in signal_maps:
        binned_profiles.append(radial_avg_own(imap, res, radius_bins))
    
    plt.figure(figsize=(figsize, figsize))
    plt.title("Average binned radial profiles (kappa map)")
    plt.xlabel("arcmin")
    plt.ylabel("Kappa")
    for i in range(len(binned_profiles)):
        labeltext = ("" if (labels is None or i >= len(labels)) else labels[i])
        plt.plot(radius_centers * (180./np.pi) * 60., binned_profiles[i], label=labeltext)
    
    if labels is not None: plt.legend()
    plt.savefig(output_filename)
    plt.clf()

    return binned_profiles

def radial_profile_ratio(signal_maps, reference, labels=None, output_filename="binned_radial_profiles_ratio.png",
                         radius=10*RESOLUTION, res=RESOLUTION, Nbins=20, figsize=10):

    radius_bins = np.linspace(0, radius, Nbins)
    radius_centers = 0.5 * (radius_bins[1:] + radius_bins[:-1])
    reference_profile = np.array(radial_avg_own(reference, res, radius_bins))
    ratio_profiles = []
    for imap in signal_maps:
        imap_profile = np.array(radial_avg_own(imap, res, radius_bins))
        ratio_profiles.append((imap_profile - reference_profile) / reference_profile)

    plt.figure(figsize=(figsize, figsize))
    plt.title("Difference in avg binned radial kappa profiles")
    plt.xlabel("arcmin")
    plt.ylabel(r'$(\kappa^{x} - \kappa^{WS kap.fits}) / \kappa^{WS kap.fits}$')

    for i in range(len(ratio_profiles)):
        labeltext = ("" if (labels is None or i >= len(labels)) else labels[i])
        plt.plot(radius_centers * (180./np.pi) * 60., ratio_profiles[i], label=labeltext)

    plt.plot(radius_centers * (180./np.pi) * 60., ratio_profiles[0] * 0., '--')
    if labels is not None: plt.legend()
    plt.savefig(output_filename)
    plt.clf()

    return ratio_profiles
