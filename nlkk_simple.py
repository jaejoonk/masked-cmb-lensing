from __future__ import print_function
from orphics import maps,io,cosmology,stats
from pixell import enmap,lensing as plensing,curvedsky as cs, utils, enplot
from enlib import bench
import numpy as np
import os,sys
import healpy as hp
from falafel import qe,utils
import pytempura

import cmb_ps
# my own code
import websky_lensing_reconstruction as wl_recon 
import time

t1 = time.time()
# The estimators to test lensing for
ests = ['TT']

#CMB_ALM_LOC = "/global/project/projectdirs/act/data/actsims_data/signal_v0.4/fullskyUnlensedCMB_alm_set00_00000.fits" 
#KAP_LOC = "/home/joshua/research/cmb_lensing_2022/masked-cmb-lensing/kap.fits"
#WEBSKY_ALM_LOC = "/home/joshua/research/cmb_lensing_2022/masked-cmb-lensing/lensed_alm.fits"
KAP_LOC = "/global/homes/j/jaejoonk/masked-cmb-lensing/websky/kap.fits"
WEBSKY_ALM_LOC = "/global/homes/j/jaejoonk/masked-cmb-lensing/websky/lensed_alm.fits"

# Decide on a geometry for the intermediate operations
res = 0.5 # resolution in arcminutes
shape,wcs = enmap.fullsky_geometry(res=np.deg2rad(res/60.),proj="car")
## create a pixelization object with shape/wcs
px = qe.pixelization(shape,wcs)

# Maximum multipole for alms
mlmax = 8000

# Filtering configuration
glmax = 2000
lmax = 3000
lmin = 300
beam_fwhm = 1.5
noise_t = 10.

ucls, tcls = cmb_ps.get_theory_dicts_white_noise_websky(beam_fwhm, noise_t, lmax=lmax)
## Get normalizations
# create a different normalization function/object for each (glmax,lmax) combination
# and combine them into a list

kells = np.arange(lmax+1)
sym_shape, sym_wcs = enmap.geometry(res=np.deg2rad(res/60.), pos=[0,0],
                                    shape=(2000,2000), proj="plain")

sym_gnorms = wl_recon.get_s_norms(ests,ucls,tcls,lmin,lmax,sym_shape,sym_wcs,GLMAX=glmax)
sym_norms = wl_recon.get_s_norms(ests,ucls,tcls,lmin,lmax,sym_shape,sym_wcs)

# LMIN = 1, LWIDTH (FOR BINNING) = 50
Als_temp = pytempura.get_norms(ests,ucls,tcls,lmin,lmax,k_ellmax=mlmax,no_corr=False)
Als_g_sym = wl_recon.s_norms_formatter(sym_gnorms,kells,sym_shape,sym_wcs,1,lmax,50)
Als_sym = wl_recon.s_norms_formatter(sym_norms,kells,sym_shape,sym_wcs,1,lmax,50)

lensed_websky_alm = hp.read_alm(WEBSKY_ALM_LOC, hdu=(1,2,3))
kap_alm = hp.map2alm(hp.read_map(KAP_LOC))
ikalm = utils.change_alm_lmax(kap_alm.astype(np.complex128), mlmax)

## Filter alms
Xdats = utils.isotropic_filter(lensed_websky_alm,tcls,lmin,lmax)
gXdats = utils.isotropic_filter(lensed_websky_alm,tcls,lmin,glmax)

## Reconstruct
"""
recon = qe.qe_all(px,ucls,mlmax,
                  fTalm=Xdats[0],fEalm=Xdats[1],fBalm=Xdats[2],
                  estimators=ests,
                  xfTalm=Xdats[0],xfEalm=Xdats[1],xfBalm=gXdats[2])

grecon = qe.qe_all(px,ucls,mlmax,
                   fTalm=Xdats[0],fEalm=Xdats[1],fBalm=Xdats[2],
                   estimators=ests,
                   xfTalm=gXdats[0],xfEalm=gXdats[1],xfBalm=gXdats[2])
"""

## Cross-correlate
kalms = {}
kalms_sym = {}
#icls = hp.alm2cl(ikalm,ikalm)
#ells = np.arange(len(icls))
#bin_edges = np.geomspace(2,mlmax,20)
#binner = stats.bin1D(bin_edges)
#bin = lambda x: binner.bin(ells,x)

ells = np.arange(lmax+1)

## Plot
# loop over all estimators
for est in ests:
    pl = io.Plotter('CL')
    pl._ax.set_title("Nlkk for \'%s\;' est, (glmax=%d, lmax=%d)" % (est, glmax, lmax))
    pl.add(ells, (ells*(ells+1.))/4. * Als_sym[est], ls="--", label=("Nlkk, glmax=%d" % lmax))
    pl.add(ells, (ells*(ells+1.))/4. * Als_g_sym[est], ls="--", label=("Nlkk, glmax=%d" % glmax))
    # pl.add(ells, (ells*(ells+1.)/2.)**2 * Als_temp[est][0], ls="--", label="Nlkk, tempura")
    
    pl.done(f'websky_nlkk_recon_{glmax}_cut_{lmax}.png')

print("Time elapsed: %0.5f seconds" % (time.time() - t1))
