from __future__ import print_function
from orphics import maps,io,cosmology,stats
from pixell import enmap,lensing as plensing,curvedsky as cs, utils, enplot
from enlib import bench
import numpy as np
import os,sys
import healpy as hp
from falafel import qe,utils
import pytempura
# mostly A. van Engelen from the websky project
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
glmaxes = [6000, 2000]
lmaxes = [6000, 6000]
lmin = 300
beam_fwhm = 1.5
noise_t = 10.

## Theory
ucls, tcls = cmb_ps.get_theory_dicts_white_noise_websky(beam_fwhm, noise_t, lmax=mlmax)

## Get normalizations
# create a different normalization function/object for each (glmax,lmax) combination
# and combine them into a list
kells = np.arange(mlmax+1)
sym_shape, sym_wcs = enmap.geometry(res=np.deg2rad(res/60.), pos=[0,0],
                                    shape=(2000,2000), proj="plain")
sym_gnorms = [wl_recon.get_s_norms(ests,ucls,tcls,lmin,lmaxes[i],
                                   sym_shape,sym_wcs,GLMAX=glmaxes[i]) 
              for i in range(len(lmaxes))]

# LMIN = 1, LWIDTH (FOR BINNING) = 50
#Als_temp = pytempura.get_norms(ests,ucls,tcls,lmin,lmax,k_ellmax=mlmax,no_corr=False)
Als_sym = [wl_recon.s_norms_formatter_to_temp(sym_gnorms[i],kells,sym_shape,sym_wcs,1,lmaxes[i],50)
           for i in range(len(lmaxes))]

## Get lensed alms to cross correlate
lensed_websky_alm = hp.read_alm(WEBSKY_ALM_LOC)
kap_alm = hp.map2alm(hp.read_map(KAP_LOC))
ikalm = utils.change_alm_lmax(kap_alm.astype(np.complex128), mlmax)

## Filter alms
Xdats = [utils.isotropic_filter([lensed_websky_alm,lensed_websky_alm*0.,lensed_websky_alm*0.],tcls,lmin,lmax)
         for lmax in lmaxes]
gXdats = [utils.isotropic_filter([lensed_websky_alm,lensed_websky_alm*0.,lensed_websky_alm*0.],tcls,lmin,glmax)
          for glmax in glmaxes]

## Reconstruct
recons = [qe.qe_all(px,ucls,mlmax,
                   fTalm=Xdats[i][0],fEalm=Xdats[i][1],fBalm=Xdats[i][2],
                   estimators=ests,
                   xfTalm=gXdats[i][0],xfEalm=gXdats[i][1],xfBalm=gXdats[i][2])
          for i in range(len(lmaxes))]

## Cross-correlate
kalms = {}
kalms_sym = {}
icls = hp.alm2cl(ikalm,ikalm)
ells = np.arange(len(icls))
bin_edges = np.geomspace(2,mlmax,20)
binner = stats.bin1D(bin_edges)
bin = lambda x: binner.bin(ells,x)

## Plot
# loop over all estimators
for est in ests:
    for j in range(len(lmaxes)):
        glmax, lmax = glmaxes[j], lmaxes[j]
        Als, recon = Als_sym[j], recons[j]

        pl = io.Plotter('CL')
        pl._ax.set_title("recon vs input for \'%s\' est, (glmax=%d, lmax=%d)" % (est, glmax, lmax))
        pl2 = io.Plotter('rCL',xyscale='loglin')
        pl2._ax.set_title("Delta C_l^(k-hat,k) / C_l^(kk) for \'%s\' est (glmax=%d, lmax=%d)" % (est, glmax, lmax))
        #kalms[est] = plensing.phi_to_kappa(hp.almxfl(recon[est][0].astype(np.complex128),Als_temp[est][0])) # ignore curl
        kalms[est] = plensing.phi_to_kappa(hp.almxfl(recon[est][0].astype(np.complex128),Als[est])) # ignore curl
        pl.add(ells,(ells*(ells+1.)/2.)**2. * Als[est],ls='--', label="noise PS (per mode)")
        cls = hp.alm2cl(kalms[est],ikalm)
        acls = hp.alm2cl(kalms[est],kalms[est])
        pl.add(ells,acls,label='r x r')
        pl.add(ells,cls,label = 'r x i')
        pl.add(ells,icls, label = 'i x i')
        pl2.add(*bin((cls-icls)/icls),marker='o')
        pl2.hline(y=0)
        pl2._ax.set_ylim(-0.1,0.05)
        pl2.done(f'websky_gcut_filtertest_recon_diff_{est}_{lmax}_{j}.png')
        #pl._ax.set_ylim(1e-9,1e-5)
        pl.done(f'websky_gcut_filtertest_recon_{est}_{lmax}_{j}.png')

print("Time elapsed: %0.5f seconds" % (time.time() - t1))
