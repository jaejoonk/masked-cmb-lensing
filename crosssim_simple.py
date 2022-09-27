from __future__ import print_function
from orphics import maps,io,cosmology,stats
from pixell import enmap,lensing as plensing,curvedsky as cs, utils, enplot
from enlib import bench
import numpy as np
import os,sys
import healpy as hp
from falafel import qe,utils
import pytempura

# The estimators to test lensing for
# ests = ['TT','mv','mvpol','EE','TE','EB','TB']
#ests = ['mv']
ests = ['TT', 'EE', 'TE', 'EB', 'TB', 'MV', 'MVPOL']

CMB_ALM_LOC = "/global/project/projectdirs/act/data/actsims_data/signal_v0.4/fullskyUnlensedCMB_alm_set00_00000.fits" 
KAP_LOC = "/global/homes/j/jaejoonk/masked-cmb-lensing/websky/kap.fits"
WEBSKY_ALM_LOC = "/global/homes/j/jaejoonk/masked-cmb-lensing/websky/lensed_alm.fits"
# Decide on a geometry for the intermediate operations
res = 1.5 # resolution in arcminutes
shape,wcs = enmap.fullsky_geometry(res=np.deg2rad(res/60.),proj="car")
## create a pixelization object with shape/wcs
px = qe.pixelization(shape,wcs)

# Choose sim index
sindex = 1

# Maximum multipole for alms
mlmax = 5000

# Filtering configuration
lmax = 3000
lmin = 300
beam_fwhm = 1.5
noise_t = 10.

# Get theory spectra
ucls,tcls = utils.get_theory_dicts_white_noise(beam_fwhm,noise_t)

# Get normalizations
Als = pytempura.get_norms(ests,ucls,tcls,lmin,lmax,k_ellmax=mlmax,no_corr=False)
 
# Get input kappa alms
cmb_alm = hp.read_alm(CMB_ALM_LOC, hdu=(1,2,3))
kap_alm = hp.map2alm(hp.read_map(KAP_LOC))
# convert kappa to phi
phi_alm = cs.almxfl(alm=kap_alm, lfilter=lambda x: 2./(x*(x+1)))
phi_alm[0] = 0.+0j
# create new lensing map
lensed_map = plensing.lens_map_curved((3,) + shape, wcs, phi_alm, cmb_alm, output="l")

# alms to cross correlate
lensed_alm = cs.map2alm(lensed_map[0], lmax=mlmax).astype(np.complex128)
lensed_websky_alm = hp.read_alm(WEBSKY_ALM_LOC, hdu=(1,2,3))
ikalm = utils.change_alm_lmax(kap_alm.astype(np.complex128), mlmax)

# Filter
Xdat = utils.isotropic_filter(lensed_alm,tcls,lmin,lmax)
Xdat_ws = utils.isotropic_filter(lensed_websky_alm,tcls,lmin,lmax)

# Reconstruct
recon = qe.qe_all(px,ucls,mlmax,
                  fTalm=Xdat[0],fEalm=Xdat[1],fBalm=Xdat[2],
                  estimators=ests,
                  xfTalm=Xdat[0],xfEalm=Xdat[1],xfBalm=Xdat[2])

recon_ws = qe.qe_all(px,ucls,mlmax,
                     fTalm=Xdat_ws[0],fEalm=Xdat_ws[1],fBalm=Xdat_ws[2],
                     estimators=ests,
                     xfTalm=Xdat_ws[0],xfEalm=Xdat_ws[1],xfBalm=Xdat_ws[2])

# Cross-correlate and plot
kalms = {}
kalms_ws = {}
icls = hp.alm2cl(ikalm,ikalm)
ells = np.arange(len(icls))
bin_edges = np.geomspace(2,mlmax,15)
print(bin_edges)
binner = stats.bin1D(bin_edges)
bin = lambda x: binner.bin(ells,x)
print(ells.shape)
for est in ests:
    pl = io.Plotter('CL')
    pl2 = io.Plotter('rCL',xyscale='loglin')
    kalms[est] = plensing.phi_to_kappa(hp.almxfl(recon[est][0].astype(np.complex128),Als[est][0] )) # ignore curl
    kalms_ws[est] = plensing.phi_to_kappa(hp.almxfl(recon_ws[est][0].astype(np.complex128),Als[est][0]))
    pl.add(ells,(ells*(ells+1.)/2.)**2. * Als[est][0],ls='--', label="noise PS (per mode)")
    cls = hp.alm2cl(kalms[est],ikalm)
    cls_ws = hp.alm2cl(kalms_ws[est],ikalm)
    acls = hp.alm2cl(kalms[est],kalms[est])
    acls_ws = hp.alm2cl(kalms_ws[est],kalms_ws[est])
    pl.add(ells,acls,label='r x r')
    pl.add(ells,acls_ws,label='r x r (WS lensed alms)')
    pl.add(ells,cls,label = 'r x i')
    pl.add(ells,cls_ws,label='r x i (WS lensed alms)')
    pl.add(ells,icls, label = 'i x i')
    pl2.add(*bin((cls-icls)/icls),marker='o', label='actsims unlensed + WS phi alms')
    pl2.add(*bin((cls_ws-icls)/icls),marker='o', label='WS lensed alms')
    pl2.hline(y=0)
    pl2._ax.set_ylim(-0.2,0.1)
    pl2.done(f'webskyXactsims_simple_recon_diff_{est}.png')
    #pl._ax.set_ylim(1e-9,1e-5)
    pl.done(f'webskyXactsims_simple_recon_{est}.png')
