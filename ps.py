import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from orphics import maps, cosmology, io, pixcov, mpi, stats
from falafel import qe, utils as futils
from pixell import enmap, curvedsky as cs, lensing as plensing
import healpy as hp
import pytempura
from mpi4py import MPI

import websky_lensing_reconstruction as wlrecon
import cmb_ps

import time

ESTS = ['TT']
RESOLUTION = np.deg2rad(0.5/60.)
COMM = MPI.COMM_WORLD
rank = COMM.Get_rank() 

t1 = time.time()

# let's convert our lensed alms to a map
KAP_FILENAME = "websky/kap.fits"
ALM_FILENAME = "websky/lensed_alm.fits"
MAP_FILENAME = "inpainted_map_SNR_5.fits"

# try reconstructing?
print("recon time")
LMIN = 300
LMAX = 6000
MLMAX = 8000

BEAM_FWHM = 1.5
NOISE_T = 10.

inpainted_alm = cs.map2alm(enmap.read_map(MAP_FILENAME), lmax=MLMAX)
lensed_alm = futils.change_alm_lmax(wlrecon.almfile_to_alms(alm_filename=ALM_FILENAME),
                                        lmax=MLMAX)

ucls, tcls, iXdats  = wlrecon.alms_inverse_filter(inpainted_alm, lmin=LMIN, lmax=LMAX,
                                                  beam_fwhm = BEAM_FWHM, noise_t = NOISE_T)
_, _, Xdats         = wlrecon.alms_inverse_filter(lensed_alm, lmin=LMIN, lmax=LMAX,
                                                  beam_fwhm = BEAM_FWHM, noise_t = NOISE_T)

#ucls, tcls = cmb_ps.get_theory_dicts_white_noise_websky(BEAM_FWHM, NOISE_T, grad=True, lmax=MLMAX)

#Xdats = futils.isotropic_filter([lensed_alm, lensed_alm*0., lensed_alm*0.], tcls, lmin=LMIN, lmax=LMAX)
#iXdats = futils.isotropic_filter([inpainted_alm, inpainted_alm*0., inpainted_alm*0.], tcls, lmin=LMIN, lmax=LMAX)

recon_alms = wlrecon.falafel_qe(ucls, Xdats, mlmax=MLMAX, ests=ESTS, res=RESOLUTION) 
irecon_alms = wlrecon.falafel_qe(ucls, iXdats, mlmax=MLMAX, ests=ESTS, res=RESOLUTION)

# normalize using tempura
Al = pytempura.get_norms(ESTS,ucls,tcls,LMIN,LMAX,k_ellmax=MLMAX,no_corr=False)

# plot cross spectra vs auto spectra?
print("plotting time")
ikalm = futils.change_alm_lmax(hp.map2alm(hp.read_map(KAP_FILENAME)), MLMAX)
icls = hp.alm2cl(ikalm,ikalm)
ells = np.arange(len(icls))

norm_recon_alms = {}
norm_irecon_alms = {}

bin_edges = np.geomspace(2,MLMAX,15)
binner = stats.bin1D(bin_edges)
bin = lambda x: binner.bin(ells,x)

for est in ESTS:
    pl = io.Plotter('CL', figsize=(16,12))
    pl2 = io.Plotter('rCL',xyscale='loglin',figsize=(16,12))
    pl3 = io.Plotter('rCL',xyscale='loglin',figsize=(16,12))
    pl4 = io.Plotter('rCL',xyscale='loglin',figsize=(16,12))
    
    norm_recon_alms[est] = plensing.phi_to_kappa(hp.almxfl(recon_alms[est][0].astype(np.complex128), Al[est][0]))
    norm_irecon_alms[est] = plensing.phi_to_kappa(hp.almxfl(irecon_alms[est][0].astype(np.complex128), Al[est][0]))

    inpcls = hp.alm2cl(norm_irecon_alms[est], norm_irecon_alms[est])
    rcls = hp.alm2cl(norm_recon_alms[est], norm_recon_alms[est])

    #rxinp_cls = hp.alm2cl(norm_irecon_alms[est], norm_recon_alms[est])
    rxi_cls = hp.alm2cl(norm_recon_alms[est], ikalm)
    ixinp_cls = hp.alm2cl(ikalm, norm_irecon_alms[est])
   
    pl.add(ells,(ells*(ells+1.)/2.)**2. * Al[est][0],ls='--', label="noise PS (per mode)")
    #pl.add(ells,rcls,label='recon x recon')
    pl.add(ells,ixinp_cls,label='inpaint x recon')
    pl.add(ells,rxi_cls,label='recon x input')
    pl.add(ells,icls,label='input x input')
    #pl.add(ells,ixicls,label='inpaint x inpaint')

    pl2.add(*bin((ixinp_cls-icls)/icls),marker='o',label="inpaint x input")
    pl2.add(*bin((rxi_cls-icls)/icls),marker='o',label="recon x input")
    pl2._ax.set_ylabel(r'$(C_L^{\hat\kappa \kappa_{i}} - C_L^{\kappa_{i} \kappa_{i}}) /  C_L^{\kappa_{i} \kappa_{i}}$')
    pl2._ax.legend()
    pl2.hline(y=0)
    pl2._ax.set_ylim(-0.25,0.25)

    pl3.add(*bin((rcls - inpcls)/inpcls),marker='o')
    pl3._ax.set_ylabel(r'$(C_L^{\hat\kappa_{r} \hat\kappa_{r}} - C_L^{\hat\kappa_{inp} \hat\kappa_{inp}}) /  C_L^{\hat\kappa_{inp} \hat\kappa_{inp}}$')
    pl3.hline(y=0)

    pl4.add(*bin((ixinp_cls-rxi_cls)/rxi_cls),marker='o')
    pl4._ax.set_ylabel(r'$(C_L^{\hat\kappa_{inp} \kappa_{i}} - C_L^{\hat\kappa_{r} \kappa_{i}}) /  C_L^{\hat\kappa_{r} \kappa_{i}}$')
    pl4.hline(y=0)
    pl4._ax.set_ylim(-0.5,0.5)

    pl.done(f'ps_all_{est}.png')
    pl2.done(f'ps_cross_vs_auto_diff_{est}.png')
    pl3.done(f'ps_recon_vs_inp_auto_diff_{est}.png')
    pl4.done(f'ps_recon_vs_inp_cross_diff_{est}.png')
    #pl._ax.set_ylim(1e-9,1e-5)

print("time elapsed: %0.5f seconds" % (time.time() - t1))