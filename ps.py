import numpy as np
import matplotlib.pyplot as plt

from orphics import maps, cosmology, io, pixcov, mpi, stats
from falafel import qe, utils as futils
from pixell import enmap, curvedsky as cs, lensing as plensing, enplot
import healpy as hp
from healpy.fitsfunc import read_alm
import pytempura
from mpi4py import MPI

#import websky_lensing_reconstruction as wlrecon
import cmb_ps

import time

PATH_TO_SCRATCH = "/global/cscratch1/sd/jaejoonk/"
ESTS = ['TT']
RESOLUTION = np.deg2rad(0.5/60.)
COMM = MPI.COMM_WORLD
rank = COMM.Get_rank() 
full_shape, full_wcs = enmap.fullsky_geometry(res=RESOLUTION)
t1 = time.time()

# constants
LMIN = 600
LMAX = 3000
MLMAX = 6000

BEAM_FWHM = 1.5
NOISE_T = 10.

# let's convert our lensed alms to a map
PATH_TO_SCRATCH = "/global/cscratch1/sd/jaejoonk/"
#KAP_FILENAME = PATH_TO_SCRATCH + "sehgal/tSZ_skymap_healpix_nopell_Nside4096_y_tSZrescale0p75.fits"
KAP_FILENAME = "websky/kap.fits"
ALM_FILENAME = "websky/lensed_alm.fits"
INP_MAP_FILENAME = PATH_TO_SCRATCH + "maps/inpainted_map_data_6000.fits"
NOINP_MAP_FILENAME = PATH_TO_SCRATCH + "maps/uninpainted_map_data_6000.fits"
#BAD_MAP_FILENAME = "fake_inpainted_map_SNR_5.fits"

ikalm = futils.change_alm_lmax(hp.map2alm(hp.read_map(KAP_FILENAME)), MLMAX)
try:
    hp.fitsfunc.write_alm(f"ikalm_websky_{ESTS[0]}.fits", ikalm)
    print("Saved to ikalm_websky_TT.fits")
except OSError: print("ikalms already written.")

icls = hp.alm2cl(ikalm,ikalm)
ells = np.arange(len(icls))

# binning
#bin_edges = np.geomspace(2,MLMAX,30)
bin_edges = np.arange(2,MLMAX,20)
binner = stats.bin1D(bin_edges)
bin = lambda x: binner.bin(ells,x)

def convert_alm_to_map(alm):
    return cs.alm2map(alm, enmap.empty(shape=full_shape, wcs=full_wcs))

# from inpainted map
inpainted_alm = cs.map2alm(enmap.read_map(INP_MAP_FILENAME), lmax=MLMAX)
uninpainted_alm = cs.map2alm(enmap.read_map(NOINP_MAP_FILENAME), lmax=MLMAX)
#inpainted2_alm = cs.map2alm(enmap.read_map(MAP2_FILENAME), lmax=MLMAX)
#fake_inpainted_alm = cs.map2alm(enmap.read_map(BAD_MAP_FILENAME), lmax=MLMAX)

# from lensed_alm.fits, non-inpainted map
#lensed_alm = futils.change_alm_lmax(read_alm(ALM_FILENAME, hdu=(1,2,3)),
#                                    lmax=MLMAX)
# apples to apples
#BEAM_FN = lambda ells: maps.gauss_beam(ells, BEAM_FWHM)
#INV_BEAM_FN = lambda ells: 1./maps.gauss_beam(ells, BEAM_FWHM)
#lensed_alm = cs.almxfl(cs.almxfl(lensed_alm, BEAM_FN(np.arange(LMAX+1))), INV_BEAM_FN(np.arange(LMAX+1)))
                                
# plot cltt
pl_tt = io.Plotter('rCL',xyscale='linlin',figsize=(12,12))
uninpainted_cls = hp.alm2cl(uninpainted_alm.astype(np.cdouble),
                            uninpainted_alm.astype(np.cdouble),
                            lmax=MLMAX)
inpainted_cls = hp.alm2cl(inpainted_alm.astype(np.cdouble),
                          inpainted_alm.astype(np.cdouble),
                          lmax=MLMAX)
#inpainted2_cls = hp.alm2cl(inpainted2_alm.astype(np.cdouble),
#                           inpainted2_alm.astype(np.cdouble),
#                           lmax=MLMAX)
#fake_inpainted_cls = hp.alm2cl(fake_inpainted_alm.astype(np.cdouble),
#                               fake_inpainted_alm.astype(np.cdouble),
#                               lmax=MLMAX)

pl_tt.add(*bin((inpainted_cls - uninpainted_cls) / uninpainted_cls), marker='o', label="inp. vs non-inp.")
#pl_tt.add(*bin((inpainted2_cls - lensed_cls) / lensed_cls), marker='o', label="inp. (ivar) vs non-inp.")
#pl_tt.add(*bin((fake_inpainted_cls - lensed_cls) / lensed_cls), marker='o', label="null inp. vs non-inp.")
pl_tt._ax.set_ylabel(r'$(C_L^{T_{inp} T_{inp}} - C_L^{TT}) /  C_L^{TT}$', fontsize=20)
pl_tt._ax.set_xlabel(r'$L$', fontsize=20)
pl_tt._ax.legend(fontsize=30)
pl_tt.hline(y=0)
pl_tt.done(f"ps_cltt_2_3_websky_{ESTS[0]}.png")

# try reconstructing?
ucls, tcls = cmb_ps.get_theory_dicts_white_noise_websky(BEAM_FWHM, NOISE_T, grad=True, lmax=LMAX)

INV_BEAM_FN = lambda ells: 1./maps.gauss_beam(ells, BEAM_FWHM)
inpainted_alm = cs.almxfl(inpainted_alm, INV_BEAM_FN)
uninpainted_alm = cs.almxfl(uninpainted_alm, INV_BEAM_FN)

iXdats = futils.isotropic_filter([inpainted_alm, inpainted_alm*0., inpainted_alm*0.], tcls, lmin=LMIN, lmax=LMAX)
#iXdats2 = futils.isotropic_filter([inpainted2_alm, inpainted2_alm*0., inpainted2_alm*0.], tcls, lmin=LMIN, lmax=LMAX)
#fiXdats = futils.isotropic_filter([fake_inpainted_alm, fake_inpainted_alm*0., fake_inpainted_alm*0.], tcls, lmin=LMIN, lmax=LMAX)
Xdats = futils.isotropic_filter([uninpainted_alm, uninpainted_alm*0., uninpainted_alm*0.], tcls, lmin=LMIN, lmax=LMAX)

#ucls, tcls = cmb_ps.get_theory_dicts_white_noise_websky(BEAM_FWHM, NOISE_T, grad=True, lmax=MLMAX)

# sanity check for maps post filter pre recon
#map1 = convert_alm_to_map(iXdats[0])
#map2 = convert_alm_to_map(Xdats[0])
#io.hplot(map1, "post_filter_inpainted", downgrade=8)
#io.hplot(map2, "post_filter_recon", downgrade=8)

#del map1
#del map2

px = qe.pixelization(shape=full_shape, wcs=full_wcs)

recon_alms = qe.qe_all(px, ucls, mlmax=MLMAX,
                       fTalm=Xdats[0], fEalm=Xdats[1], fBalm=Xdats[2],
                       estimators=ESTS,
                       xfTalm=None, xfEalm=None, xfBalm=None)

irecon_alms = qe.qe_all(px, ucls, mlmax=MLMAX,
                        fTalm=iXdats[0], fEalm=iXdats[1], fBalm=iXdats[2],
                        estimators=ESTS,
                        xfTalm=None, xfEalm=None, xfBalm=None)

#irecon2_alms = qe.qe_all(px, ucls, mlmax=MLMAX,
#                        fTalm=iXdats2[0], fEalm=iXdats2[1], fBalm=iXdats2[2],
#                        estimators=ESTS,
#                        xfTalm=None, xfEalm=None, xfBalm=None)                        

#firecon_alms = qe.qe_all(px, ucls, mlmax=MLMAX,
#                         fTalm=fiXdats[0], fEalm=fiXdats[1], fBalm=fiXdats[2],
#                         estimators=ESTS,
#                         xfTalm=None, xfEalm=None, xfBalm=None)

#recon_alms = wlrecon.falafel_qe(ucls, Xdats, mlmax=MLMAX, ests=ESTS, res=RESOLUTION) 
#irecon_alms = wlrecon.falafel_qe(ucls, iXdats, mlmax=MLMAX, ests=ESTS, res=RESOLUTION)

# sanity check for maps post recon
#map1 = convert_alm_to_map(recon_alms['TT'][0])
#map2 = convert_alm_to_map(irecon_alms['TT'][0])
#io.hplot(map1, "recon_no_normalize_inpainted", downgrade=8)
#io.hplot(map2, "recon_no_normalize", downgrade=8)

#del map1
#del map2

# normalize using tempura
Al = pytempura.get_norms(ESTS,ucls,tcls,LMIN,LMAX,k_ellmax=MLMAX,no_corr=False)

# plot cross spectra vs auto spectra
norm_recon_alms = {}
norm_irecon_alms = {}
#norm_irecon2_alms = {}
#norm_firecon_alms = {}

# use this if first bin object is doing linear bins
bin_edges2 = np.geomspace(2,LMAX,40)
binner2 = stats.bin1D(bin_edges2)
bin2 = lambda x: binner2.bin(ells,x)

for est in ESTS:
    pl = io.Plotter('CL', figsize=(16,12))
    pl2 = io.Plotter('rCL',xyscale='loglin',figsize=(16,12))
    #pl3 = io.Plotter('rCL',xyscale='loglin',figsize=(16,12))
    #pl4 = io.Plotter('rCL',xyscale='loglin',figsize=(16,12))
    
    # convolve reconstruction w/ normalization
    norm_recon_alms[est] = plensing.phi_to_kappa(hp.almxfl(recon_alms[est][0].astype(np.complex128), Al[est][0]))
    hp.fitsfunc.write_alm(f"uninpainted_recon_alms_websky_2_3_{est}.fits", norm_recon_alms[est])
    norm_irecon_alms[est] = plensing.phi_to_kappa(hp.almxfl(irecon_alms[est][0].astype(np.complex128), Al[est][0]))
    hp.fitsfunc.write_alm(f"inpainted_recon_alms_websky_2_3_{est}.fits", norm_irecon_alms[est])

    #enmap.write_map("inpainted-kappa-ivar.fits", convert_alm_to_map(norm_irecon_alms[est]), fmt="fits")
    #enmap.write_map("uninpainted-kappa-ivar.fits", convert_alm_to_map(norm_recon_alms[est]), fmt="fits")
    #print("Wrote kappa maps.")

    #norm_irecon2_alms[est] = plensing.phi_to_kappa(hp.almxfl(irecon2_alms[est][0].astype(np.complex128), Al[est][0]))
    #norm_firecon_alms[est] = plensing.phi_to_kappa(hp.almxfl(firecon_alms[est][0].astype(np.complex128), Al[est][0]))

    # plot full sky maps
    #recon_map = cs.alm2map(norm_recon_alms[est], enmap.empty(shape=full_shape, wcs=full_wcs))
    #irecon_map = cs.alm2map(norm_irecon_alms[est], enmap.empty(shape=full_shape, wcs=full_wcs))
    #io.hplot(recon_map, "recon_map", downgrade=8)
    #io.hplot(irecon_map, "inpainted_recon_map", downgrade=8)

    inpcls = hp.alm2cl(norm_irecon_alms[est], norm_irecon_alms[est])
    #finpcls = hp.alm2cl(norm_firecon_alms[est], norm_firecon_alms[est])
    rcls = hp.alm2cl(norm_recon_alms[est], norm_recon_alms[est])

    #rxinp_cls = hp.alm2cl(norm_irecon_alms[est], norm_recon_alms[est])
    rxi_cls = hp.alm2cl(norm_recon_alms[est], ikalm)
    ixinp_cls = hp.alm2cl(norm_irecon_alms[est], ikalm)
    #ixinp2_cls = hp.alm2cl(norm_irecon2_alms[est], ikalm)
    #ixfinp_cls = hp.alm2cl(norm_firecon_alms[est], ikalm)

    pl.add(ells,(ells*(ells+1.)/2.)**2. * Al[est][0],ls='--', label="noise PS (per mode)")
    #pl.add(ells,rcls,label='recon x recon')
    pl.add(ells,ixinp_cls,label='inpaint (2/3) x input')
    #pl.add(ells,ixfinp_cls,label='null inpaint x input')
    pl.add(ells,rxi_cls,label='non-inpaint x input')
    pl.add(ells,icls,label='input x input')

    pl2.add(*bin2((ixinp_cls-icls)/icls),marker='o',label="inpaint (c.f. 2/3) x input Clkk")
    #pl2.add(*bin2((ixinp2_cls-icls)/icls),marker='o',label="inpaint (ivar) x input Clkk")
    pl2.add(*bin2((rxi_cls-icls)/icls),marker='o',label="non-inpaint x input Clkk")
    #pl2.add(*bin2((ixfinp_cls-icls)/icls),marker='o',label="null inpaint x input Clkk")
    #pl2.add(*bin2((ixinp_cls-rxi_cls)/rxi_cls), marker='o', label="inp. Clkk vs non-inp. Clkk")
    pl2.add(*bin2((inpainted_cls-uninpainted_cls)/uninpainted_cls), marker='o', label="inp. ClTT vs non-inp. ClTT")
    pl2._ax.set_xlabel(r'$L$', fontsize=20)
    pl2._ax.legend(fontsize=30)
    #pl2._ax.set_ylabel(r'$(C_L^{\hat\kappa \kappa_{i}} - C_L^{\kappa_{i} \kappa_{i}}) /  C_L^{\kappa_{i} \kappa_{i}}$', fontsize=16)
    #pl2._ax.legend()
    pl2.hline(y=0)
    pl2._ax.set_ylim(-0.3,0.3)

    """
    pl3.add(*bin((rcls - inpcls)/inpcls),marker='o')
    pl3._ax.set_ylabel(r'$(C_L^{\hat\kappa_{r} \hat\kappa_{r}} - C_L^{\hat\kappa_{inp} \hat\kappa_{inp}}) /  C_L^{\hat\kappa_{inp} \hat\kappa_{inp}}$', fontsize=16)
    pl3.hline(y=0)

    pl4.add(*bin((ixinp_cls-rxi_cls)/rxi_cls),marker='o')
    pl4._ax.set_ylabel(r'$(C_L^{\hat\kappa_{inp} \kappa_{i}} - C_L^{\hat\kappa_{r} \kappa_{i}}) /  C_L^{\hat\kappa_{r} \kappa_{i}}$', fontsize=16)
    pl4.hline(y=0)
    pl4._ax.set_ylim(-0.5,0.5)
    """

    pl.done(f'ps_websky_2_3_{est}.png')
    pl2.done(f'ps_websky_2_3_cross_vs_auto_diff_{est}.png')
    #pl3.done(f'ps_recon_vs_inp_auto_diff_{est}.png')
    #pl4.done(f'ps_recon_vs_inp_cross_diff_{est}.png')
    #pl._ax.set_ylim(1e-9,1e-5)

print("Time elapsed: %0.5f seconds" % (time.time() - t1))
