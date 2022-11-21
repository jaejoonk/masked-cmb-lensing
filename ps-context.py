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

INP_MAP_FILENAMES = [PATH_TO_SCRATCH + "maps/inpainted_map_6000_onethird.fits",
                     PATH_TO_SCRATCH + "maps/inpainted_map_6000_onehalf.fits",
                     PATH_TO_SCRATCH + "maps/inpainted_map_6000_twothirds.fits",
                     PATH_TO_SCRATCH + "maps/inpainted_map_6000_one.fits"]
NOINP_MAP_FILENAMES = [PATH_TO_SCRATCH + "maps/uninpainted_map_6000_onethird.fits",
                       PATH_TO_SCRATCH + "maps/uninpainted_map_6000_onehalf.fits",
                       PATH_TO_SCRATCH + "maps/uninpainted_map_6000_twothirds.fits",
                       PATH_TO_SCRATCH + "maps/uninpainted_map_6000_one.fits"]
LABELS = ["context = 1/3 (r = 8 arcmin)", "context = 1/2 (r = 9 arcmin)",
          "context = 2/3 (r = 10 arcmin)", "context = 1 (r = 12 arcmin)"]

ikalm = futils.change_alm_lmax(hp.map2alm(hp.read_map(KAP_FILENAME)), MLMAX)
icls = hp.alm2cl(ikalm,ikalm)
ells = np.arange(len(icls))

# binning
bin_edges = np.arange(2,MLMAX,20)
binner = stats.bin1D(bin_edges)
bin = lambda x: binner.bin(ells,x)

def convert_alm_to_map(alm):
    return cs.alm2map(alm, enmap.empty(shape=full_shape, wcs=full_wcs))

# from inpainted map
inpainted_alms = [cs.map2alm(enmap.read_map(mapname), lmax=MLMAX)
                  for mapname in INP_MAP_FILENAMES]
uninpainted_alms = [cs.map2alm(enmap.read_map(mapname), lmax=MLMAX)
                    for mapname in NOINP_MAP_FILENAMES]                 
# plot cltt
pl_tt = io.Plotter('rCL',xyscale='linlin',figsize=(12,12))
inpainted_cls = [hp.alm2cl(alm.astype(np.cdouble),
                           alm.astype(np.cdouble),
                           lmax=MLMAX)
                 for alm in inpainted_alms]
uninpainted_cls = [hp.alm2cl(alm.astype(np.cdouble),
                             alm.astype(np.cdouble),
                             lmax=MLMAX)
                   for alm in uninpainted_alms]

for i in range(len(inpainted_cls)):
    pl_tt.add(*bin((inpainted_cls[i] - uninpainted_cls[i]) / uninpainted_cls[i]),
              marker='o', label=LABELS[i])

pl_tt._ax.set_ylabel(r'$(C_L^{T_{inp} T_{inp}} - C_L^{TT}) /  C_L^{TT}$', fontsize=20)
pl_tt._ax.set_xlabel(r'$L$', fontsize=20)
pl_tt._ax.legend(fontsize=30)
pl_tt.hline(y=0)
pl_tt.done(f"ps_cltt_context_fractions_{ESTS[0]}.png")

# try reconstructing?
ucls, tcls = cmb_ps.get_theory_dicts_white_noise_websky(BEAM_FWHM, NOISE_T, grad=True, lmax=LMAX)

INV_BEAM_FN = lambda ells: 1./maps.gauss_beam(ells, BEAM_FWHM)
inpainted_alms = [cs.almxfl(inpainted_alm, INV_BEAM_FN)
                  for inpainted_alm in inpainted_alms]
uninpainted_alms = [cs.almxfl(uninpainted_alm, INV_BEAM_FN)
                    for uninpainted_alm in uninpainted_alms]

iXdats = [futils.isotropic_filter([inpainted_alms[i], inpainted_alms[i]*0., inpainted_alms[i]*0.],
                                  tcls, lmin=LMIN, lmax=LMAX)
          for i in range(len(inpainted_alms))]
Xdat = futils.isotropic_filter([uninpainted_alms[0], uninpainted_alms[0]*0., uninpainted_alms[0]*0.],
                                 tcls, lmin=LMIN, lmax=LMAX)


px = qe.pixelization(shape=full_shape, wcs=full_wcs)

irecon_alms_all = [qe.qe_all(px, ucls, mlmax=MLMAX,
                             fTalm=iXdat[0], fEalm=iXdat[1], fBalm=iXdat[2],
                             estimators=ESTS,
                             xfTalm=None, xfEalm=None, xfBalm=None)
                   for iXdat in iXdats]

recon_alms = qe.qe_all(px, ucls, mlmax=MLMAX,
                            fTalm=Xdat[0], fEalm=Xdat[1], fBalm=Xdat[2],
                            estimators=ESTS,
                            xfTalm=None, xfEalm=None, xfBalm=None)

# normalize using tempura
Al = pytempura.get_norms(ESTS,ucls,tcls,LMIN,LMAX,k_ellmax=MLMAX,no_corr=False)

# plot cross spectra vs auto spectra

# use this if first bin object is doing linear bins
bin_edges2 = np.geomspace(2,LMAX,30)
binner2 = stats.bin1D(bin_edges2)
bin2 = lambda x: binner2.bin(ells,x)

for est in ESTS:
    pl = io.Plotter('rCL',xyscale='loglin',figsize=(16,12))
    pl2 = io.Plotter('rCL',xyscale='loglin',figsize=(16,12))
    
    # just need one set of uninpainted kappa
    norm_recon_alms = {}
    norm_recon_alms[est] = plensing.phi_to_kappa(hp.almxfl(recon_alms[est][0].astype(np.complex128), Al[est][0]))
    rcls = hp.alm2cl(norm_recon_alms[est], norm_recon_alms[est])
    rxi_cls = hp.alm2cl(norm_recon_alms[est], ikalm)

    for i in range(len(irecon_alms_all)):
        norm_irecon_alms = {}  
        irecon_alms = irecon_alms_all[i]

        # convolve reconstruction w/ normalization
        norm_irecon_alms[est] = plensing.phi_to_kappa(hp.almxfl(irecon_alms[est][0].astype(np.complex128), Al[est][0]))

        inpcls = hp.alm2cl(norm_irecon_alms[est], norm_irecon_alms[est])
        ixinp_cls = hp.alm2cl(norm_irecon_alms[est], ikalm)

        pl.add(*bin2((ixinp_cls-icls)/icls),marker='o',label=f"inpaint {LABELS[i]} x input Clkk")
        pl2.add(*bin2((ixinp_cls-icls)/(rxi_cls-icls)),marker='o',label=f"inpaint {LABELS[i]} / non-inpaint")
    pl.add(*bin2((rxi_cls-icls)/icls),marker='o',label="non-inpaint x input Clkk")

    pl._ax.set_xlabel(r'$L$', fontsize=20)
    pl._ax.legend(fontsize=30)
    pl._ax.set_ylabel(r'$(C_L^{\hat\kappa \kappa_{input}} - C_L^{\kappa_{input} \kappa_{input}}) /  C_L^{\kappa_{input} \kappa_{input}}$', fontsize=16)
    #pl2._ax.legend()
    pl.hline(y=0)
    pl._ax.set_ylim(-0.2,0.2)
    
    pl2._ax.set_xlabel(r'$L$', fontsize=20)
    pl2._ax.legend(fontsize=30)
    pl2._ax.set_ylabel(r'$(\Delta C_L^{\hat\kappa_{inp} \kappa} - C_L^{\kappa \kappa}) /  (\Delta C_L^{\hat\kappa \kappa} - C_L^{\kappa \kappa})$', fontsize=16)
    pl2.hline(y=1)
    pl2._ax.set_ylim(0.5,1.5)

    pl.done(f'ps_websky_context_fractions_cross_vs_auto_diff_{est}.png')
    pl2.done(f'ps_websky_context_fractions_ratio_diff_{est}.png')
    #pl3.done(f'ps_recon_vs_inp_auto_diff_{est}.png')
    #pl4.done(f'ps_recon_vs_inp_cross_diff_{est}.png')
    #pl._ax.set_ylim(1e-9,1e-5)

print("Time elapsed: %0.5f seconds" % (time.time() - t1))
