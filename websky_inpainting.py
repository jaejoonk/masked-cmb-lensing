import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from orphics import maps, cosmology, io, pixcov, mpi, stats
from falafel import qe, utils as futils
from pixell import enmap, curvedsky as cs, lensing as plensing
import healpy as hp
import pytempura
from mpi4py import MPI

import websky_lensing_reconstruction as josh_wlrecon
import websky_stack_and_visualize as josh_wstack
import cmb_ps

import time

ESTS = ['TT']
RESOLUTION = np.deg2rad(1.0/60.)
COMM = MPI.COMM_WORLD
rank = COMM.Get_rank() 

t1 = time.time()

# let's convert our lensed alms to a map
KAP_FILENAME = "websky/kap.fits"
ALM_FILENAME = "websky/lensed_alm.fits"
lensed_map = josh_wlrecon.almfile_to_map(alm_filename=ALM_FILENAME, res=RESOLUTION)

# generate coords
print("generate coords")
MIN_MASS = 2.0 # 1e14
MAX_MASS = 10.0 # 1e14
COORDS_FILENAME = "output_halos.txt"
COORDS = 1000

if rank == 0:
    ra, dec = josh_wstack.read_coords_from_file(COORDS_FILENAME, lowlim=MIN_MASS, highlim=MAX_MASS)
    indices = np.random.choice(len(ra), COORDS, replace=False)
    coords = np.array([[dec[i], ra[i]] for i in indices]) 

    np.savetxt("random-coords.txt", coords) 
else:
    coords = None

coords = COMM.bcast(coords, root=0)


# inpainting time
print("inpainting time")
HOLE_RADIUS = np.deg2rad(6.0/60.)
IVAR = lensed_map * 0. + 1.
OUTPUT_DIR = "inpaint_geos/"
BEAM_FWHM = 1.5 # arcmin
BEAM_SIG = BEAM_FWHM / (8 * np.log(2))**0.5 
NOISE_T = 10. # muK arcmin
LMAX = 6000
ucls, tcls = cmb_ps.get_theory_dicts_white_noise_websky(BEAM_FWHM, NOISE_T, lmax=LMAX) 
THEORY_FN = cosmology.default_theory().lCl
## probably wrong, but gaussian centered at l = 0 and sigma derived from beam fwhm   
BEAM_FN = lambda ells: maps.gauss_beam(ells, BEAM_FWHM)

pixcov.inpaint_uncorrelated_save_geometries(coords, HOLE_RADIUS, IVAR, OUTPUT_DIR,
                                            theory_fn=THEORY_FN, beam_fn=BEAM_FN,
                                            pol=False, comm=COMM)

## reconvolve beam?
shape, wcs = enmap.fullsky_geometry(res=RESOLUTION)
omap = enmap.empty(shape, wcs, dtype=np.float32)
lensed_map = cs.alm2map(cs.almxfl(cs.map2alm(lensed_map, lmax=LMAX), 
                                  BEAM_FN(np.arange(LMAX+1))),
                        omap) 

inpainted_map = pixcov.inpaint_uncorrelated_from_saved_geometries(lensed_map, OUTPUT_DIR)

# don't need parallel processes anymore?
if rank != 0: exit()

# SAVE MAP
enmap.write_map(f"inpainted_map_{MIN_MASS:.1f}_to_{MAX_MASS:.1f}.fits", inpainted_map, fmt="fits")

# try reconstructing?
print("recon time")
LMIN = 300
MLMAX = 8000

lensed_alm = cs.map2alm(lensed_map, lmax=LMAX) 
inpainted_alm = cs.map2alm(inpainted_map, lmax=LMAX)

Xdats = futils.isotropic_filter([lensed_alm, lensed_alm*0., lensed_alm*0.], tcls, lmin=LMIN, lmax=LMAX)
iXdats = futils.isotropic_filter([inpainted_alm, inpainted_alm*0., inpainted_alm*0.], tcls, lmin=LMIN, lmax=LMAX)

recon_alms = josh_wlrecon.falafel_qe(ucls, Xdats, mlmax=MLMAX, ests=ESTS, res=RESOLUTION) 
irecon_alms = josh_wlrecon.falafel_qe(ucls, iXdats, mlmax=MLMAX, ests=ESTS, res=RESOLUTION)

# normalize using tempura
Al = pytempura.get_norms(ESTS,ucls,tcls,LMIN,LMAX,k_ellmax=MLMAX,no_corr=False)

# plot cross spectra vs auto spectra?
print("plotting time")
ikalm = futils.change_alm_lmax(hp.map2alm(hp.read_map(KAP_FILENAME)), MLMAX)
norm_recon_alms = {}
norm_irecon_alms = {}
icls = hp.alm2cl(ikalm,ikalm)
ells = np.arange(len(icls))

bin_edges = np.geomspace(2,MLMAX,15)
binner = stats.bin1D(bin_edges)
bin = lambda x: binner.bin(ells,x)

for est in ESTS:
    pl = io.Plotter('CL')
    pl2 = io.Plotter('rCL',xyscale='loglin')
    pl3 = io.Plotter('CL')
    
    norm_recon_alms[est] = plensing.phi_to_kappa(hp.almxfl(recon_alms[est][0].astype(np.complex128), Al[est][0]))
    norm_irecon_alms[est] = plensing.phi_to_kappa(hp.almxfl(irecon_alms[est][0].astype(np.complex128), Al[est][0]))

    inpcls = hp.alm2cl(norm_irecon_alms[est], norm_irecon_alms[est])
    rcls = hp.alm2cl(norm_recon_alms[est], norm_recon_alms[est])

    rxinp_cls = hp.alm2cl(norm_irecon_alms[est], norm_recon_alms[est])
    rxi_cls = hp.alm2cl(norm_recon_alms[est], ikalm)
    ixinp_cls = hp.alm2cl(ikalm, norm_irecon_alms[est])
   
    pl.add(ells,(ells*(ells+1.)/2.)**2. * Al[est][0],ls='--', label="noise PS (per mode)")
    pl.add(ells,icls,label='input x input')
    pl.add(ells,rxi_cls,label='recon x input')
    pl.add(ells,rcls,label='recon x recon')
    pl.add(ells,rxinp_cls,label='recon x inpaint')
    #pl.add(ells,ixicls,label='inpaint x inpaint')

    pl3.add(ells,(ells*(ells+1.)/2.)**2. * Al[est][0],ls='--', label="noise PS (per mode)")
    pl3.add(ells,icls,label='input x input')
    pl3.add(ells,ixinp_cls,label='input x inpaint')
    pl3.add(ells,inpcls,label='inpaint x inpaint')
    pl3.add(ells,rxinp_cls,label='recon x inpaint')

    pl2.add(*bin((ixinp_cls-icls)/icls),marker='o')
    pl2._ax.set_ylabel(r'$(\kappa_{rec x i} - \kappa_{i x i}) / \kappa_{i x i}$')
    pl2.hline(y=0)
    #pl2._ax.set_ylim(-0.1,0.1)
    pl2.done(f'inpaint_ixrec_vs_ixi_recon_diff_{est}.png')
    #pl._ax.set_ylim(1e-9,1e-5)
    pl.done(f'inpaint_recon_x_input_{est}.png')
    pl3.done(f'inpaint_inpaint_x_input_{est}.png')

print("time elapsed: %0.5f seconds" % (time.time() - t1))
