import numpy as np
import matplotlib.pyplot as plt

from orphics import io, stats
from pixell import enmap, curvedsky as cs
from falafel import utils as futils
import healpy as hp

import time

PATH_TO_SCRATCH = "/global/cscratch1/sd/jaejoonk/"
ESTS = ['TT']
RESOLUTION = np.deg2rad(0.5/60.)
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
INP_MAP_FILENAME = PATH_TO_SCRATCH + "maps/inpainted_map_data_fake3.fits"
NOINP_MAP_FILENAME = PATH_TO_SCRATCH + "maps/uninpainted_map_data_fake3.fits"

ikalm = futils.change_alm_lmax(hp.map2alm(hp.read_map(KAP_FILENAME)), MLMAX)
icls = hp.alm2cl(ikalm,ikalm)
ells = np.arange(len(icls))

# binning
bin_edges = np.arange(2,MLMAX,20)
binner = stats.bin1D(bin_edges)
bin = lambda x: binner.bin(ells,x)

# from inpainted map
inpainted_alm = cs.map2alm(enmap.read_map(INP_MAP_FILENAME), lmax=MLMAX, tweak=True)
uninpainted_alm = cs.map2alm(enmap.read_map(NOINP_MAP_FILENAME), lmax=MLMAX, tweak=True)
                       
# plot cltt
pl_tt = io.Plotter('rCL',xyscale='linlin',figsize=(12,12))
uninpainted_cls = hp.alm2cl(uninpainted_alm.astype(np.cdouble),
                            uninpainted_alm.astype(np.cdouble),
                            lmax=MLMAX)
inpainted_cls = hp.alm2cl(inpainted_alm.astype(np.cdouble),
                          inpainted_alm.astype(np.cdouble),
                          lmax=MLMAX)

pl_tt.add(*bin((inpainted_cls - uninpainted_cls) / uninpainted_cls), marker='o', label="inp. data vs non-inp. data")

pl_tt._ax.set_ylabel(r'$(C_L^{T_{inp} T_{inp}} - C_L^{TT}) /  C_L^{TT}$', fontsize=20)
pl_tt._ax.set_xlabel(r'$L$', fontsize=20)
pl_tt._ax.legend(fontsize=30)
pl_tt.hline(y=0)
pl_tt.done(f"ps_cltt_data_fake_{ESTS[0]}.png")
