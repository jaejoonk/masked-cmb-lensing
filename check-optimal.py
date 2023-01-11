# output a bunch of thumbnail pics on some of the coordinates
import websky_lensing_reconstruction as josh_wlrecon
import numpy as np
from pixell import enmap, curvedsky as cs
import healpy as hp
from orphics import maps, io
import cmb_ps
from falafel import utils as futils

COORDS_FILENAME = "coords-snr-5-fake-10259.txt"
ALM_FILENAME = "websky/lensed_alm.fits"

PATH_TO_SCRATCH = "/global/cscratch1/sd/jaejoonk/"
OPTIMAL_MAP = PATH_TO_SCRATCH + "optimal_filtered_websky_random.fits"
#INP2_FILENAME = PATH_TO_SCRATCH + "maps/uninpainted_map_data_fake2.fits"
#INP_FILENAME = "fake_inpainted_map_SNR_5.fits"

NUM_COORDS = 10
RESOLUTION = np.deg2rad(0.5/60.) # 0.5 arcmin by default
RADIUS = 40 * RESOLUTION

LMAX = 4000
MLMAX = 6000

# plot the thumbnails before and after inpainting
coords = np.loadtxt(COORDS_FILENAME)
sampled = coords[np.random.choice(len(coords), NUM_COORDS, replace=False)]

optimal_map = enmap.read_map(OPTIMAL_MAP)
optimal_map = cs.alm2map(cs.almxfl(cs.map2alm(optimal_map, lmax=LMAX),
                                   lambda ells: maps.gauss_beam(ells, 1.5)),
                         enmap.empty(*(enmap.fullsky_geometry(res=RESOLUTION))))
iso_alm = cs.almxfl(hp.fitsfunc.read_alm(ALM_FILENAME), 
                               lambda ells: maps.gauss_beam(ells, 1.5))
# isotropically filter
ucls, tcls = cmb_ps.get_theory_dicts_white_noise_websky(1.5, 10., grad=False, lmax=MLMAX)
ucls['TT'] = ucls['TT'][:LMAX+1]
tcls['TT'] = tcls['TT'][:LMAX+1]
iso_alm = futils.isotropic_filter([iso_alm, iso_alm*0., iso_alm*0.], tcls, 1, LMAX)[0]

iso_map = cs.alm2map(iso_alm, enmap.empty(*(enmap.fullsky_geometry(res=RESOLUTION))))
         
josh_wlrecon.optimal_vs_iso_map(optimal_map, iso_map,
                                sampled, title="websky-of-vs-iso",
                                radius=RADIUS, res=RESOLUTION)

# relative stacked profiles
# do separately idk

