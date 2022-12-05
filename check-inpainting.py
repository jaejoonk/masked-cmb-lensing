# output a bunch of thumbnail pics on some of the coordinates
import websky_lensing_reconstruction as josh_wlrecon
import numpy as np
from pixell import enmap, curvedsky as cs
import healpy as hp
from orphics import maps, io

COORDS_FILENAME = "coords-snr-5.txt"
ALM_FILENAME = "websky/lensed_alm.fits"

PATH_TO_SCRATCH = "/global/cscratch1/sd/jaejoonk/"
INP_FILENAME = "optimal-filtered-websky-map-6000.fits"
#INP2_FILENAME = PATH_TO_SCRATCH + "maps/uninpainted_map_data_fake2.fits"
#INP_FILENAME = "fake_inpainted_map_SNR_5.fits"

NUM_COORDS = 10
RESOLUTION = np.deg2rad(0.5/60.) # 0.5 arcmin by default
RADIUS = 40 * RESOLUTION

# plot the thumbnails before and after inpainting
coords = np.loadtxt(COORDS_FILENAME)
sampled = coords[np.random.choice(len(coords), NUM_COORDS, replace=False)]

inpainted_map = enmap.read_map(INP_FILENAME)
inpainted_map = cs.alm2map(cs.almxfl(cs.map2alm(inpainted_map, lmax=6000),
                                     lambda ells: maps.gauss_beam(ells, 1.5)),
                           enmap.empty(*(enmap.fullsky_geometry(res=RESOLUTION))))
uninpainted_map = cs.alm2map(hp.fitsfunc.read_alm(ALM_FILENAME),
                             enmap.empty(*(enmap.fullsky_geometry(res=RESOLUTION))))
io.hplot(uninpainted_map, "non_inpainted_map", downgrade=4)           
josh_wlrecon.lensed_vs_inpaint_map(inpainted_map, uninpainted_map,
                                   sampled, title="websky-of-vs-alm",
                                   radius=RADIUS, res=RESOLUTION)

# relative stacked profiles
# do separately idk

