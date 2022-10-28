import websky_lensing_reconstruction as josh_wlrecon
import numpy as np
from pixell import enmap

COORDS_FILENAME = "random-coords.txt"
ALM_FILENAME = "websky/lensed_alm.fits"
INP_FILENAME = "inpainted_map_2.0_to_10.0.fits"

NUM_COORDS = 10
RESOLUTION = np.deg2rad(0.5/60.) # 0.5 arcmin by default
RADIUS = 40 * RESOLUTION

# plot the thumbnails before and after inpainting
coords = np.loadtxt(COORDS_FILENAME)

inpainted_map = enmap.read_map(INP_FILENAME)
lensed_map = josh_wlrecon.almfile_to_map(ALM_FILENAME)

josh_wlrecon.lensed_vs_inpaint_map(inpainted_map, lensed_map,
                                   coords[:NUM_COORDS],
                                   radius=RADIUS, res=RESOLUTION)

# relative stacked profiles
# do separately idk

