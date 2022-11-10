# output a bunch of thumbnail pics on some of the coordinates
import websky_lensing_reconstruction as josh_wlrecon
import numpy as np
from pixell import enmap

COORDS_FILENAME = "coords-snr-5.txt"
ALM_FILENAME = "websky/lensed_alm.fits"
INP_FILENAME = "inpainted_map_ivar_SNR_5.fits"
#INP_FILENAME = "fake_inpainted_map_SNR_5.fits"

NUM_COORDS = 20
RESOLUTION = np.deg2rad(0.5/60.) # 0.5 arcmin by default
RADIUS = 40 * RESOLUTION

# plot the thumbnails before and after inpainting
coords = np.loadtxt(COORDS_FILENAME)
sampled = coords[np.random.choice(len(coords), NUM_COORDS, replace=False)]

inpainted_map = enmap.read_map(INP_FILENAME)
lensed_map = josh_wlrecon.almfile_to_map(ALM_FILENAME)

josh_wlrecon.lensed_vs_inpaint_map(inpainted_map, lensed_map,
                                   sampled,
                                   radius=RADIUS, res=RESOLUTION)

# relative stacked profiles
# do separately idk

