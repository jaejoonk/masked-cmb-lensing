# output a bunch of thumbnail pics on some of the coordinates
import websky_lensing_reconstruction as josh_wlrecon
import numpy as np
from pixell import enmap

COORDS_FILENAME = "coords-snr-5.txt"
# ALM_FILENAME = "websky/lensed_alm.fits"
INP_FILENAME = "inpainted_null_map_beam_conv_6000.fits"
INP2_FILENAME = "uninpainted_null_map_beam_conv_6000.fits"
#INP_FILENAME = "fake_inpainted_map_SNR_5.fits"

NUM_COORDS = 20
RESOLUTION = np.deg2rad(0.5/60.) # 0.5 arcmin by default
RADIUS = 40 * RESOLUTION

# plot the thumbnails before and after inpainting
coords = np.loadtxt(COORDS_FILENAME)
sampled = coords[np.random.choice(len(coords), NUM_COORDS, replace=False)]

inpainted_map = enmap.read_map(INP_FILENAME)
uninpainted_map = enmap.read_map(INP2_FILENAME)

josh_wlrecon.lensed_vs_inpaint_map(inpainted_map, uninpainted_map,
                                   sampled, title="websky-inpaint-vs-noninpaint",
                                   radius=RADIUS, res=RESOLUTION)

# relative stacked profiles
# do separately idk

