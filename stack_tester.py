import numpy as np
import websky_lensing_reconstruction as wlrecon
from pixell import enmap

COORDS_FILENAME = "coords-snr-5.txt"
#ALM_FILENAME = "websky/lensed_alm.fits"
#INP_FILENAME = "inpainted_map_SNR_5.fits"
IMAP_FILENAME = "websky/kap.fits"

NUM_COORDS = 1000

imap = wlrecon.kapfile_to_map(kap_filename=IMAP_FILENAME, lmax=6000, res=np.deg2rad(0.5/60.))
#inpainted_map = enmap.read_map(INP_FILENAME)
#lensed_map = wlrecon.almfile_to_map(ALM_FILENAME)

coords = np.loadtxt(COORDS_FILENAME)
sampled = coords[np.random.choice(len(coords), NUM_COORDS, replace=False)]

decs, ras = sampled[:,0], sampled[:,1]

wlrecon.stack_and_plot_maps([imap, imap], ras, decs, NUM_COORDS,
                             labels=["stack on %d clusters" % NUM_COORDS, "copy"],
                             error_bars=False)