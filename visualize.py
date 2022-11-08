import numpy as np
from orphics import io
from pixell import curvedsky as cs, enmap
from falafel import utils as futils

import websky_lensing_reconstruction as josh_wlrecon

ALM_FILENAME = "websky/lensed_alm.fits"
MAP_FILENAME = "inpainted_map_SNR_5.fits"
RES = np.deg2rad(0.5/60.)
MLMAX = 8000

shape, wcs = enmap.fullsky_geometry(res=RES)
inpainted = cs.alm2map(cs.map2alm(enmap.read_map(MAP_FILENAME), lmax=MLMAX),
                       enmap.empty(shape, wcs=wcs))

original = cs.alm2map(futils.change_alm_lmax(
                        josh_wlrecon.almfile_to_alms(alm_filename=ALM_FILENAME),
                        lmax=MLMAX), enmap.empty(shape, wcs=wcs))

io.hplot(inpainted, "inpainted-lensed", downgrade=4)
io.hplot(original, "original-lensed", downgrade=4)
            