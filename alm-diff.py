from orphics import stats, io
from pixell import enmap, curvedsky as cs
from falafel import utils as futils
import websky_lensing_reconstruction as w

import numpy as np

alms1 = cs.map2alm(enmap.read_map("inpainted_map_2.0_to_10.0.fits"), lmax=8000)
alms2 = futils.change_alm_lmax(w.almfile_to_alms("websky/lensed_alm.fits"), lmax=8000)

bin_edges = np.geomspace(2, len(alms1), 100)
binner = stats.bin1D(bin_edges)
bin = lambda x: binner.bin(np.arange(len(alms1)), x)

pl = io.Plotter('rCL', xyscale='loglin', figsize=(12,8))
pl._ax.set_ylim(-0.2, 0.2)
pl.add(*bin((np.abs(alms1) - np.abs(alms2[0]))/np.abs(alms2[0])), marker='o')

pl._ax.set_xlabel(r'$l')
pl._ax.set_ylabel(r'$\Delta a_{lm} / a_{lm}$')
pl.hline(y=0)
pl.done("alm-abs-plot.png")

