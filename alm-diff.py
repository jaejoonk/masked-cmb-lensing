from orphics import stats, io, maps
from pixell import enmap, curvedsky as cs
from falafel import utils as futils
import websky_lensing_reconstruction as w
import healpy as hp
import numpy as np

LMAX = 8000
FIGSIZE = (14,10)

alms1 = cs.map2alm(enmap.read_map("inpainted_map_2.0_to_10.0.fits"), lmax=LMAX).astype(np.cdouble)
alms2 = futils.change_alm_lmax(w.almfile_to_alms("websky/lensed_alm.fits"), lmax=LMAX).astype(np.cdouble)
# convolve beam for comparison
alms2 = cs.almxfl(alms2[0], maps.gauss_beam(np.arange(LMAX+1), 1.5))

#print(len(alms1), len(alms2))
#print(LMAX * (LMAX+1))
#print(LMAX * (LMAX-1))

def alms2al(alms, lmax=LMAX):
    result = []
    for l in range(lmax+1):
        # m goes from 0 to l
        idx_l = int(l*(l+1)/2)
        result.append(sum(alms[idx_l:idx_l+l+1]))
    return np.array(result)

#al1 = alms2al(alms1) 
#al2 = alms2al(alms2[0])

bin_edges = np.geomspace(2, LMAX, 30)
binner = stats.bin1D(bin_edges)
bin = lambda x: binner.bin(ells, x)

"""pl = io.Plotter('rCL', xyscale='loglin', figsize=FIGSIZE)
pl._ax.set_ylim(-0.3, 0.3)
pl.add(*bin((np.abs(alms1) - np.abs(alms2[0]))/np.abs(alms2[0])), marker='o')

pl._ax.set_xlabel(r'index $= l^2 + l + m + 1$')
pl._ax.set_ylabel(r'$\Delta a_{lm} / a_{lm}$')
pl.hline(y=0)
pl.done("alm-abs-plot.png")

bin_edges2 = np.geomspace(1, len(al1), 100)
binner2 = stats.bin1D(bin_edges2)
bin2 = lambda x: binner.bin(np.arange(len(al1)), x)

pl2 = io.Plotter('rCL', xyscale='loglin', figsize=FIGSIZE)
pl2._ax.set_ylim(-0.4, 0.4)
pl2.add(*bin2((np.abs(al1) - np.abs(al2))/np.abs(al2)), marker='o')

pl2._ax.set_xlabel(r'marginalized l')
pl2._ax.set_ylabel(r'$\Delta a_{l} / a_{l}$')
pl2.hline(y=0)
pl2.done("al-abs-plot.png")
"""

inp_auto_cls = hp.alm2cl(alms1, alms1, lmax=LMAX)
i_auto_cls = hp.alm2cl(alms2, alms2, lmax=LMAX)
ells = np.arange(len(inp_auto_cls))

pl2 = io.Plotter('CL', figsize=FIGSIZE)
pl2.add(ells, inp_auto_cls, label="inpaint x inpaint")
pl2.add(ells, i_auto_cls, label="input x input")
pl2.done("alm-cls.png.")

pl = io.Plotter('rCL', xyscale='loglin', figsize=FIGSIZE)
pl._ax.set_ylim(-0.2, 0.2)
pl.add(*bin((inp_auto_cls - i_auto_cls)/i_auto_cls), marker='o')
pl.hline(y=0)
pl.done("alm-cls-diff.png")
