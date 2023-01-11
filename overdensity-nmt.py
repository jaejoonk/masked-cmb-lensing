import numpy as np
import matplotlib.pyplot as plt

from orphics import maps, cosmology, io, pixcov, mpi, stats
from falafel import qe, utils as futils
from pixell import enmap, curvedsky as cs, lensing as plensing, enplot
import healpy as hp
from healpy.fitsfunc import read_alm
import cmb_ps

import pymaster as nmt

PATH_TO_SCRATCH = "/global/cscratch1/sd/jaejoonk/"
KAP_FILENAME = "websky/kap.fits"
ALM_FILENAME = "websky/lensed_alm.fits"
OVERDENSITY_FILENAME = PATH_TO_SCRATCH + "galaxy_delta_hp_2.fits"

galaxy_map = hp.read_map(OVERDENSITY_FILENAME)
nside = hp.get_nside(galaxy_map)
# first try no mask, auto power spectrum
mask = np.ones(shape=galaxy_map.shape)
print("Computing nmtfield...")
f_1 = nmt.NmtField(mask, [galaxy_map])

# Initialize binning scheme with 4 ells per bandpower
b = nmt.NmtBin.from_nside_linear(nside, 4)

# compute
print("Computing cls...")
cls = nmt.compute_full_master(f_1, f_1, b)
ells = b.get_effective_ells()

ell_arr = b.get_effective_ells()
print("plotting...")
plt.plot(ell_arr, cls[0], 'r-', label='TT')
plt.loglog()
plt.xlabel('$\\ell$', fontsize=16)
plt.ylabel('$C_\\ell$', fontsize=16)
plt.legend(loc='upper right', ncol=2, labelspacing=0.1)
plt.savefig("overdensity-nmt.png")



