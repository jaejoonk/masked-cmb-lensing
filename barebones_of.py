import numpy as np
from orphics import maps, cosmology
from falafel import utils as futils
from pixell import enmap, curvedsky as cs
import healpy as hp
from optweight import filters
#ALM_FILENAME = "websky/lensed_alm.fits"

HOLE_RADIUS = np.deg2rad(6.0/60.)
RESOLUTION = np.deg2rad(0.5/60.)
NOISE_T = 10. # muK arcmin
BEAM_FWHM = 1.5 # arcmin
NITER = 100

FULL_SHAPE, FULL_WCS = enmap.fullsky_geometry(res=RESOLUTION)
IVAR = maps.ivar(shape=FULL_SHAPE, wcs=FULL_WCS, noise_muK_arcmin=NOISE_T)
BEAM_FN = lambda ells: maps.gauss_beam(ells, BEAM_FWHM)

LMAX = 7000

# [[dec1, ra1], [dec2, ra2], [dec3, ra3], ...]
coords = np.array([[-2.355215343687713236e-01, 1.197169200564904834e+00],
                   [2.510070513750106702e-01,3.860551650130365164e+00],
                   [1.220215871502352556e-01,2.903845166905568487e+00],
                   [-5.844911768477909497e-01,1.462714000801221825e+00],
                   [1.167396077186890757e-01,2.301096007797570397e+00]])

# problematic?
mask_bool = enmap.distance_from(FULL_SHAPE, FULL_WCS, coords.T, rmax=HOLE_RADIUS) >= HOLE_RADIUS

#alms = hp.fitsfunc.read_alm(ALM_FILENAME, hdu=(1,2,3))[0]
alms = cs.rand_alm(cosmology.default_theory().lCl('TT', np.arange(LMAX+1)), lmax=LMAX)
alms = futils.change_alm_lmax(cs.almxfl(alms, BEAM_FN), LMAX)
# stamp out holes in the ivar map, thanks Mat!
IVAR[~mask_bool] = 0.
    
theory_cls, _ = futils.get_theory_dicts(lmax=LMAX, grad=False)
theory_cls['TT'] = theory_cls['TT'][:LMAX+1]
b_ells = BEAM_FN(np.arange(len(theory_cls['TT'])))
    
filtered_alm_dict = filters.cg_pix_filter(alms, theory_cls, b_ells, LMAX,
                                          icov_pix=IVAR, cov_noise_ell=None,
                                          niter=NITER, verbose=True)

omap = cs.alm2map(filtered_alm_dict['ialm'], enmap.empty(shape=FULL_SHAPE, wcs=FULL_WCS))
enmap.write_map("optimally_filtered_map.fits", omap)
