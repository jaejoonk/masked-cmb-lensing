import camb
import pyfisher
import orphics
from orphics import cosmology, stats

from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt

from camb import model
from camb.sources import GaussianSourceWindow, SplinedSourceWindow

import argparse

###############################################
# Arguments
###############################################
DEBUG = False
FSKYDEG2 = 3000.
FSKY = FSKYDEG2 / (4*np.pi*(180./np.pi)**2.)
NGAL = 0.2 # arcmin^-2
LMAX = 400
MU = 1.5
SIGMA = 0.2
USEFWHM = False
BINWIDTH = 5
BIAS = 1.0
NOISEFILE = "noise.txt"

parser = argparse.ArgumentParser()
parser.add_argument("--fsky", type=float, default=FSKY, help="sky fraction as decimal")
parser.add_argument("--fskydeg2", type=float, default=FSKYDEG2, help="sky fraction as sq. deg.")
parser.add_argument("--lmax", type=int, default=LMAX, help="maximum l multipole to compute to")
parser.add_argument("--ngal", type=float, default=NGAL, help="galaxy number density per area (1 / arcmin^2)")
parser.add_argument("--mu", type=float, default=MU, help="mean of galaxy redshift sample")
parser.add_argument("--sigma", type=float, default=SIGMA, help="stdev of galaxy redshift sample")
parser.add_argument("--binwidth", type=int, default=BINWIDTH, help="width of bins for SNR calculation")
parser.add_argument("--bias", type=float, default=BIAS, help="galaxy bias")
parser.add_argument("--noisefile", type=str, default=NOISEFILE, help="filename of Nlkk data (col 1 = ells, col 2 = N_ells)")
parser.add_argument("--usefwhm", action="store_true", help="use fwhm instead of stdev as dndz scatter param")
parser.add_argument("--verbose", action="store_true", help="output debug / verbose text")

args = parser.parse_args()

if args.verbose:
    DEBUG = args.verbose

def dprint(s):
    if DEBUG: print(s)
    else: return

if args.fsky and not args.fskydeg2:
    dprint(f"Using fsky = {args.fsky}.")
    FSKY = args.fsky
if args.fskydeg2:
    dprint(f"Using fsky = {args.fskydeg2} deg^2.")
    FSKYDEG2, FSKY = args.fskydeg2, args.fskydeg2 / (4*np.pi*(180./np.pi)**2.)
if args.lmax:
    dprint(f"Using lmax = {args.lmax}.")
    LMAX = args.lmax
if args.ngal:
    dprint(f"Using n_gal = {args.ngal} arcmin^-2.")
    NGAL = args.ngal
if args.mu:
    dprint(f"Using mu = {args.mu}.")
    MU = args.mu
if args.sigma:
    dprint(f"Using sigma = {args.sigma}.")
    SIGMA = args.sigma
if args.binwidth:
    dprint(f"Using a bin width of {args.binwidth}.")
    BINWIDTH = args.binwidth
if args.noisefile:
    dprint(f"Using noise Nlkks from {args.noisefile}.")
    NOISEFILE = args.noisefile
if args.usefwhm:
    dprint("Using fwhm instead of stdev as dndz scatter.")
    USEFWHM = args.usefwhm

# using 0.2 as fwhm, could be sigma instead in which case
# you replace sig(0.2) -> 0.2

def calculate_snr():
    sig = lambda fwhm: fwhm / (2 * np.sqrt(2 * np.log(2)))
    #DNDZ = lambda z: np.exp((-(z - MU)**2.) / (2 * (sig(0.2) if USEFWHM else SIGMA)**2.))
    ARCMIN2_TO_STER = lambda amin2: (amin2 / 60.**2) / (180. / np.pi)**2

    pars = camb.CAMBparams()
    PARAMS = {'omch2': 0.1203058,
            'ombh2': 0.02219218,
            'H0': 67.02393,
            'ns': 0.9625356,
            'As': 2.15086031154146e-9,
            'mnu': 0.06,
            'w': -1.0,
            'tau':0.06574325,
            'nnu':3.046,
            'wa': 0.}

    pars = camb.set_params(**PARAMS)
    pars.set_for_lmax(LMAX, lens_potential_accuracy=1)
    pars.Want_CMB = False 

    pars.NonLinear = camb.model.NonLinear_both
    #Set up W(z) window functions, later labelled W1, W2. Gaussian here.
    pars.SourceWindows = [
        GaussianSourceWindow(redshift=MU, source_type='counts', bias=BIAS, sigma=(sig(SIGMA) if USEFWHM else SIGMA)),
        # unused
        GaussianSourceWindow(redshift=MU, source_type='lensing', sigma=SIGMA)]

    results = camb.get_results(pars)
    cls = results.get_source_cls_dict()

    clphiphi, clphig, dlgg = cls['PxP'], cls['W1xP'], cls['W1xW1']

    # 0.03 per arcmin^2 -> # per steradian
    shot_noise = ARCMIN2_TO_STER(1/NGAL)

    ls = np.arange(2, LMAX+1)
    bin_edges = np.arange(2, LMAX, BINWIDTH)

    D_ell_prefactor = (ls * (ls+1))**2 / (2 * np.pi)
    phi_to_kappa = ls * (ls+1) / 2
    clkk = clphiphi[2:LMAX+1] * phi_to_kappa**2 / D_ell_prefactor

    nlkk = np.loadtxt(NOISEFILE)[:LMAX-1,1]
    nlgg = clkk*0. + shot_noise
    # strip off the l(l+1) / 2pi factors and convert rest of phis to kappas
    clkg = clphig[2:LMAX+1] / np.sqrt(ls*(ls+1)) * np.pi
    clgg = dlgg[2:LMAX+1] / (ls*(ls+1)/(2*np.pi))

    # gaussian_band_covariance ONLY TAKES IN FUNCTIONS (interpolated)
    def intp(d, ells=ls): return interp1d(ells, d)

    cls_dict = {"kk": intp(clkk), "kg": intp(clkg), "gg": intp(clgg)}
    nls_dict = {"kk": intp(nlkk), "kg": intp(clkk*0.), "gg": intp(nlgg)}

    cov = 1/FSKY * pyfisher.gaussian_band_covariance(bin_edges, ["kg"],
                                                    cls_dict, nls_dict, interpolate=False)

    sanity = clkg / (np.sqrt(clgg * clkk))

    plt.loglog(ls, clkk, label=r"$C_L^{\kappa \kappa}$ (CAMB)")
    plt.loglog(ls, nlkk, label=r"$N_L^{\kappa \kappa}$")
    plt.loglog(ls, clgg, label=r"$C_L^{gg}$")
    plt.loglog(ls, clgg*0. + shot_noise, label=r"$N_L^{gg}$")
    plt.loglog(ls, clkg, label=r"$C_L^{\kappa g}$")

    plt.xlabel(r'$\ell$')
    plt.ylabel(r'$C_\ell$')
    plt.legend()
    plt.savefig("snr-cls-and-nls.png")

    for v in sanity:
        try: assert np.abs(v) <= 1.
        except AssertionError:
            print("Sanity check for C_l^(kg) failed -- cross spectrum values don't make sense!")
            return None

    _, clkg_bin = stats.bin1D(bin_edges).bin(ls, clkg)

    print("")
    snr = 0.
    for i in range(len(clkg_bin)):
        snr += clkg_bin[i]**2 / cov[i,0,0]
    return snr ** 0.5

if __name__ == '__main__':
    print(calculate_snr())