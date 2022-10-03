from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
from orphics import maps,io,cosmology,stats
from pixell import enmap,curvedsky as cs,lensing
from enlib import bench
import numpy as np
import os,sys
import healpy as hp
from falafel import qe,utils as futils

import json,pickle

import pytempura
import timings

NBINS = 30

"""
Verify full-sky lensing with flat-sky norm. Going to try to adapt for temperatures only for now.
No MPI
"""
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

import argparse
# Parse command line
parser = argparse.ArgumentParser(description='Do a thing.')
parser.add_argument("--res",     type=float,  default=1.5,help="Resolution in arcminutes.")
parser.add_argument("--lmaxt",     type=int,  default=6000,help="help")
parser.add_argument("--lmaxp",     type=int,  default=5000,help="help")
parser.add_argument("--lmint",     type=int,  default=300,help="help")
parser.add_argument("--lminp",     type=int,  default=100,help="help")
parser.add_argument("--beam",     type=float,  default=1.4,help="Beam in arcminutes.")
parser.add_argument("--noise",     type=float,  default=10.,help="Noise in muK-arcmin.")
parser.add_argument("--dtype",     type=int,  default=64,help="dtype bits")
parser.add_argument("-a", "--all", action='store_true',help='Do all.')
args = parser.parse_args()
dtype = np.complex128 if args.dtype==64 else np.complex64

if args.all: print("Doing all estimators")

# own timer function
T = timings.Timings()
T.start()

# Resolution
res = args.res
shape,iwcs = enmap.fullsky_geometry(res=np.deg2rad(res/60.),proj="car")
# why do we do this? changes car:cdelt from [-0.025,0.025] to [0.025,0.025]
wcs = enmap.zeros(shape,iwcs)[...,::-1].wcs
mlmax = max(args.lmaxt,args.lmaxp) + 2000
px = qe.pixelization(shape=shape, wcs=wcs)

# Data location
DIR = "/global/homes/j/jaejoonk/masked-cmb-lensing/websky/"
#DIR = "/home/joshua/research/cmb_lensing_2022/masked-cmb-lensing/"
LENSED_CMB_ALMS_LOC = DIR + "lensed_alm.fits"
KAP_LOC = DIR + "kap.fits"

# Beam and noise
ells = np.arange(mlmax)
lbeam = maps.gauss_beam(ells,args.beam)
ps_noise = np.zeros((3,3,ells.size))
ps_noise[0,0] = (args.noise*np.pi/180./60.)**2.
ps_noise[1,1] = (args.noise*np.pi/180./60.)**2. * 2.
ps_noise[2,2] = (args.noise*np.pi/180./60.)**2. * 2.

s = stats.Stats()


# Reduction
#malm2cl = lambda x,y: hp.alm2cl(x,y)
def reduce(name,irecon,ialm):
    [c_irecon, g_irecon], [c_als, g_als] = irecon.astype(dtype), Als[name.upper()]
    crecon = cs.almxfl(c_irecon, c_als)
    grecon = cs.almxfl(g_irecon, g_als)
    s.add_to_stats(name+"_gauto",hp.alm2cl(grecon,grecon))
    s.add_to_stats(name+"_cauto",hp.alm2cl(crecon,crecon))
    s.add_to_stats(name+"_gcross",hp.alm2cl(grecon,ialm))
    s.add_to_stats(name+"_ccross",hp.alm2cl(crecon,ialm))

T.add("data, map, beam+noise setup")

# Theory
ucls, tcls = futils.get_theory_dicts(lmax=mlmax)

# Norms
ESTIMATORS = ['TT','TE','EE','EB','TB','MV','MVPOL']

Als = pytempura.get_norms(ESTIMATORS, ucls, tcls, args.lmint, args.lmaxt)
ls = np.arange(len(Als[ESTIMATORS[0]][0]))

# dumping
#labels = {est: Als[est] for est in Als.keys()}
#np.savez("Als", **labels)
T.add("get theory dicts + norms")

# Read alms
alm = maps.change_alm_lmax(hp.read_alm(LENSED_CMB_ALMS_LOC), mlmax).astype(dtype)

# Add beam deconvolved noise
alm = alm + np.nan_to_num(qe.almxfl(cs.rand_alm_healpy(ps_noise, lmax=mlmax, seed=(100,200,0), dtype=dtype),1./lbeam))
ntt = maps.interp(ells,np.nan_to_num(ps_noise[0,0]/lbeam**2.))
npp = maps.interp(ells,np.nan_to_num(ps_noise[1,1]/lbeam**2.))

# Filter
talm_y = qe.filter_alms(alm[0],1./(ucls['TT'] + ntt(np.arange(len(ucls['TT'])))),args.lmint,args.lmaxt)
ealm_y = qe.filter_alms(alm[1],1./(ucls['EE'] + npp(np.arange(len(ucls['EE'])))),args.lminp,args.lmaxp)
balm_y = qe.filter_alms(alm[2],1./(ucls['BB'] + npp(np.arange(len(ucls['BB'])))),args.lminp,args.lmaxp)

# Inputs
ikalm = maps.change_alm_lmax(hp.map2alm(hp.read_map(KAP_LOC), lmax=args.lmaxt),mlmax)
T.add("read alms, filter + add noise")

if args.all:
    res = qe.qe_all(px,ucls,mlmax,talm_y,ealm_y,balm_y,
                    estimators=ESTIMATORS)
    for key in res.keys():
        reduce(key,res[key],ikalm)

else:
    talm_x = qe.filter_alms(talm_y,ucls['TT'],args.lmint,args.lmaxt) + \
             qe.filter_alms(ealm_y,ucls['TE'],args.lmint,args.lmaxt)
    ealm_x = qe.filter_alms(ealm_y,ucls['EE'],args.lmint,args.lmaxt) + \
             qe.filter_alms(ealm_y,ucls['TE'],args.lmint,args.lmaxt)
    balm_x = qe.filter_alms(balm_y,ucls['BB'],args.lminp,args.lmaxp)
    t0alm_x = qe.filter_alms(talm_y,ucls['TT'],args.lmint,args.lmaxt)
    e0alm_x = qe.filter_alms(ealm_y,ucls['EE'],args.lmint,args.lmaxt)
    del alm

    # Recons
    recon_mv,dmap_t,dmap_p = qe.qe_mv(px,talm_x,ealm_x,balm_x,talm_y,ealm_y,balm_y,mlmax)
    reduce("mv",recon_mv,ikalm)
    del recon_mv

    recon_tonly = qe.qe_temperature_only(px,t0alm_x,talm_y,mlmax)
    reduce("TT",recon_tonly,ikalm)
    del recon_tonly,talm_x,talm_y,dmap_t

    recon_ponly = qe.qe_pol_only(px,e0alm_x,balm_x,ealm_y,balm_y,mlmax)
    reduce("mvpol",recon_ponly,ikalm)
    del recon_ponly, balm_x, balm_y,dmap_p

    recon_eonly = qe.qe_pol_only(px,e0alm_x,e0alm_x*0,ealm_y,ealm_y*0,mlmax)
    reduce("EE",recon_eonly,ikalm)
    del recon_eonly, ealm_x, ealm_y

s.add_to_stats("iauto",hp.alm2cl(ikalm,ikalm))
del ikalm

T.add("do recon for all ests")

# Get data
s.get_stats()

icls = s.stats['iauto']['mean']
ells = np.arange(len(icls))
combs = ['TT','mvpol','EE','mv'] if not(args.all) else ESTIMATORS
ostats = ['gauto','cauto','gcross','ccross']
cls = {}
for comb in combs:
    cls[comb] = {}
    for stat in ostats:
        cls[comb][stat] = {}
        cls[comb][stat]['mean'] = s.stats[comb+"_"+stat]['mean']
        cls[comb][stat]['err'] = s.stats[comb+"_"+stat]['errmean']

T.add("get stats for all combinations")
#print("all data processed")

# dump ucls
#np.savetxt("ucls-kk.txt", ucls['kk'])
    
# dump iauto-mean
#np.savetxt("iauto-mean-icls.txt", icls)

# bin
bin_edges = np.geomspace(2, mlmax, NBINS)
binner = stats.bin1D(bin_edges)
bin_fn = lambda x: binner.bin(ells, x)

# Make plots
for comb in combs:
    # dump cls[comb]
    #for stat in ostats:
    #    np.savez("cls-"+comb+"-"+stat, mean=cls[comb][stat]['mean'],
    #                                   err=cls[comb][stat]['err'])

    # grad
    pl = io.Plotter(xyscale='loglog',xlabel='$L$',ylabel='$C_L$')
    fells = np.arange(2,mlmax)
    pl.add(fells,ucls['kk'][fells],lw=3, label="theory")
    pl.add(ells,icls,color='k',alpha=0.5, label="input auto PS")
    pl.add(ls,Als[comb][0]*ls*(ls+1)/4.,ls="--", label="noise PS (per mode)")
        #if comb=='TE': pl.add(ls,Al_te_alt*ls*(ls+1)/4.,ls="-.")
    pl.add(ls,maps.interp(ells,icls)(ls) + (Als[comb][0]*ls*(ls+1)/4.), label="noise + auto PS")
    pl.add(ells,cls[comb]['gauto']['mean'], label="auto mean")
    # bin cross mean?
    pl.add(ells,np.abs(cls[comb]['gcross']['mean']), label="cross mean")
    # pl._ax.set_ylim(1e-9,4e-6)
    pl._ax.legend()
    pl.done(DIR+'verify_abs_grad_%s.png' % comb)

    # grad diff
    pl = io.Plotter(xyscale='loglin',xlabel='$L$',ylabel='$\\Delta C_L / C_L$')
    pl.add(*bin_fn((cls[comb]['gcross']['mean']-icls)/icls), label="grad diff")
    pl.hline()
    # pl._ax.set_ylim(-0.2,0.4)
    pl.done(DIR+'verify_grad_diff_%s.png' % comb)

    pl = io.Plotter(xyscale='linlin',xlabel='$L$',ylabel='$\\Delta C_L / C_L$')
    pl.add(*bin_fn((cls[comb]['gcross']['mean']-icls)/icls), label="grad diff")
    pl.hline()
    # pl._ax.set_ylim(-0.2,0.4)
    pl.done(DIR+'verify_grad_diff_lin_%s.png' % comb)
        
    # curl
    pl = io.Plotter(xyscale='loglin',xlabel='$L$',ylabel='$C_L$')
    pl.add(ells,cls[comb]['ccross']['mean'], label="cross mean")
    pl.hline()
    pl._ax.legend()
    pl.done(DIR+'verify_curl_%s.png' % comb)

T.add("plot all")
T.list()
