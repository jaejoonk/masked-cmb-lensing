from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
from orphics import maps,io,cosmology,stats,mpi
from pixell import enmap,curvedsky as cs,lensing
from enlib import bench
import numpy as np
import os,sys
import healpy as hp
from falafel import qe,utils as futils

import pytempura
"""
Verify full-sky lensing with flat-sky norm. Going to try to adapt for temperatures only for now.
"""

import argparse
# Parse command line
parser = argparse.ArgumentParser(description='Do a thing.')
parser.add_argument("--nsims",     type=int,  default=5,help="Number of sims. If not specified, runs on data.")
parser.add_argument("--res",     type=float,  default=1.5,help="Resolution in arcminutes.")
parser.add_argument("--lmaxt",     type=int,  default=3000,help="help")
parser.add_argument("--lmaxp",     type=int,  default=5000,help="help")
parser.add_argument("--lmint",     type=int,  default=100,help="help")
parser.add_argument("--lminp",     type=int,  default=100,help="help")
parser.add_argument("--beam",     type=float,  default=1.4,help="Beam in arcminutes.")
parser.add_argument("--noise",     type=float,  default=10.,help="Noise in muK-arcmin.")
parser.add_argument("--sim-location",     type=str,  default=None,help="Path to sims.")
parser.add_argument("--norm-file",     type=str,  default="norm.txt",help="Norm file.")
parser.add_argument("--dtype",     type=int,  default=64,help="dtype bits")
parser.add_argument("-a", "--all", action='store_true',help='Do all.')
args = parser.parse_args()
dtype = np.complex128 if args.dtype==64 else np.complex64

# Resolution
res = args.res
shape,iwcs = enmap.fullsky_geometry(res=np.deg2rad(res/60.),proj="car")
wcs = enmap.zeros(shape,iwcs)[...,::-1].wcs
mlmax = max(args.lmaxt,args.lmaxp) + 2000
px = qe.pixelization(shape=shape, wcs=wcs)

# Sim location
DIR = "/home/joshua/research/cmb_lensing_2022/masked-cmb-lensing/"
LENSED_CMB_ALMS_LOC = DIR + "lensed_alm.fits"
KAP_LOC = DIR + "kap.fits"

"""try:
    if args.sim_location is not None: raise
    from soapack import interfaces as sints
    sim_location = sints.dconfig['actsims']['signal_path']
except:
    sim_location = args.sim_location
    assert sim_location is not None"""

# Beam and noise
ells = np.arange(mlmax)
lbeam = maps.gauss_beam(ells,args.beam)
ps_noise = np.zeros((3,3,ells.size))
ps_noise[0,0] = (args.noise*np.pi/180./60.)**2.
ps_noise[1,1] = (args.noise*np.pi/180./60.)**2. * 2.
ps_noise[2,2] = (args.noise*np.pi/180./60.)**2. * 2.

# MPI
comm = mpi.MPI.COMM_WORLD
rank = comm.Get_rank()
numcores = comm.Get_size()
Nsims = args.nsims
num_each,each_tasks = mpi.mpi_distribute(Nsims,numcores)
if rank==0: print ("At most ", max(num_each) , " tasks...")
my_tasks = each_tasks[rank]
s = stats.Stats(comm)

# Reduction
malm2cl = lambda x,y: hp.alm2cl(x,y)
def reduce(name,irecon,ialm):
    [c_irecon, g_irecon], [c_als, g_als] = irecon, Als[name.upper()]
    crecon = cs.almxfl(c_irecon, c_als)
    grecon = cs.almxfl(g_irecon, g_als)
    s.add_to_stats(name+"_gauto",malm2cl(grecon,grecon))
    s.add_to_stats(name+"_cauto",malm2cl(crecon,crecon))
    s.add_to_stats(name+"_gcross",malm2cl(grecon,ialm))
    s.add_to_stats(name+"_ccross",malm2cl(crecon,ialm))
    #orecon = qe.almxfl(irecon,Als[name.upper()])
    #s.add_to_stats(name+"_gauto",malm2cl(orecon[0],orecon[0]))
    #s.add_to_stats(name+"_cauto",malm2cl(orecon[1],orecon[1]))
    #s.add_to_stats(name+"_gcross",malm2cl(orecon[0],ialm))
    #s.add_to_stats(name+"_ccross",malm2cl(orecon[1],ialm))

# Theory
#thloc = sim_location + "cosmo2017_10K_acc3"
#theory = cosmology.loadTheorySpectraFromCAMB(thloc,get_dimensionless=False)
ucls, tcls = futils.get_theory_dicts(lmax=mlmax)

# Norm
Als = pytempura.get_norms(['TT', 'EE', 'EB', 'TB', 'MVPOL', 'MV', 'TE'], ucls, tcls, args.lmint, args.lminp)
ls = np.arange(len(Als['TT']))
#ls,Als['TT'],Als['EE'],Als['EB'],Al_te_alt,Als['TB'],Als['mvpol'],Als['mv'],Als['TE'] = np.loadtxt(args.norm_file,unpack=True)
    
for task in my_tasks:

    # Load sims
    #sindex = str(task).zfill(5)
    #alm = maps.change_alm_lmax(hp.read_alm(sim_location+"fullskyLensedUnabberatedCMB_alm_set00_%s.fits" % sindex,hdu=(1,2,3)),mlmax).astype(dtype)
    alm = maps.change_alm_lmax(hp.read_alm(LENSED_CMB_ALMS_LOC), mlmax).astype(dtype)

    # Add beam deconvolved noise
    alm = alm + np.nan_to_num(qe.almxfl(cs.rand_alm_healpy(ps_noise, lmax=mlmax, seed=(100,200,task), dtype=dtype),1./lbeam))
    ntt = maps.interp(ells,np.nan_to_num(ps_noise[0,0]/lbeam**2.))
    npp = maps.interp(ells,np.nan_to_num(ps_noise[1,1]/lbeam**2.))

    # Filter
    talm_y = qe.filter_alms(alm[0],1./tcls['TT'],args.lmint,args.lmaxt)
    ealm_y = qe.filter_alms(alm[1],1./tcls['EE'],args.lminp,args.lmaxp)
    balm_y = qe.filter_alms(alm[2],1./tcls['BB'],args.lminp,args.lmaxp)
    # talm_y = qe.filter_alms(alm[0],lambda x: 1./(ucls['TT'][x]+ntt(x)),args.lmint,args.lmaxt)
    # ealm_y = qe.filter_alms(alm[1],lambda x: 1./(ucls['EE'][x]+npp(x)),args.lminp,args.lmaxp)
    # balm_y = qe.filter_alms(alm[2],lambda x: 1./(ucls['BB'][x]+npp(x)),args.lminp,args.lmaxp)
    #talm_y = futils.isotropic_filter(alm, tcls, args.lmint, args.lmaxt)[0]
    #[ealm_y, balm_y] = futils.isotropic_filter(alm, tcls, args.lminp, args.lmaxp)[1:]

    # Inputs
    ikalm = maps.change_alm_lmax(hp.map2alm(hp.read_map(KAP_LOC), lmax=args.lmaxt),mlmax)

    if args.all:

        with bench.show("qe all"):
            res = qe.qe_all(px,ucls,mlmax,talm_y,ealm_y,balm_y,
                            estimators=['TT','TE','EE','EB','TB','mv','mvpol'])
        for key in res.keys():
            reduce(key,res[key],ikalm)

    else:
        """
        talm_x = qe.filter_alms(talm_y,lambda x: ucls['TT'][x],args.lmint,args.lmaxt) + \
                 qe.filter_alms(ealm_y,lambda x: ucls['TE'][x],args.lmint,args.lmaxt)
        ealm_x = qe.filter_alms(ealm_y,lambda x: ucls['EE'][x],args.lmint,args.lmaxt) + \
                 qe.filter_alms(talm_y,lambda x: ucls['TE'][x],args.lmint,args.lmaxt)
        balm_x = qe.filter_alms(balm_y,lambda x: ucls['BB'][x],args.lminp,args.lmaxp)
        t0alm_x = qe.filter_alms(talm_y,lambda x: ucls['TT'][x],args.lmint,args.lmaxt) # pure T
        e0alm_x = qe.filter_alms(ealm_y,lambda x: ucls['EE'][x],args.lmint,args.lmaxt) # pure E
        """
        talm_x = qe.filter_alms(talm_y,ucls['TT'],args.lmint,args.lmaxt) + \
                 qe.filter_alms(ealm_y,ucls['TE'],args.lmint,args.lmaxt)
        ealm_x = qe.filter_alms(ealm_y,ucls['EE'],args.lmint,args.lmaxt) + \
                 qe.filter_alms(ealm_y,ucls['TE'],args.lmint,args.lmaxt)
        balm_x = qe.filter_alms(balm_y,ucls['BB'],args.lminp,args.lmaxp)
        t0alm_x = qe.filter_alms(talm_y,ucls['TT'],args.lmint,args.lmaxt)
        e0alm_x = qe.filter_alms(ealm_y,ucls['EE'],args.lmint,args.lmaxt)
        del alm

        # Recons

        with bench.show("recon mv"):
            recon_mv,dmap_t,dmap_p = qe.qe_mv(px,talm_x,ealm_x,balm_x,talm_y,ealm_y,balm_y,mlmax)

        reduce("mv",recon_mv,ikalm)
        del recon_mv

        with bench.show("recon tt"):
            recon_tonly = qe.qe_temperature_only(px,t0alm_x,talm_y,mlmax)

        reduce("TT",recon_tonly,ikalm)
        del recon_tonly,talm_x,talm_y,dmap_t

        with bench.show("recon pol"):
            recon_ponly = qe.qe_pol_only(px,e0alm_x,balm_x,ealm_y,balm_y,mlmax)

        reduce("mvpol",recon_ponly,ikalm)
        del recon_ponly, balm_x, balm_y,dmap_p

        with bench.show("recon eonly"):
            recon_eonly = qe.qe_pol_only(px,e0alm_x,e0alm_x*0,ealm_y,ealm_y*0,mlmax)

        reduce("EE",recon_eonly,ikalm)
        del recon_eonly, ealm_x, ealm_y
        
    s.add_to_stats("iauto",malm2cl(ikalm,ikalm))
    del ikalm

    if rank==0: print ("Rank 0 done with task ", task+1, " / " , len(my_tasks))
    
s.get_stats()
    

        
if rank==0:

    # Get data
    icls = s.stats['iauto']['mean']
    ells = np.arange(len(icls))
    combs = ['TT','mvpol','EE','mv'] if not(args.all) else ['TT','TE','EE','EB','TB','mv','mvpol']
    ostats = ['gauto','cauto','gcross','ccross']
    cls = {}
    for comb in combs:
        cls[comb] = {}
        for stat in ostats:
            cls[comb][stat] = {}
            cls[comb][stat]['mean'] = s.stats[comb+"_"+stat]['mean']
            cls[comb][stat]['err'] = s.stats[comb+"_"+stat]['errmean']
        
    
    # Make plots
    for comb in combs:
        # grad
        pl = io.Plotter(xyscale='loglog',xlabel='$L$',ylabel='$C_L$')
        fells = np.arange(2,mlmax)
        pl.add(fells,ucls['kk'][fells],lw=3)
        pl.add(ells,icls,color='k',alpha=0.5)
        pl.add(ls,Als[comb]*ls*(ls+1)/4.,ls="--")
        #if comb=='TE': pl.add(ls,Al_te_alt*ls*(ls+1)/4.,ls="-.")
        pl.add(ls,maps.interp(ells,icls)(ls) + (Als[comb]*ls*(ls+1)/4.))
        pl.add_err(ells,cls[comb]['gauto']['mean'],yerr=cls[comb]['gauto']['err'])
        pl.add_err(ells,cls[comb]['gcross']['mean'],yerr=cls[comb]['gauto']['err'])
        pl._ax.set_ylim(1e-9,4e-6)
        pl.done(os.environ['WORK']+'/verify_grad_%s.png' % comb)

        # grad diff
        pl = io.Plotter(xyscale='loglin',xlabel='$L$',ylabel='$\\Delta C_L / C_L$')
        pl.add_err(ells,(cls[comb]['gcross']['mean']-icls)/icls,yerr=cls[comb]['gcross']['err']/icls)
        pl.hline()
        pl._ax.set_ylim(-0.2,0.4)
        pl.done(os.environ['WORK']+'/verify_grad_diff_%s.png' % comb)

        pl = io.Plotter(xyscale='linlin',xlabel='$L$',ylabel='$\\Delta C_L / C_L$')
        pl.add_err(ells,(cls[comb]['gcross']['mean']-icls)/icls,yerr=cls[comb]['gcross']['err']/icls)
        pl.hline()
        pl._ax.set_ylim(-0.2,0.4)
        pl.done(os.environ['WORK']+'/verify_grad_diff_lin_%s.png' % comb)
        
        # curl
        pl = io.Plotter(xyscale='loglin',xlabel='$L$',ylabel='$C_L$')
        pl.add_err(ells,cls[comb]['ccross']['mean'],yerr=cls[comb]['cauto']['err'])
        pl.hline()
        pl.done(os.environ['WORK']+'/verify_curl_%s.png' % comb)
        


    
