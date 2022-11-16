import numpy as np
import matplotlib.pyplot as plt

from orphics import maps, cosmology, io, pixcov, mpi, stats
from pixell import enmap, curvedsky as cs, lensing as plensing, enplot
import healpy as hp

#import websky_lensing_reconstruction as wlrecon
import time

LMAX = 3000

IKALM_FILE = "ikalm-websky.fits"
UNINPAINT_FILE = "uninpainted_recon_alms_websky_TT.fits" 
INPAINT_FILE = "inpainted_recon_alms_websky_TT.fits"

t1 = time.time()

ikalm = hp.fitsfunc.read_alm(IKALM_FILE)
irecon_alms = hp.fitsfunc.read_alm(INPAINT_FILE)
urecon_alms = hp.fitsfunc.read_alm(UNINPAINT_FILE)

icls = hp.alm2cl(ikalm,ikalm)
ells = np.arange(len(icls))
# use this if first bin object is doing linear bins
bin_edges2 = np.geomspace(2,LMAX,40)
binner2 = stats.bin1D(bin_edges2)
bin2 = lambda x: binner2.bin(ells,x)


pl = io.Plotter('CL', figsize=(16,12))
pl2 = io.Plotter('rCL',xyscale='loglin',figsize=(16,12))

inpcls = hp.alm2cl(irecon_alms, irecon_alms)
#finpcls = hp.alm2cl(norm_firecon_alms[est], norm_firecon_alms[est])
rcls = hp.alm2cl(urecon_alms, urecon_alms)

#rxinp_cls = hp.alm2cl(irecon_alms, urecon_alms)
rxi_cls = hp.alm2cl(urecon_alms, ikalm)
ixinp_cls = hp.alm2cl(irecon_alms, ikalm)
#ixinp2_cls = hp.alm2cl(norm_irecon2_alms[est], ikalm)
#ixfinp_cls = hp.alm2cl(norm_firecon_alms[est], ikalm)

#pl.add(ells,(ells*(ells+1.)/2.)**2. * Al[est][0],ls='--', label="noise PS (per mode)")
#pl.add(ells,rcls,label='recon x recon')
pl.add(ells,ixinp_cls,label='inpaint x input')
#pl.add(ells,ixfinp_cls,label='null inpaint x input')
pl.add(ells,rxi_cls,label='non-inpaint x input')
pl.add(ells,icls,label='input x input')

pl2.add(*bin2((ixinp_cls-icls)/icls),marker='o',label="inpaint (ivar = 10 uK-arcmin) x input Clkk")
#pl2.add(*bin2((ixinp2_cls-icls)/icls),marker='o',label="inpaint (ivar) x input Clkk")
pl2.add(*bin2((rxi_cls-icls)/icls),marker='o',label="non-inpaint x input Clkk")
#pl2.add(*bin2((ixfinp_cls-icls)/icls),marker='o',label="null inpaint x input Clkk")
pl2.add(*bin2((ixinp_cls-rxi_cls)/rxi_cls), marker='o', label="inp. Clkk vs non-inp. Clkk")
#pl2.add(*bin2((inpainted_cls-uninpainted_cls)/uninpainted_cls), marker='o', label="inp. ClTT vs non-inp. ClTT")
pl2._ax.set_xlabel(r'$L$', fontsize=20)
pl2._ax.legend(fontsize=30)
#pl2._ax.set_ylabel(r'$(C_L^{\hat\kappa \kappa_{i}} - C_L^{\kappa_{i} \kappa_{i}}) /  C_L^{\kappa_{i} \kappa_{i}}$', fontsize=16)
#pl2._ax.legend()
pl2.hline(y=0)
pl2._ax.set_ylim(-0.3, 0.3)

"""
pl3.add(*bin((rcls - inpcls)/inpcls),marker='o')
pl3._ax.set_ylabel(r'$(C_L^{\hat\kappa_{r} \hat\kappa_{r}} - C_L^{\hat\kappa_{inp} \hat\kappa_{inp}}) /  C_L^{\hat\kappa_{inp} \hat\kappa_{inp}}$', fontsize=16)
pl3.hline(y=0)

pl4.add(*bin((ixinp_cls-rxi_cls)/rxi_cls),marker='o')
pl4._ax.set_ylabel(r'$(C_L^{\hat\kappa_{inp} \kappa_{i}} - C_L^{\hat\kappa_{r} \kappa_{i}}) /  C_L^{\hat\kappa_{r} \kappa_{i}}$', fontsize=16)
pl4.hline(y=0)
pl4._ax.set_ylim(-0.5,0.5)
"""

pl.done(f'ps_websky_null_ivar_TT.png')
pl2.done(f'ps_websky_cross_vs_auto_diff_ivar_TT.png')
#pl3.done(f'ps_recon_vs_inp_auto_diff_{est}.png')
#pl4.done(f'ps_recon_vs_inp_cross_diff_{est}.png')
#pl._ax.set_ylim(1e-9,1e-5)

print("Time elapsed: %0.5f seconds" % (time.time() - t1))
