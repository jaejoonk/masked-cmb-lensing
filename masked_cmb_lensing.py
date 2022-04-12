from __future__ import print_function

from pixell import enmap,utils,reproject,enplot
from pixell.reproject import healpix2map,thumbnails
from pixell.curvedsky import alm2map

import numpy as np
import matplotlib.pyplot as plt

import healpy as hp
from healpy.fitsfunc import read_alm,read_map

KAP_FILENAME = "kap.fits"
KSZ_FILENAME = "ksz.fits"
ALM_FILENAME = "lensed_alm.fits"
RESOLUTION = np.deg2rad(1.0 / 60.)
OMEGAM_H2 = 0.1428 # planck 2018 vi paper
RHO = 2.775e11 * OMEGAM_H2
MASS_CUTOFF = 1.0 # 1e14 solar masses

# defaults to 0.5 arcmin resolution for output geometry
def alm_to_car(filename, res=RESOLUTION):
    try:
        alm_hp = read_alm(filename)
    except:
        return None

    shape, wcs = enmap.fullsky_geometry(res=res)
    # empty CAR map
    omap = enmap.empty(shape, wcs, dtype=np.float32)
    
    return alm2map(alm_hp, omap)

# defaults to 0.5 arcmin resolution for output geometry
def px_to_car(filename, res=RESOLUTION):
    try:
        map_hp = read_map(filename)
    except:
        return None

    shape, wcs = enmap.fullsky_geometry(res=res)
    return healpix2map(map_hp.astype(np.float32), shape=shape, wcs=wcs)

# plot a centered submap with a width in pixels 
# COLOR_EXTREME changes color extremum for the colorbar
def plot_map(imap, plotname="submap", COLOR_EXTREME = 1.0, width = 1000):
    (ydim, xdim) = imap.shape
    yc, xc = ydim // 2, xdim // 2
    # based on aspect ratio of map
    height = int(width * ydim / xdim)

    top, bottom = yc - height//2, yc + height//2
    left, right = xc - width//2, xc + width//2

    imap_sub = imap[top:bottom, left:right]

    plt.figure(figsize = (20, 10))
    plt.title("Submap of input map (c = (%d,%d), (w,h) = (%d,%d))" \
            % (xc, yc, width, height))
    plt.imshow(imap_sub, vmin=COLOR_EXTREME * imap_sub.min(),
                         vmax=COLOR_EXTREME * imap_sub.max())
    plt.colorbar()  
    plt.savefig(plotname + ".png")

    enplot_map = enplot.plot(imap_sub,
                 range=0.4 * COLOR_EXTREME * (imap_sub.max() - imap_sub.min()))
    enplot.write(plotname + "-enplot", enplot_map)

# input a halo catalog .pksc file and output ra, dec in radians
def catalog_to_coords(filename = "halos_10x10.pksc", mass_cutoff = MASS_CUTOFF):
    f = open(filename)

    # number of halos
    Nhalo = np.fromfile(f, count=3, dtype=np.int32)[0]

    # halo data (10 cols):
    # x, y, z [Mpc], vx, vy, vz [km/s], M [M_sun], x_lag, y_lag, z_lag
    data = np.fromfile(f, dtype=np.float32)

    # reshape into 2d array
    data_table = np.reshape(data, (Nhalo, 10))

    # fetch data from columns
    x, y, z = data_table[:, 0], data_table[:, 1], data_table[:, 2]
    R = data_table[:, 6]

    # for mass cutoff
    mass = 4*np.pi/3.*RHO*R**3 / 1e14

    # convert to ra / dec (radians?) from readhalos.py
    colat, ra = hp.vec2ang(np.column_stack((x, y, z)))

    # convert colat to dec 
    dec = np.pi/2 - colat

    f.close()

    # truncate to mass cutoff
    return ra[mass >= mass_cutoff], dec[mass >= mass_cutoff]

# stack and average on a random subset of coordinates
# output stack, average maps
def stack_average_random(imap, ra, dec, Ncoords=1000,
                         radius=np.deg2rad(2.), res=RESOLUTION):
    idx_random = np.random.choice(len(ra), Ncoords, replace=False)
    coords = np.array([[dec[i], ra[i]] for i in idx_random])

    # create thumbnails
    thumbs = thumbnails(imap, coords, r = np.deg2rad(2.), res=res)

    # stack
    stack_map = 0
    for i in range(len(thumbs)):
        stack_map += thumbs[i]

    # average
    avg_map = stack_map / Ncoords

    return stack_map, avg_map

# stack and average on the first N coordinates
# output stack, average maps
def stack_average_firstn(imap, ra, dec, Ncoords=1000,
                         radius=np.deg2rad(2.), res=RESOLUTION):
    coords = np.array([[dec[i], ra[i]] for i in range(Ncoords)])

    # create thumbnails
    thumbs = thumbnails(imap, coords, r = np.deg2rad(2.), res=res)

    # stack
    stack_map = 0
    for i in range(len(thumbs)):
        stack_map += thumbs[i]

    # average
    avg_map = stack_map / Ncoords

    return stack_map, avg_map

# stack and average 2 maps on the same random subset of coordinates
# output stack, average maps
def stack_average_2map_random(imap1, imap2, ra, dec, Ncoords=1000,
                              radius=np.deg2rad(2.), res=RESOLUTION):
    idx_random = np.random.choice(len(ra), Ncoords, replace=False)
    coords = np.array([[dec[i], ra[i]] for i in idx_random])

    # create thumbnails
    thumbs1 = thumbnails(imap1, coords, r = np.deg2rad(2.), res=res)
    thumbs2 = thumbnails(imap2, coords, r = np.deg2rad(2.), res=res)

    # stack
    stack1_map = 0
    stack2_map = 0
    for i in range(len(thumbs1)):
        stack1_map += thumbs1[i]
        stack2_map += thumbs2[i]

    # average
    avg1_map = stack1_map / Ncoords
    avg2_map = stack2_map / Ncoords

    return stack1_map, avg1_map, stack2_map, avg2_map

# plot relevant values
def plot_all(stack_ksz, avg_ksz, stack_kap, avg_kap, filename="all-random"):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(18,18))

    im1 = axes[0,0].imshow(stack_ksz, cmap='jet')
    axes[0,0].set_title("Stacked kSZ signal", fontsize=18)
    im2 = axes[0,1].imshow(avg_ksz, cmap='jet')
    axes[0,1].set_title("Averaged kSZ signal", fontsize=18)
    im3 = axes[1,0].imshow(stack_kap, cmap='jet')
    axes[1,0].set_title("Stacked kappa signal", fontsize=18)
    im4 = axes[1,1].imshow(avg_kap, cmap='jet')
    axes[1,1].set_title("Averaged kappa signal", fontsize=18)

    fig.subplots_adjust(right=0.85)
    fig.colorbar(im1, ax = axes[0,0], fraction=0.046, pad=0.04)
    fig.colorbar(im2, ax = axes[0,1], fraction=0.046, pad=0.04)
    fig.colorbar(im3, ax = axes[1,0], fraction=0.046, pad=0.04)
    fig.colorbar(im4, ax = axes[1,1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(filename + ".png")
    
# runs everything and spits out 
def run_routine(output_filename = "all-random"):
    kap_px = px_to_car(KAP_FILENAME, res=RESOLUTION)
    ksz_px = px_to_car(KSZ_FILENAME, res=RESOLUTION)
    alm_px = alm_to_car(ALM_FILENAME, res=RESOLUTION)

    plot_map(kap_px, plotname="submap-kap")
    plot_map(ksz_px, plotname="submap-ksz")
    plot_map(alm_px, plotname="submap-alm")

    ra, dec = catalog_to_coords()

    stack_ksz,avg_ksz,stack_kap,avg_kap = stack_average_2map_random(ksz_px, kap_px, ra, dec)

    plot_all(stack_ksz, avg_ksz, stack_kap, avg_kap, filename=output_filename)
    



