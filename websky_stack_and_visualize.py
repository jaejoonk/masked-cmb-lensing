from __future__ import print_function

from pixell import enmap,utils,reproject,enplot
from pixell.reproject import healpix2map,thumbnails
from pixell.curvedsky import alm2map

import numpy as np
from cycler import cycler
from matplotlib import (cm, colors as mplcolors, pyplot as plt,
                        rcParams)

import healpy as hp
from healpy.fitsfunc import read_alm,read_map

# import multiprocessing as mp

from mpi4py import MPI
import requests, os

WEBSKY_SITE = "https://mocks.cita.utoronto.ca/data/websky/v0.0/"
KAP_FILENAME = "kap.fits"
KSZ_FILENAME = "ksz.fits"
ALM_FILENAME = "lensed_alm.fits"
NCOORDS = 1000

RESOLUTION = np.deg2rad(1.5 / 60.)
RAD = np.deg2rad(0.5)
OMEGAM_H2 = 0.1428 # planck 2018 vi paper
RHO = 2.775e11 * OMEGAM_H2
MASS_CUTOFF = 1.0 # 1e14 solar masses

COMM = MPI.COMM_WORLD

# fetch websky data
def fetch_data(data = ["kap", "ksz", "alm"]):
    def download(fname):
        if not os.path.isfile(fname):
            r = requests.get(WEBSKY_SITE + fname, allow_redirects=True)
            open(fname, 'wb').write(r.content)

    for d in data:
        if d == "kap":
            download(KAP_FILENAME)
        elif d == "ksz":
            download(KSZ_FILENAME)
        elif d == "alm":
            download(ALM_FILENAME)
        else: continue

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
def catalog_to_coords(filename = "halos_10x10.pksc", mass_cutoff = MASS_CUTOFF,
                      output_to_file = False, output_file = "output_halos.txt",
                      Nhalos=None):
    f = open(filename)

    # number of halos from binary file,
    Nhalo = np.fromfile(f, count=3, dtype=np.int32)[0]
    if Nhalos != None: Nhalo = Nhalos

    # halo data (10 cols):
    # x, y, z [Mpc], vx, vy, vz [km/s], M [M_sun], x_lag, y_lag, z_lag
    data = np.fromfile(f, count=Nhalo * 10, dtype=np.float32)

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
    ra_cutoff = ra[mass >= mass_cutoff]
    dec_cutoff = dec[mass >= mass_cutoff]
    masses = mass[mass >= mass_cutoff] * 1e14
    if not output_to_file: return ra_cutoff, dec_cutoff
    else: np.savetxt(output_file,
                     np.array(list(zip(ra_cutoff, dec_cutoff, masses))),
                     delimiter=',')

# return ra, dec from a file
def read_coords_from_file(input_filename, lowlim=None, highlim=None):
    data = np.loadtxt(input_filename, delimiter=",")
    if lowlim is not None:
        data = data[data[:,2] >= (lowlim * 1e14)]
    if highlim is not None:
        data = data[data[:,2] <= (highlim * 1e14)]
    return data[:, 0], data[:, 1]

# stack and average on a random subset of coordinates
# output stack, average maps
# default parallelized
def thumbnails_kw(i, c, radius, res):
        return thumbnails(i, c, r=radius, res=res)
    

def stack_average_random_mpi(imap, ra, dec, Ncoords=NCOORDS,
                         radius=RAD, res=RESOLUTION):
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()

    if rank == 0:
        idx_random = np.random.choice(len(ra), Ncoords, replace=False)
        coords = np.array([[dec[i], ra[i]] for i in idx_random])
        # dividing up tasks to each processor
        q, r = divmod(coords.size // 2, size)
        count = 2 * np.array([q + 1 if p < r else q for p in range(size)])
        disp = np.array([sum(count[:p]) for p in range(size)])
    else:
        coords = None
        count = np.zeros(size, dtype=np.int)
        disp = None
    
    comm.Bcast(count, root=0)
    coords_buf = np.zeros((count[rank] // 2, 2))

    comm.Scatterv([coords, count, disp, MPI.DOUBLE], coords_buf, root=0)

    thumbs = thumbnails(imap, coords_buf, r=radius, res=res)
    #print(f"After thumbs, process {rank} has data of size {thumbs.shape}")
    stack_map = np.sum(utils.allgatherv(thumbs, comm), axis=0)
 
    if rank == 0:
        avg_map = stack_map / len(ra)
        return stack_map, avg_map


def stack_average_random(imap, ra, dec, Ncoords=NCOORDS,
                         radius=RAD, res=RESOLUTION):
    idx_random = np.random.choice(len(ra), Ncoords, replace=False)
    coords = np.array([[dec[i], ra[i]] for i in idx_random])
    
    # split up coords
    #multi_coords = np.array_split(coords, mp.cpu_count())
    #params = [(imap, multi_coords[i], radius, res) for i in range(len(multi_coords))]
    # create thumbnails in parallelization
    #pool = mp.Pool(mp.cpu_count())
    #thumbs = pool.starmap(thumbnails_kw, params)
    
    #pool.close()
    
    thumbs = thumbnails(imap, coords, r=radius, res=res)
    # stack
    stack_map = 0
    for i in range(len(thumbs)):
        stack_map += thumbs[i]

    # average
    avg_map = stack_map / Ncoords

    return stack_map, avg_map

def stack_random_all(imap, ra, dec, Ncoords=NCOORDS,
                     radius=RAD, res=RESOLUTION):
    idx_random = np.random.choice(len(ra), Ncoords, replace=False)
    coords = np.array([[dec[i], ra[i]] for i in idx_random])

    thumbs = thumbnails(imap, coords, r=radius, res=res)
    return thumbs

def stack_random_all_mpi(imap, ra, dec, Ncoords=NCOORDS,
                         radius=RAD, res=RESOLUTION):
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()

    if rank == 0:
        idx_random = np.random.choice(len(ra), Ncoords, replace=False)
        coords = np.array([[dec[i], ra[i]] for i in idx_random])
        # dividing up tasks to each processor
        q, r = divmod(coords.size // 2, size)
        count = 2 * np.array([q + 1 if p < r else q for p in range(size)])
        disp = np.array([sum(count[:p]) for p in range(size)])
    else:
        coords = None
        count = np.zeros(size, dtype=np.int)
        disp = None
    
    comm.Bcast(count, root=0)
    coords_buf = np.zeros((count[rank] // 2, 2))

    comm.Scatterv([coords, count, disp, MPI.DOUBLE], coords_buf, root=0)

    thumbs = thumbnails(imap, coords_buf, r=radius, res=res)
    # probably stupid, dangerous, or both

    return utils.allgatherv(thumbs, comm)

def stack_random_from_thumbs(thumbs):
    stack = sum([thumbs[i] for i in range(len(thumbs))])
    return stack, stack / len(thumbs)

# stack and average on the first N coordinates
# output stack, average maps
def stack_average_firstn(imap, ra, dec, Ncoords=NCOORDS,
                         radius=RAD, res=RESOLUTION):
    coords = np.array([[dec[i], ra[i]] for i in range(Ncoords)])

    # create thumbnails
    thumbs = thumbnails(imap, coords, r = radius, res=res)

    # stack
    stack_map = 0
    for i in range(len(thumbs)):
        stack_map += thumbs[i]

    # average
    avg_map = stack_map / Ncoords

    return stack_map, avg_map

# stack and average 2 maps on the same random subset of coordinates
# output stack, average maps
def stack_average_2map_random(imap1, imap2, ra, dec, Ncoords=NCOORDS,
                              radius=RAD, res=RESOLUTION):
    idx_random = np.random.choice(len(ra), Ncoords, replace=False)
    coords = np.array([[dec[i], ra[i]] for i in idx_random])

    # create thumbnails
    thumbs1 = thumbnails(imap1, coords, r = radius, res=res)
    thumbs2 = thumbnails(imap2, coords, r = radius, res=res)

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
    fetch_data()

    kap_px = px_to_car(KAP_FILENAME, res=RESOLUTION)
    ksz_px = px_to_car(KSZ_FILENAME, res=RESOLUTION)
    alm_px = alm_to_car(ALM_FILENAME, res=RESOLUTION)

    # plot_map(kap_px, plotname="submap-kap")
    # plot_map(ksz_px, plotname="submap-ksz")
    # plot_map(alm_px, plotname="submap-alm")

    ra, dec = catalog_to_coords()

    stack_ksz,avg_ksz,stack_kap,avg_kap = stack_average_2map_random(ksz_px, kap_px, ra, dec)

    plot_all(stack_ksz, avg_ksz, stack_kap, avg_kap, filename=output_filename)
    
# straight from https://github.com/cristobal-sifon/plottery/blob/master/src/plottery/plotutils.py
def update_rcParams(dict={}):
    """
    Update matplotlib's rcParams with any desired values. By default,
    this function sets lots of parameters to my personal preferences,
    which basically involve larger font and thicker axes and ticks,
    plus some tex configurations.
    Returns the rcParams object.
    """
    default = {}
    for tick in ('xtick', 'ytick'):
        default['{0}.major.size'.format(tick)] = 8
        default['{0}.minor.size'.format(tick)] = 4
        default['{0}.major.width'.format(tick)] = 2
        default['{0}.minor.width'.format(tick)] = 2
        default['{0}.minor.visible'.format(tick)] = True
        default['{0}.labelsize'.format(tick)] = 20
        default['{0}.direction'.format(tick)] = 'in'
    default['xtick.top'] = True
    default['ytick.right'] = True
    default['axes.linewidth'] = 2
    default['axes.labelsize'] = 22
    default['font.family'] = 'sans-serif'
    default['font.size'] = 22
    default['legend.fontsize'] = 18
    default['lines.linewidth'] = 2
    default['text.latex.preamble']=['\\usepackage{amsmath}']
    # Matthew Hasselfield's color-blind-friendly style
    default['axes.prop_cycle'] = \
        cycler(color=['#2424f0','#df6f0e','#3cc03c','#d62728','#b467bd',
                      '#ac866b','#e397d9','#9f9f9f','#ecdd72','#77becf'])
    for key in default:
        # some parameters are not valid in different matplotlib functions
        try:
            rcParams[key] = default[key]
        except KeyError:
            pass
    # if any parameters are specified, overwrite anything previously
    # defined
    for key in dict:
        try:
            rcParams[key] = dict[key]
        except KeyError:
            pass
    return

