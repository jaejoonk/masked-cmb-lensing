import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from orphics import maps, cosmology, io, pixcov, stats, catalogs
from falafel import qe, utils as futils
from pixell import enmap, curvedsky as cs, lensing as plensing
import healpy as hp
import pytempura

import websky_stack_and_visualize as webstack
import time, string, os

PATH_TO_SCRATCH = "/global/cscratch1/sd/jaejoonk/"
CATALOG_NAME = "overdensity_catalog.txt"
RESOLUTION = np.deg2rad(0.5/60.)
OUTPUT_MAP_NAME = PATH_TO_SCRATCH + "galaxy_counts_hp.fits"
OUTPUT_DELTA_MAP_NAME = PATH_TO_SCRATCH + "galaxy_delta_hp.fits"

# lets get a catalog of galaxies, from websky
# 9x10^12 ~ 5x10^13 Msun, 0.3 ~ 0.7 z
def get_catalog(out_filename=CATALOG_NAME,
                in_filename=(PATH_TO_SCRATCH + "halos.pksc"), Nhalos=None,
                m_cutoff_lo=0.09, m_cutoff_hi=0.5, z_cutoff_lo=0.3, z_cutoff_hi=0.7):
    f = open(in_filename)

    # number of halos from binary file
    Nhalo = np.fromfile(f, count=3, dtype=np.int32)[0]
    if Nhalos != None: Nhalo = Nhalos

    # x, y, z [Mpc], vx, vy, vz [km/s], M [M_sun], x_lag, y_lag, z_lag
    data = np.fromfile(f, count=Nhalo * 10, dtype=np.float32)

    # reshape into 2d array
    data_table = np.reshape(data, (Nhalo, 10))

    # fetch data from columns
    x, y, z, R = data_table[:,0], data_table[:,1], data_table[:,2], data_table[:,6]
    # from websky/readhalos.py
    z_red = webstack.zofchi(np.sqrt(x**2 + y**2 + z**2))
    # for mass cutoff
    mass = 4*np.pi/3.*webstack.RHO*R**3 / 1e14

    # convert to ra / dec (radians?) from readhalos.py
    colat, ra = hp.vec2ang(np.column_stack((x, y, z)))

    # convert colat to dec 
    dec = np.pi/2 - colat

    f.close()

    # truncate to mass range
    mass_indices = np.logical_and(mass >= m_cutoff_lo, mass <= m_cutoff_hi)
    ra, dec = ra[mass_indices], dec[mass_indices]
    mass, z_red = mass[mass_indices], z_red[mass_indices]
    # truncate to redshift range
    z_indices = np.logical_and(z_red >= z_cutoff_lo, z_red <= z_cutoff_hi)
    ra, dec = ra[z_indices], dec[z_indices]
    mass, z_red = mass[z_indices], z_red[z_indices]
    
    result = np.column_stack((ra, dec, mass * 1e14, z_red))
    np.savetxt(out_filename, result, delimiter=',')
    print(f"Wrote out catalog to {out_filename}.")
    return result

def get_overdensity_map(coords):
    dec_deg, ra_deg = coords[:,1] * 180./np.pi, coords[:,0] * 180./np.pi
    # fullsky geometry
    # shape, wcs = enmap.fullsky_geometry(res=RESOLUTION)

    catMap = catalogs.CatMapper(ra_deg, dec_deg, nside=2048)
    return catMap.get_map(), catMap.get_delta()

def do_all(got_coords=False, out_map_name=OUTPUT_MAP_NAME,
           out_delta_name=OUTPUT_DELTA_MAP_NAME):
    if got_coords: data = np.loadtxt(CATALOG_NAME, delimiter=",")
    else: data = get_catalog()
    coords = np.column_stack((data[:,0], data[:,1]))

    count_map, delta_map = get_overdensity_map(coords)
    hp.write_map(out_map_name, count_map)
    hp.write_map(out_delta_name, delta_map)

if __name__ == '__main__':
    do_all(got_coords=True)
   