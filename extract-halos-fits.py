from astropy.io import fits
import numpy as np
import healpy as hp

SNR = 5
#FILENAME = "../nemo_allfgs_wnoise-wdr6dn_tsz-psmask-cori_optimalCatalog.fits"
FILENAME = "sehgal_catalog.fits"
OUTPUT_NAME = "coords-snr"

def gen_coords_snr(fname=FILENAME, output_name=OUTPUT_NAME, snr=SNR):
    output_name += f"-{snr}.txt"
    hdul = fits.open(fname)
    data = hdul[1].data

    # filter SNR
    data = data[data['SNR'] > snr]

    ras = np.deg2rad(data['RADeg'])
    decs = np.deg2rad(data['decDeg'])

    coords = np.column_stack((decs, ras))

    np.savetxt(output_name, coords)
    print(f"Saved {len(coords)} coords to {output_name}.")

def gen_coords_fake(num, dec_min=-60., dec_max=20., output_name=OUTPUT_NAME):
    dec_min *= np.pi / 180.
    dec_max *= np.pi / 180.
    c = random_coords_range(num, dec_min=dec_min, dec_max=dec_max)
    output_name += f"-fake-{len(c)}.txt"

    # write as (dec,ra) in radians
    assert np.min(c[:,0]) > 0. # check that all ras are positive radians
    np.savetxt(output_name, np.column_stack((c[:,1], c[:,0])))
    print(f"Saved {len(c)} coords to {output_name}.")

## Generate a uniform distribution of ra,dec around a spherical projection
def random_ra_dec(N, zero=1e-4):
    xyz = []
    while len(xyz) < N:
        [x,y,z] = np.random.normal(size=3)
        # for rounding errors
        if (x**2 + y**2 + z**2)**0.5 > zero: xyz.append([x,y,z])
    colat, ra = hp.vec2ang(np.array(xyz))
    return ra, np.pi/2 - colat

# generate (N, 2) array of coordinates within a RA range + dec range
# iterative
def random_coords_range(N, ra_min=None, ra_max=None, dec_min=None, dec_max=None, zero=1e-6):
    coords = np.empty((0,2))
    while len(coords) < N:
        ras, decs = random_ra_dec(N, zero)
        prob_coords = np.column_stack((ras, decs))
        # filter ras
        if ra_min is not None:
            prob_coords = prob_coords[prob_coords[:,0] >= ra_min]
        if ra_max is not None:
            prob_coords = prob_coords[prob_coords[:,0] <= ra_max]
        if dec_min is not None:
            prob_coords = prob_coords[prob_coords[:,1] >= dec_min]
        if dec_max is not None:
            prob_coords = prob_coords[prob_coords[:,1] <= dec_max]
        coords = np.append(coords, prob_coords, axis=0)
    return np.array(coords[:N])

if __name__ == '__main__':
    gen_coords_fake(10259)

