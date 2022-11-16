from astropy.io import fits
import numpy as np

SNR = 5
#FILENAME = "../nemo_allfgs_wnoise-wdr6dn_tsz-psmask-cori_optimalCatalog.fits"
FILENAME = "sehgal_catalog.fits"
OUTPUT_NAME = "sehgal-coords-snr"

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

if __name__ == '__main__':
    #gen_coords_snr(snr=4)
    #gen_coords_snr(snr=6)
    #gen_coords_snr(snr=7)
    gen_coords_snr(snr=4)
    gen_coords_snr(snr=5)
    gen_coords_snr(snr=6)
    gen_coords_snr(snr=7)

