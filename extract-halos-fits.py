from astropy.io import fits
import numpy as np

SNR = 5
FILENAME = "../nemo_allfgs_wnoise-wdr6dn_tsz-psmask-cori_optimalCatalog.fits"
OUTPUT_NAME = "coords-snr-5.txt"

hdul = fits.open(FILENAME)
data = hdul[1].data

# filter SNR
data = data[data['SNR'] > SNR]

ras = np.deg2rad(data['RADeg'])
decs = np.deg2rad(data['decDeg'])

coords = np.column_stack((decs, ras))

np.savetxt(OUTPUT_NAME, coords)

