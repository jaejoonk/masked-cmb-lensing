import get_cmb_powerspectra as websky_cmb
from falafel import utils as futils
import numpy as np
import matplotlib.pyplot as plt

LMAX=10050
websky_obj = websky_cmb.websky_cmb_spectra()
ucls_falafel, _ = futils.get_theory_dicts(lmax=LMAX, grad=False)

ucls_websky = {}
ucls_websky['TT'] = websky_obj['lensed_scalar'][0,0,:]
ucls_websky['EE'] = websky_obj['lensed_scalar'][1,1,:]
ucls_websky['TE'] = websky_obj['lensed_scalar'][1,0,:]
ucls_websky['BB'] = websky_obj['lensed_scalar'][2,2,:]

ells = np.arange(LMAX+1)
for est in ['TT', 'EE', 'TE', 'BB']:
    #ucls_falafel[est] = np.nan_to_num(ucls_falafel[est])
    #ucls_websky[est] = np.nan_to_num(ucls_websky[est])

    plt.title("Comparing falafel + websky theory Cls for %s estimator" % est)
    plt.yscale("log")
    plt.plot(ells[2:], ucls_falafel[est][2:], label="falafel Cl")
    plt.plot(ells[2:], ucls_websky[est][2:], label="websky Cl")
    plt.legend()
    plt.savefig("compare-ucls-%s.png" % est)
    plt.clf()

    # replace nans with zeros
    err = (ucls_websky[est] - ucls_falafel[est]) / ucls_falafel[est]
    plt.title("%% error of websky theory Cl vs falafel for %s estimator" % est)
    plt.plot(ells[2:], 100.*err[2:])
    plt.ylim(-20.,20.)
    plt.savefig("compare-ucls-diff-%s.png" % est)
    plt.close()
