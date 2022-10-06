# websky functions stolen from https://github.com/ajvanengelen/webskylensing/ (py/get_cmb_powerspectra.py)
import camb, numpy as np
import pdb
from orphics import cosmology, maps


def websky_cosmology():
#stolen from ~https://mocks.cita.utoronto.ca/data/websky/v0.0/cosmology.py
    output = {}
    output['omega_b'] = 0.049
    output['omega_c'] = 0.261
    output['omega_m'] = output['omega_b'] + output['omega_c']
    output['h']      = 0.68
    output['n_s']     = 0.965
    # sigma8 = 0.81

    output['A_s'] = 2.022e-9 #note this gets me s8 = 0.81027349, pretty close to the specified 0.81

    return output

def websky_cmb_spectra(return_lensing = False, lmax = 10000, grad = False, return_tensors = False):

    websky_params = websky_cosmology()

    #Set up a new set of parameters for CAMB
    pars = camb.CAMBparams()
    #This function sets up CosmoMC-like settings, with one massive neutrino and helium set using BBN consistency
    pars.set_cosmology(
        H0 = websky_params['h'] * 100,
        ombh2 = websky_params['omega_b'] * websky_params['h']**2,
        omch2 = websky_params['omega_c'] * websky_params['h']**2,
        mnu = 0.,
        omk = 0,
        tau = 0.055)

    if return_tensors:
        pars.WantTensors = True
        
    pars.InitPower.set_params(
        websky_params['A_s'],
        ns = websky_params['n_s'],
        r= (0 if (not return_tensors) else 1) , nt = 0)

    pars.set_for_lmax(lmax, lens_potential_accuracy=2)


    results = camb.get_results(pars)

    powers =results.get_cmb_power_spectra(pars, CMB_unit='muK')
    powers_grads = results.get_lensed_gradient_cls(CMB_unit='muK') if grad else None

    for name in powers: print(name)

    power_types = ['unlensed_scalar', 'lensed_scalar', 'tensor']

    output = {}

    for power_type in power_types:
        power = powers[power_type]
        #power_grad = powers_grads[power_type]

        output[power_type] = np.zeros((3, 3, power.shape[0]))

        ells = np.arange(power.shape[0])
        camb_factor = np.append([0], 2. * np.pi / (ells[1:] * (ells[1:] + 1) ))

        if (power_type == 'lensed_scalar' and grad):
            output[power_type][0,0,:] = powers_grads[:,0] * camb_factor #TT
            output[power_type][1,1,:] = powers_grads[:,1] * camb_factor #EE
            output[power_type][2,2,:] = powers_grads[:,2] * camb_factor #BB
        else:
            output[power_type][0,0,:] = power[:,0] * camb_factor #TT
            output[power_type][1,1,:] = power[:,1] * camb_factor #EE
            output[power_type][2,2,:] = power[:,2] * camb_factor #BB

        output[power_type][1,0,:] = power[:,3] * camb_factor #TE
        output[power_type][0,1,:] = power[:,3] * camb_factor #TE

    if return_lensing:
        output['lens_potential']  = powers['lens_potential']        
        
        # pdb.set_trace()
        # output['lens_potential'][0,0,:] = powers['unlensed_scalar'][:, 0] * camb_factor
        # output['lens_potential'][0,0,:] = powers['unlensed_scalar'][:, 0] * camb_factor

    # pdb.set_trace()
    return output

## Get theory spectra

# noise_t, noise_p in muK, beam_fwhm in arcmin
def noised_tcls(ucls, beam_fwhm, noise_t, noise_p=None):
    tcls = {}
    ells = np.arange(ucls[list(ucls.keys())[0]].size)
    if noise_p == None: noise_p = noise_t * np.sqrt(2.)
    
    ncls_T = ((noise_t * np.pi/180./60.) / maps.gauss_beam(beam_fwhm, ells))**2
    ncls_P = ((noise_p * np.pi/180./60.) / maps.gauss_beam(beam_fwhm, ells))**2
    
    tcls['TT'] = ucls['TT'] + ncls_T
    tcls['EE'] = ucls['EE'] + ncls_P
    tcls['TE'] = ucls['TE']
    tcls['BB'] = ucls['BB'] + ncls_P
    
    return tcls

def get_theory_dicts_white_noise_websky(beam_fwhm, noise_t, grad=True, noise_p=None, nells=None, lmax=5000):
    websky_spectra = websky_cmb_spectra(return_lensing=True, lmax=lmax, grad=grad)

    ucls = {}
    ucls['TT'] = websky_spectra['lensed_scalar'][0,0,:]
    ucls['TE'] = websky_spectra['lensed_scalar'][0,1,:]
    ucls['EE'] = websky_spectra['lensed_scalar'][1,1,:]
    ucls['BB'] = websky_spectra['lensed_scalar'][2,2,:]
    ucls['kk'] = websky_spectra['lens_potential']

    tcls = noised_tcls(ucls, beam_fwhm, noise_t, noise_p)
    
    return ucls, tcls

