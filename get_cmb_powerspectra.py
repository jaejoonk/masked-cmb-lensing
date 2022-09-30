# stolen from https://github.com/ajvanengelen/webskylensing/ (py/get_cmb_powerspectra.py)

import camb, numpy as np
import pdb

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

def websky_cmb_spectra(return_lensing = False, lmax = 10000, return_tensors = False):

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
    for name in powers: print(name)

    power_types = ['unlensed_scalar', 'lensed_scalar', 'tensor']

    output = {}

    for power_type in power_types:
        power = powers[power_type]

        output[power_type] = np.zeros((3, 3, power.shape[0]))

        ells = np.arange(power.shape[0])
        camb_factor = np.append([0], 2. * np.pi / (ells[1:] * (ells[1:] + 1) ))


        output[power_type][0,0,:] = power[:,0] * camb_factor #TT
        output[power_type][1,1,:] = power[:,1] * camb_factor #EE
        output[power_type][1,0,:] = power[:,3] * camb_factor #TE
        output[power_type][0,1,:] = power[:,3] * camb_factor #TE

        output[power_type][2,2,:] = power[:,2] * camb_factor #BB


    if return_lensing:
        output['lens_potential']  = powers['lens_potential']        
        
        # pdb.set_trace()
        # output['lens_potential'][0,0,:] = powers['unlensed_scalar'][:, 0] * camb_factor
        # output['lens_potential'][0,0,:] = powers['unlensed_scalar'][:, 0] * camb_factor

    # pdb.set_trace()

    return output

