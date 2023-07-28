#!/usr/bin/env python

import numpy as np
from qutip import tensor, sigmax, sigmay, sigmaz, qeye, mesolve 
from qutip.floquet import floquet_modes

def drive(t, args):
    h0 = args['h0']
    h = args['h']
    w = args['omega']
    return h0 + h * np.cos(w * t)

def get_sxt(k, nT, params, ntimes=1000, initstate=sigmaz().groundstate()[1]):
    sx = sigmax()
    sz = sigmaz()

    # construct the hamiltonian
    H0 = np.sin(k) * sx + np.cos(k) * sz
    H1 = sz
    
    H = [H0,[H1,drive]]   
    T = 2 * np.pi/params['omega']
    times = np.linspace(0, nT * T, ntimes)
    return mesolve(H, initstate, times, e_ops=[sz], args=params)

def get_floquet_isingfermion(k, args, **kwargs):
    
    sx = sigmax()
    sz = sigmaz()
    
    # construct the hamiltonian
    H0 = np.sin(k) * sx + np.cos(k) * sz
    H1 = sz
    
    H = [H0,[H1,drive]]    
    T = 2 * np.pi/args['omega']

    return floquet_modes(H, T, args, **kwargs)

def get_uv(k, params):
    f_states, f_energies = get_floquet_isingfermion(k, params, sort=True)
    return f_states[-1].full().flatten()

def get_tpdm(args):
    kq,params = args
    k,q = kq
    uk, vk = get_uv(k, params)
    uk_c, vk_c = np.conjugate(uk),np.conjugate(vk)
    
    uq, vq = get_uv(q, params)
    uq_c, vq_c = np.conjugate(uq),np.conjugate(vq)
    
    rho_matelem = vk_c * uk * uq_c * vq
    if k == 0 or q == 0:
        rho_matelem = 0.0
    elif k == q:
        rho_matelem -= np.abs(uk * vq)**2 + np.abs(uq * vk)**2 + uk_c * vk * vq_c * uq
    elif k == -q:
        rho_matelem += np.abs(uk * vq)**2 + np.abs(uq * vk)**2 - uk_c * vk * vq_c * uq
    return rho_matelem


import itertools, h5py
from qutip import parfor
from scipy.special import j0, jn_zeros

N = 1000

num_cpus = 48
omega_max = 50
num_omegas = 100
h0 = 0.0

fname="average_iprs_TFIM.hdf5"
fbz = np.linspace(-np.pi, np.pi, N)
kq_pairs = [x for x in itertools.combinations(fbz, 2)]

omegas = np.linspace(0.1, omega_max, num_omegas)
eta = jn_zeros(0,10)[0]
f = h5py.File(fname, "a")
ipr_dset = f.create_dataset(f'omega_ipr_{N}',(num_omegas,2),dtype=np.float64)

for idx,omega in enumerate(omegas):
    h = eta * omega/2
    params = {'h0':h0, 'h':h, 'omega':omega}
    arglist = [(kq, params) for kq in kq_pairs]
    rholist = parfor(get_tpdm, arglist, num_cpus=num_cpus)
    rho = np.zeros((N,N), dtype=np.complex128)
    for ri, kq in enumerate(kq_pairs):
        k, q = kq
        i, = np.where(fbz==k)
        j, = np.where(fbz==q)
        rho[i,j] = rholist[ri]
    rho = rho + np.conjugate(rho.T)
    n, phis = np.linalg.eig(rho)

    ipr =  np.sum(np.abs(phis)**4)/N

    ipr_dset[idx,:] = [omega, ipr]
    f.flush()
ipr_dset.attrs['nspins'] = N
ipr_dset.attrs['h0'] = h0
ipr_dset.attrs['h'] = h
ipr_dset.attrs['j0_val'] = j0(2*h/omega)

f.close()
