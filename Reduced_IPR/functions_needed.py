import numpy as np
import scipy.linalg as la
from numpy import angle, pi
from qutip import Qobj, propagator, jmat
from qutip import tensor, sigmax, sigmay, sigmaz, qeye, mesolve 
from qutip.floquet import floquet_modes
from scipy.special import j0, jn_zeros

def floquet_modes_mod(H, T, args=None, parallel=False, sort=False, U=None):
    if 'opts' in args:
        options = args['opts']
    else:
        options = Options()
        options.rhs_reuse = True
        rhs_clear() 
    
    if U is None:
        # get the unitary propagator
        U = propagator(H, T, [], args, parallel=parallel, progressbar=True, options=options)
    
    # find the eigenstates for the propagator
    evals, evecs = la.eig(U.full())

    eargs = angle(evals)

    # make sure that the phase is in the interval [-pi, pi], so that
    # the quasi energy is in the interval [-pi/T, pi/T] where T is the
    # period of the driving.  eargs += (eargs <= -2*pi) * (2*pi) +
    # (eargs > 0) * (-2*pi)
    eargs += (eargs <= -pi) * (2 * pi) + (eargs > pi) * (-2 * pi)
    e_quasi = -eargs / T

    # sort by the quasi energy
    if sort:
        order = np.argsort(-e_quasi)
    else:
        order = list(range(len(evals)))

    # prepare a list of kets for the floquet states
    new_dims = [U.dims[0], [1] * len(U.dims[0])]
    new_shape = [U.shape[0], 1]
    kets_order = [Qobj(np.matrix(evecs[:, o]).T,
                       dims=new_dims, shape=new_shape) for o in order]

    return kets_order, e_quasi[order]

def drive(t, args):
    h0 = args['h0']
    h = args['h']
    w = args['omega']
    return h0 + h * np.cos(w * t)

def get_floquet_isingfermion(k, args, **kwargs):
    
    sx = sigmax()
    sz = sigmaz()
    
    # construct the hamiltonian
    H0 = np.sin(k) * sx + np.cos(k) * sz
    H1 = sz
    
    H = [H0,[H1,drive]]    
    T = 2 * np.pi/args['omega']

    return floquet_modes(H, T, args, **kwargs)

def get_iprvals_exact(k, params, **kwargs):
    f_states, f_energies = get_floquet_isingfermion(k, params, **kwargs)
    floquet_matrix = np.array(f_states)[:,:,0]
    return np.sum(np.abs(floquet_matrix)**4, axis=-1)

def drive_sx_rwa(t, args):
    n = args['order']
    w = args['omega']
    h = args['h']
    eta = 2*h/w
    cos_indices = np.arange(1,n+1)
        
    return np.sum([2 *jn(2*m, eta)*np.cos(2*m*w*t) for m in cos_indices])

def drive_sy_rwa(t, args):
    n = args['order']
    w = args['omega']
    h = args['h']
    eta = 2*h/w
    sin_indices = np.arange(1,n+1)          
    return np.sum([2*jn(2*m-1, eta)*np.sin((2*m-1)*w*t) for m in sin_indices])

def get_floquet_isingfermion_RWA(k, args, **kwargs):
    h = args['h']
    w = args['omega']
    eta = 2*h/w
    Dk = np.sin(k) 
    fk = np.cos(k)
    n = args['order']
    sx, sy, sz = sigmax(), sigmay(), sigmaz()
    H_rwa = Dk * sx * j0(eta) + sz * fk
    if n==0:
        H = H_rwa
    else:
        H = [H_rwa, [2 * Dk * sx, drive_sx_rwa] ,[-2 * Dk * sy, drive_sy_rwa]]
    
    T = 2 * np.pi/w
    return floquet_modes_mod(H, T, args, **kwargs)

def get_iprvals_RWA(k, params, **kwargs):
    f_states, f_energies = get_floquet_isingfermion_RWA(k, params, **kwargs)
    floquet_matrix = np.array(f_states)[:,:,0]
    return np.sum(np.abs(floquet_matrix)**4, axis=-1)

###   RWA CODES

def get_hamiltonians_RWA(N, args):
    w = args['omega']
    h = args['h']
    h0 = args['h0']
    sx,sy,sz = jmat(N,"x"),jmat(N,"y"),jmat(N,"z")
    kn =  1.0/(N-1)   
    
    H0 = -kn * (N/2 *(N/2+1) - sx*sx) - h0 * 2*sx
    H0 += (-kn) * (sz*sz - sy*sy) * j0(4*h/w)
    H1 = -2 * kn * (sz*sz - sy*sy)
    H2 = -2 * kn * (sy*sz + sz*sy)
    return H0,H1,H2


def floq_evolv_RWA(args):
    N = args['N']
    h = args['h']
    T = 2 * np.pi/args['omega']
    opts = args['opts']
    order = args['order']
    omega = args['omega']
    H0, H1, H2 = get_hamiltonians_RWA(N,args)
    if order==0:
        f_states, _ = floquet_modes_mod(H0, T, args=args)
    else:
        H = [H0]
        for n in np.arange(1,order+1,2): 
            sin_drive = 'sin((' + str(n) + '*omega'')*t)'
            H.append([H2*jn(n,4*h/w), sin_drive])
            
        for m in np.arange(2,order+1,2):     
            cos_drive = 'cos((' + str(m) + '*omega'')*t)'
            H.append([H1*jn(m, 4*h/w), cos_drive])
            
        f_states, _ = floquet_modes_mod(H, T, args=args)
        
    return f_states

from qutip import mesolve, basis, jmat

def get_hamiltonians(N):
    sx,sy,sz = jmat(N,"x"),jmat(N,"y"),jmat(N,"z")
    kn =  2.0/(N-1)                                      # kacNorm
    H0 = kn * sz **2 
    H1 = 2 * sx
    return H0,H1

def floq_evolv_lmg(args):
    N = args['N']
    T = 2 * np.pi/args['omega']
    opts = args['opts']
    H0, H1 = get_hamiltonians(N)
    H = [H0,[H1,drive]]
    f_states, _ = floquet_modes_mod(H, T, args=args)
    return f_states