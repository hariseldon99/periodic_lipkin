from qutip import mesolve, basis, jmat, Options, expect, Qobj
from multiprocessing import Pool
from tqdm import tqdm
from scipy.special import jn_zeros
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.figsize": (20, 20),
    "font.size": 40})

def drive(t, args):
    w = args['omega']
    h = args['h']
    h0 = args['h0']
    return h * np.sin(w*t) + h0

def get_hamiltonians(N):
    sx,sy,sz = jmat(N,"x"),jmat(N,"y"),jmat(N,"z")
    kn =  2.0/(N-1)                                     
    H0 = kn * sz **2 
    H1 = 2 * sx
    return H0,H1

def sys_evolve(w,N,opts):
    sx,sy,sz = jmat(N,"x"),jmat(N,"y"),jmat(N,"z")
    T = 2 * np.pi/w
    h = jn_zeros(0,5)[0]* w / 4.0
    params = {'h0':np.pi/32, 'h':h,'omega':w, 'N':N} 
    H0, H1 = get_hamiltonians(N)
    H = [H0,[H1,drive]]
    en, sts = sx.eigenstates() 
    rho0 = sts[0]
    times = np.linspace(0,75*T, 500)
    hbar = []
    out = mesolve(H, rho0, times, [], [], args = params, options=opts)
    psi_ts = out.states
    
    for i,t in enumerate(times):
        Ht = H0 + H1 * drive(t, params)
        hbar.append(expect(Qobj(Ht),psi_ts[i]))
    hbar = (np.array(hbar)/N).reshape(len(times),1)
    return hbar.real, (times/(2*np.pi/w)).reshape(len(times),1)

if __name__ == '__main__':
    nprocs = 6
    N = 50
    o1 = np.linspace(0.1, 1.58, 20)
    o2 = np.linspace(1.6, 2.5, 20)
    o3 = np.linspace(2.8, 4.38, 15)
    o4 = np.linspace(8, 9.4, 15)
    o5 = np.linspace(9.5, 20.0, 15)
    o6 = np.linspace(20.1, 50.0, 11)


    omega_vals = np.concatenate((o1, o2, o3, o4, o5, o6))
    #omega_vals = np.concatenate((o3, o4, o5, o6))
    p = Pool(processes = nprocs)
    print("running for TSS spin N=",N, 'nprocs=',nprocs," !")
    opts = Options(nsteps=1e5, num_cpus=nprocs, openmp_threads=1, atol=1e-12, rtol=1e-14)
    data = np.array(p.starmap(sys_evolve,tqdm([(w,N, opts) for w in omega_vals])))
    
    
    hbaravg =[]
    for i,w in enumerate(omega_vals):
      hbar = data[i][0][:,0]
      times = data[i][1][:,0]

      hbaravg.append(np.average(hbar))
    lbl = 'N=' + str(N)
    plt.plot(omega_vals,hbaravg, label = lbl)

    plt.legend()
    plt.xlabel('w')
    plt.ylabel('<<H(t)>>')
    #plt.xlim(-5,10)
    figname = 'exact_pretherm_N' + str(N) + '.jpeg'
    #plt.savefig(figname, dpi = 800)
    plt.show()
