'''
The code from Javier Olivares (2017) has been tested and improved by Yuda Arif Hidayat (2019)
'''

import sys
import numpy as np
from numba import jit
import scipy.stats as st

from scipy.special import hyp2f1

@jit
def Support(rc,a,g):
    if rc <= 0 : return False
    if a  <= 0 : return False
    if g  <  0 : return False
    if g  >= 2 : return False
    return True

@jit
def cdf(r,params,Rm):
    rc = params[0]
    a  = params[1]
    g  = params[2]

    # Normalisation constant
    x = -((rc/Rm)**(-1.0/a) + 0.0j) 
    y = -((r/rc)**(1.0/a)   + 0.0j)
    u = -a*(g-2.0)
    v = 1.0 + a*g

    c = ((x**u)/u)*hyp2f1(u,1.0-v,1.0 + u,x)
    d = ((y**u)/u)*hyp2f1(u,1.0-v,1.0 + u,y)

    return d.real/c.real

@jit
def Number(r,params,Rm,Nstr):
    # res = np.zeros_like(r)
    # for i,s in enumerate(r):
    #     res[i] = cdf(r[i],params,Rm)
    return Nstr*cdf(r,params,Rm)

@jit
def Kernel(r,rc,a,g):
    y = ((r/rc)**g)*((1.0 + (r/rc)**(1.0/a))**(-a*g))
    return 1.0/y

@jit
def Density(r,params,Rm):
    rc = params[0]
    a  = params[1]
    g  = params[2]

    # Normalisation constant
    x = -1.0*((rc/Rm)**(-1.0/a))
    u = -a*(g-2.0)
    v = 1.0+ a*g
    z = ((-1.0+0j)**(a*(g-2.0)))*a*(rc**2)
    betainc = (((x+0j)**u)/u)*hyp2f1(u,1.0-v,1.0+u,x)
    k  = 1.0/np.abs(z*betainc)

    return k*Kernel(r,rc,a,g)

@jit
def LikeField(r,rm):
    return 2.*r/rm**2

class Module:
    """
    Chain for computing the likelihood 
    """
    def __init__(self,cdts,Rmax,hyp,Dist):
        """
        Constructor of the logposteriorModule
        """
        self.pro        = cdts[:,2]
        self.rad        = cdts[:,3]
        self.Rmax       = Rmax
        self.Prior_0    = st.halfcauchy(loc=0,scale=hyp[0])
        self.Prior_1    = st.uniform(loc=0.01,scale=hyp[1])
        self.Prior_2    = st.uniform(loc=0,scale=2)
        print("Module Initialized")

    def Priors(self,params, ndim, nparams):
        params[0]  = self.Prior_0.ppf(params[0])
        params[1]  = self.Prior_1.ppf(params[1])
        params[2]  = self.Prior_2.ppf(params[2])

    def LogLike(self,params,ndim,nparams):
        rc = params[0]
        a  = params[1]
        g  = params[2]
         #----- Checks if parameters' values are in the ranges
        if not Support(rc,a,g) : 
            return -1e50

        # Computes likelihood
        lf = (1.0-self.pro)*LikeField(self.rad,self.Rmax)
        lk = self.rad*(self.pro)*Kernel(self.rad,rc,a,g)

        # Normalisation constant
        x = -1.0*((rc/self.Rmax)**(-1.0/a))
        u = -a*(g-2.0)
        v = 1.0+ a*g
        z = ((-1.0+0j)**(a*(g-2.0)))*a*(rc**2)
        betainc = (((x+0j)**u)/u)*hyp2f1(u,1.0-v,1.0+u,x)
        k  = 1.0/np.abs(z*betainc)

        llike  = np.sum(np.log((k*lk + lf)))
        # print(llike)
        return llike


