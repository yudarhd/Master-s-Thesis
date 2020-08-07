'''
The code from Javier Olivares (2017) has been tested and improved by Yuda Arif Hidayat (2019)
'''

import sys
import numpy as np
from numba import jit
import scipy.stats as st
from Functions import Deg2pc,TruncSort
from pandas import cut, value_counts
import scipy.integrate as integrate

print "OGKing Centre imported!"
lo = 1e-5
a  = 0.548
b  = 1.378

@jit
def Support(rc,rt):
    # if rc <= 0 : return False
    if rt <= rc : return False
    return True

@jit
def Kernel(r,params):
    rc = params[2]
    rt = params[3]
    x  = (1.0 +  (r/rc)**(1./a))**-a
    y  = (1.0 + (rt/rc)**(1./a))**-a
    return (x-y)**b

def cdf(r,params,Rm):
    return NormCte(params,r)/NormCte(params,Rm)


def Number(r,params,Rm,Nstr):
    if params[3] < Rm :
        up = params[3]
    else:
        up = Rm
    cte = NormCte(params,up)
    Num = np.vectorize(lambda y: integrate.quad(lambda x:x*Kernel(x,params)/cte,lo,y,
                epsabs=1.49e-03, epsrel=1.49e-03,limit=1000)[0])
    return Nstr*Num(r)


def Density(r,params,Rm):
    if params[3] < Rm :
        up = params[3]
    else:
        up = Rm
    cte = NormCte(params,up)
    Den = np.vectorize(lambda x:Kernel(x,params)/cte)
    return Den(r)

def NormCte(params,up):
    cte = integrate.quad(lambda x:x*Kernel(x,params),lo,up,epsabs=1.49e-03, epsrel=1.49e-03,limit=1000)[0]
    return cte

@jit
def LikeField(r,rm):
    return 2.*r/rm**2

class Module:
    """
    Chain for computing the likelihood 
    """
    def __init__(self,cdts,Rcut,hyp,Dist,centre_init):
        """
        Constructor of the logposteriorModule
        """
        rad,thet        = Deg2pc(cdts,centre_init,Dist)
        c,r,t,self.Rmax = TruncSort(cdts,rad,thet,Rcut)
        self.pro        = c[:,2]
        self.cdts       = c[:,:2]
        self.Dist       = Dist
        #------------- poisson ----------------
        self.quadrants  = [0,np.pi/2.0,np.pi,3.0*np.pi/2.0,2.0*np.pi]
        self.poisson    = st.poisson(len(r)/4.0)
        #-------------- priors ----------------
        self.Prior_0    = st.norm(loc=centre_init[0],scale=hyp[0])
        self.Prior_1    = st.norm(loc=centre_init[1],scale=hyp[1])
        self.Prior_2    = st.halfcauchy(loc=0.01,scale=hyp[2])
        self.Prior_3    = st.halfcauchy(loc=0.01,scale=hyp[3])
        print("Module Initialized")

    def Priors(self,params, ndim, nparams):
        params[0]  = self.Prior_0.ppf(params[0])
        params[1]  = self.Prior_1.ppf(params[1])
        params[2]  = self.Prior_2.ppf(params[2])
        params[3]  = self.Prior_3.ppf(params[3])

    def LogLike(self,params,ndim,nparams):
        ctr= params[:2]
        rc = params[2]
        rt = params[3]
         #----- Checks if parameters' values are in the ranges
        if not Support(rc,rt) : 
            return -1e50

        #------- Obtains radii and angles ---------
        radii,theta    = Deg2pc(self.cdts,ctr,self.Dist)

        ############### Radial likelihood ###################
        # Computes likelihood
        lf = (1.0-self.pro)*LikeField(radii,self.Rmax)
        lk = radii*(self.pro)*Kernel(radii,params)

        # In king's profile no objects is larger than tidal radius
        idBad = np.where(radii > rt)[0]
        lk[idBad] = 0.0

        # Normalisation constant
        if rt < self.Rmax :
            up = rt
        else:
            up = self.Rmax
        cte = NormCte(params,up)

        k = 1.0/cte

        llike_r  = np.sum(np.log((k*lk + lf)))
        if np.isnan(llike_r):
            return -1e50
        ##################### POISSON ###################################
        quarter  = cut(theta,bins=self.quadrants,include_lowest=True)
        counts   = value_counts(quarter)
        llike_t  = self.poisson.logpmf(counts).sum()
        ##################################################################

        llike = llike_t + llike_r
        # print(llike)
        return llike



