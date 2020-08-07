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
print "GKing Centre imported!"

lo     = 1e-5

@jit
def Support(rc,rt,a,b):
    # if rc <= 0 : return False
    if rt <= rc : return False
    # if a <= 0 : return False
    # if b <= 0 : return False
    return True


@jit
def Kernel(r,rc,rt,a,b):
    x  = (1.0 +  (r/rc)**(1./a))**-a
    y  = (1.0 + (rt/rc)**(1./a))**-a
    z  = (x-y + 0j)**b
    return z.real

@jit
def Kernel1(r,rc,rt,a,b):
    x  = (1.0 +  (r/rc)**(1./a))**-a
    y  = (1.0 + (rt/rc)**(1./a))**-a
    z  = (x-y+0j)**b
    return z.real

def cdf(r,params,Rm):
    return NormCte(np.array([params,r]))/NormCte(np.array([params,Rm]))


def Number(r,params,Rm,Nstr):
    rc  = params[2]
    rt  = params[3]
    a   = params[4]
    b   = params[5]

    # Normalisation constant
    if rt < Rm:
        up = rt
    else:
        up = Rm

    cte = NormCte(np.array([rc,rt,a,b,up]))
    Num = np.vectorize(lambda y: integrate.quad(lambda x:x*Kernel1(x,rc,rt,a,b)/cte,lo,y,
                epsabs=1.49e-03, epsrel=1.49e-03,limit=1000)[0])
    return Nstr*Num(r)


def Density(r,params,Rm):
    rc  = params[2]
    rt  = params[3]
    a   = params[4]
    b   = params[5]

    # Normalisation constant
    if rt < Rm:
        up = rt
    else:
        up = Rm

    cte = NormCte(np.array([rc,rt,a,b,up]))
    Den = np.vectorize(lambda x:Kernel1(x,rc,rt,a,b)/cte)
    return Den(r)


def NormCte(z):
    cte = integrate.quad(lambda x:x*Kernel1(x,z[0],z[1],z[2],z[3]),lo,z[4],
                    epsabs=1.49e-03, epsrel=1.49e-03,limit=1000)[0]
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
        self.Prior_4    = st.truncexpon(b=hyp[4],loc=0.01,scale=hyp[5])
        self.Prior_5    = st.truncexpon(b=hyp[4],loc=0.01,scale=hyp[5])
        print("Module Initialized")

    def Priors(self,params, ndim, nparams):
        params[0]  = self.Prior_0.ppf(params[0])
        params[1]  = self.Prior_1.ppf(params[1])
        params[2]  = self.Prior_2.ppf(params[2])
        params[3]  = self.Prior_3.ppf(params[3])
        params[4]  = self.Prior_4.ppf(params[4])
        params[5]  = self.Prior_5.ppf(params[5])

    def LogLike(self,params,ndim,nparams):
        ctr= params[:2]
        rc = params[2]
        rt = params[3]
        a  = params[4]
        b  = params[5]
         #----- Checks if parameters' values are in the ranges
        if not Support(rc,rt,a,b) : 
            return -1e50

        #------- Obtains radii and angles ---------
        radii,theta    = Deg2pc(self.cdts,ctr,self.Dist)

        ############### Radial likelihood ###################
        # Computes likelihood
        lf = (1.0-self.pro)*LikeField(radii,self.Rmax)
        lk = radii*(self.pro)*Kernel(radii,rc,rt,a,b)

        # In king's profile no objects is larger than tidal radius
        idBad = np.where(radii > rt)[0]
        lk[idBad] = 0.0

        # Normalisation constant
        if rt < self.Rmax:
            up = rt
        else:
            up = self.Rmax

        cte = NormCte(np.array([rc,rt,a,b,up]))

        k = 1.0/cte

        llike_r  = np.sum(np.log((k*lk + lf)))
        ##################### POISSON ###################################
        quarter  = cut(theta,bins=self.quadrants,include_lowest=True)
        counts   = value_counts(quarter)
        llike_t  = self.poisson.logpmf(counts).sum()
        ##################################################################

        llike = llike_t + llike_r
        # print(llike)
        return llike



