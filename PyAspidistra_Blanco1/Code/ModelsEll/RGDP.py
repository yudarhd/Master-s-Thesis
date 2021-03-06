'''
The code from Javier Olivares (2017) has been tested and improved by Yuda Arif Hidayat (2019)
'''

import sys
import numpy as np
from numba import jit
import scipy.stats as st
from Functions import Deg2pc,TruncSort,RotRadii
from pandas import cut, value_counts

from scipy.special import hyp2f1

print "RGDP Elliptic imported!"

@jit
def Support(rca,rcb,a,b):
    # if rca <= 0 : return False
    # if rcb <= 0 : return False
    if rcb > rca: return False
    # if a   <= 0 : return False
    # if b   <  0 : return False
    # if a > 100.0 or b > 100.0 : return False
    return True

@jit
def cdf(r,params,Rm):
    rca = params[3]
    rcb = params[4]
    a   = params[5]
    b   = params[6]

    # Normalisation constant
    x = -((rca/Rm)**(-1.0/a) + 0.0j) 
    y = -((r/rca)**(1.0/a)   + 0.0j)
    u = 2.0*a

    c = ((x**u)/u)*hyp2f1(u,a*b,1.0 + u,x)
    d = ((y**u)/u)*hyp2f1(u,a*b,1.0 + u,y)

    return np.abs(d)/np.abs(c)

@jit
def Number(r,params,Rm,Nstr):
    return Nstr*cdf(r,params,Rm)

@jit
def Kernel(r,rc,a,b):
    y = (1.0 + (r/rc)**(1.0/a))**(a*b)
    return 1.0/y

@jit
def Kernel1(r,rc,a,b):
    y = (1.0 + (r/rc)**(1.0/a))**(a*b)
    return 1.0/y

@jit
def Density(r,params,Rm):
    rca = params[3]
    rcb = params[4]
    a   = params[5]
    b   = params[6]

    # Normalisation constant
    x = -1.0*((rca/Rm)**(-1.0/a))
    c = np.abs((Rm**2)*hyp2f1(2.0*a,a*b,1 + 2.0*a,x))
    k  = 2.0/c

    return k*Kernel1(r,rca,a,b)

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
        self.Prior_2    = st.uniform(loc=-0.5*np.pi,scale=np.pi)
        self.Prior_3    = st.halfcauchy(loc=0.0,scale=hyp[2])
        self.Prior_4    = st.halfcauchy(loc=0.0,scale=hyp[2])
        self.Prior_5    = st.truncexpon(b=hyp[3],loc=0.01,scale=hyp[4])
        self.Prior_6    = st.truncexpon(b=hyp[3],loc=0.01,scale=hyp[4])
        print("Module Initialized")

    def Priors(self,params, ndim, nparams):
        params[0]  = self.Prior_0.ppf(params[0])
        params[1]  = self.Prior_1.ppf(params[1])
        params[2]  = self.Prior_2.ppf(params[2])
        params[3]  = self.Prior_3.ppf(params[3])
        params[4]  = self.Prior_4.ppf(params[4])
        params[5]  = self.Prior_5.ppf(params[5])
        params[6]  = self.Prior_6.ppf(params[6])

    def LogLike(self,params,ndim,nparams):
        ctr = params[:2]
        dlt = params[2]
        rca = params[3]
        rcb = params[4]
        a   = params[5]
        b   = params[6]
         #----- Checks if parameters' values are in the ranges
        if not Support(rca,rcb,a,b) : 
            return -1e50

        #------- Obtains radii and angles ---------
        radii,theta    = RotRadii(self.cdts,ctr,self.Dist,dlt)

        rcs = (rca*rcb)/np.sqrt((rcb*np.cos(theta))**2+(rca*np.sin(theta))**2)

        if not Support(np.min(rcs),1,a,b) : 
            return -1e50

        ############### Radial likelihood ###################

        # Computes likelihood
        lf = (1.0-self.pro)*LikeField(radii,self.Rmax)
        lk = radii*(self.pro)*Kernel(radii,rcs,a,b)

        # Normalisation constant
        x = -((rcs/self.Rmax)**(-1.0/a))
        c = (self.Rmax**2)*hyp2f1(2.0*a,a*b,1 + 2.0*a,x).real
        k  = 2.0/c

        llike_r  = np.sum(np.log((k*lk + lf)))
        ##################### POISSON ###################################
        quarter  = cut(theta,bins=self.quadrants,include_lowest=True)
        counts   = value_counts(quarter)
        llike_t  = self.poisson.logpmf(counts).sum()
        ##################################################################

        llike = llike_t + llike_r
        # print(llike)
        if not np.isfinite(llike):
            return -1e50
        return llike


