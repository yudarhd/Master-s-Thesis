#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 14:57:39 2019

@author: yudarhd
"""

import numpy as np
import matplotlib.pyplot as mpy

#average or sbar

def average(xn, xm, yn, ym, nt):
    return ((2.)/(nt*(nt-1)))*(((xn-xm)**2. + (yn-ym)**2.)**0.5)

#smoothing lengths
def smooth(nt, sbar):
    return ((((2.)/nt)**0.5)*sbar)

#distance from center grid to another star
def jarak(x, xn, y, yn):
    return (((x-xn)**2. + (y-yn)**2.)**.5)

#counting for member star of the pixel to weight (nearest)
def nearest(arrV, divV, V):
    minV = min(arrV)
    maxV = max(arrV)
    x = (maxV - minV)/divV
    bb = np.int(np.floor((V-minV)/x))
    ba = np.int(np.floor((V+x-minV)/x))
    if bb < 0:
        bb = 0
    if ba < 0:
        ba = 0
    if ba >= len(arrV):
        ba = len(arrV)-1
    if bb >= len(arrV):
        bb = len(arrV)-1
    if abs(V - arrV[bb])<abs(V-arrV[ba]):
        return bb
    else:
        return ba

#list, array, and used parameters
Ra, Dec, J, eJ, x1, y1 = ([] for i in range(6))
Int = np.zeros((201,201))
Rap = 0.853
Decp = -29.958
Dt = 0.8

#Define list for array pixel
x = np.linspace((Rap-Dt), (Rap + Dt), 201)
y = np.linspace((Decp - Dt), (Decp + Dt), 201)
h = 0.03 #value smoothing length if not same with theory

#Read Data
RA, DEC, J, e_J = np.loadtxt('Blanco_1.csv', delimiter =',', skiprows = 1, usecols=(0,1,19,22), unpack = True)

#count sbar
sbar = 0.
for k in range (len(RA)-1):
    for l in range(k+1, len(RA)):
        sbar+= average(RA[k], RA[l], DEC[k], DEC[l], len(RA))
        
#count smoothing length
h = smooth(len(RA), sbar)

#count weight each pixel
for p in range (len(RA)):
    RAdkt = nearest(x, 201, RA[p])
    DEdkt = nearest(y, 201, DEC[p])
    dlt = RAdkt - nearest(x,201,x[RAdkt]-h)
    for i in range(RAdkt - dlt, RAdkt + dlt,1):
        if i >=0 and i < len(x):
            for j in range (DEdkt - dlt, DEdkt + dlt, 1):
                if j>= 0 and j<len(y):
                    r = jarak(RA[p], x[i], DEC[p], y[j])
                    if r<=h:
                        Int[j][i]+=((3*(h-r))/(np.pi*h**3))
x, y = np.meshgrid(x,y)

#contouring map
fig, ax = mpy.subplots()
im = ax.contourf(x,y, Int)

fig.colorbar(im)
mpy.title('Blanco 1')
mpy.xlabel('RA (Deg)')
mpy.ylabel('Dec (Deg)')
                        