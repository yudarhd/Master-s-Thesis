#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 15:45:30 2019

@author: yudarhd
"""

import numpy as np
import matplotlib.pyplot as plt

g1, bp1, rp1 = np.loadtxt("Blanco77_5.txt", usecols = (25, 26, 27), unpack = True)
#g2, bp2, rp2 = np.loadtxt("Collinder397_8.txt", usecols = (25, 26, 27), unpack = True)
#g3, bp3, rp3 = np.loadtxt("Collinder397_9.txt", usecols = (25, 26, 27), unpack = True)
Gmag1, bp_rp1 = np.loadtxt("Blanco_1.csv", usecols = (9, 10), skiprows =1,delimiter = ',', unpack = True)

Esc= input('Masukkan Nilai E(B-V) = ')
d1 = input('Masukkan Nilai d(pc) = ')

Av = 2.740 * Esc
Ag = 0.8615 * Av
Abp = 1.07185 * Av
Arp = 0.65069 * Av

mgm = []
mbpm = []
mrpm = []
mbprpm = []

for i in range (len(g1)):
    a = g1[i] - 5 + (5*np.log10(d1)) + Ag
    b = bp1[i] - 5 + (5*np.log10(d1)) + Abp
    c = rp1[i] - 5 + (5*np.log10(d1)) + Arp
    mgm.append(a)
    mbpm.append(b)
    mrpm.append(c)

BPRPmm = []
BPmm = mbpm
for i in range (len(mbpm)):
    mbprpm = mbpm[i] - mrpm[i]
    BPRPmm.append(mbprpm)

plt.plot(bp_rp1, Gmag1, 'r.')
plt.plot(BPRPmm, mgm, 'b--', lw = .5)
#plt.plot((bp2-rp2), g2-mu)
#plt.plot((bp3-rp3), g3-mu)
plt.ylim(max(g1), min(g1))
plt.xlabel('G$_{BP-RP}$ [mag]')
plt.ylabel('G [mag]')
plt.text(-0.26, 15, 'logt=7.75 (yr), E(B-V)=0.014, D=205 pc', fontsize=7)
'''

reddening = np.random.normal(0.54, 0.1, 10)
modulus = np.random.normal(12.0513, 0.1, 10)

Av = []
Ab = []
r = []

for i in range(len(modulus)):
    Av.append(reddening[i]*3.1)
    Ab.append(1.30*Av[i])
    r.append(modulus[i] + 5.0 - Av[i]) #in 5logd


def app_mag(mag_abs, r, A): #function to calculate apparent magnitude
    return -5 + r + A + mag_abs



print 'calculate BV & V of isochrones model'
Vk = np.zeros((len(Av), len(bp1)))
Bk = np.zeros((len(Av), len(bp1)))
BViso = np.zeros((len(Av), len(bp1)))

for j in range(len(Av)):
    print('test 1', j)
    for i in range(len(bp1)):
        Vk[j,i] = app_mag(rp1[i], r[j], Av[j])
        Bk[j,i] = app_mag(bp1[i], r[j], Ab[j])
        BViso[j,i] = Bk[j,i] - Vk[j,i]
print 'done'

print('''

''')

print 'make layers of V for data of cluster'
M_Vi = []
M_BVi = []
M_Visoi = []
M_BVisoi = []
for j in range(len(Av)):
    print('test 2', j)
    bin_V = np.arange(10.0, max(Gmag1), 0.5)
    M_V = []
    M_BV = []
    for i in range(len(bin_V)-1):
        BVn = []
        Vn = []
        bin0 = bin_V[i]
        for j in range(len(Gmag1)):
            if Gmag1[j] >= 10.0:
                if bin0 <= Gmag1[j] <= bin_V[i+1]:
                    BVn.append(bp_rp1[j])
                    Vn.append(bp_rp1[j])
        M_V.append(np.mean(Vn))
        M_BV.append(np.mean(BVn))
    M_Vi.append(M_V)
    M_BVi.append(M_BV)
print 'done'

print('''

''')

print 'make layers of V for isochrones model'
for j in range(len(Av)):
    print('test 3', j)
    bin_Viso = np.arange(10.0, max(Vk[j]), 0.5)
    M_Viso = []
    M_BViso = []
    for i in range(len(bin_Viso)-1):
        BVniso = []
        Vniso = []
        for l in range(len(bp1)):
            bin0iso = bin_Viso[i]
            if Vk[j,l] >= 10.0:
                if bin0iso <= Vk[j,l] <= bin_Viso[i+1]:
                    BVniso.append(BViso[j,l])
                    Vniso.append(Vk[j,l])
        M_Viso.append(np.mean(Vniso))
        M_BViso.append(np.mean(BVniso))
    M_Visoi.append(M_Viso)
    M_BVisoi.append(M_BViso)
print 'done'

print('''

''')


tot_CS = []
print 'calculating chi squared of each guess values ...'
for j in range(len(Av)):
    CS = []
    print('test 5', j)
    for i in range(len(M_BV)):
        test = abs(((M_BVi[j][i] - M_BVisoi[j][i])**2)/M_BVisoi[j][i])
        CS.append(test)
    tot_CS.append(np.sum(CS))
print 'done'
print('''

''')


best_value = min(tot_CS)
print 'finding the best value ...'

for i in range(len(Av)):
    if tot_CS[i] == best_value:
        print (reddening[i], modulus[i], tot_CS[i])
        
#plt.scatter(M_BV, M_V)

plt.plot(M_BViso, M_Viso, color = 'orange', label='WEBDA')

plt.plot(bp_rp1, Gmag1, 'r.')
plt.ylim(max(Gmag1), min(Gmag1))
plt.xlim(min(bp_rp1), max(bp_rp1))
'''