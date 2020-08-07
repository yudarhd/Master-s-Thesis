'''
The code from Javier Olivares (2017) has been tested and improved by Yuda Arif Hidayat (2019)
'''

from __future__ import absolute_import, unicode_literals, print_function
import sys
#sys.path.insert(0, "/pcdisk/boneym5/jolivares/PyAspidistra/Code")
import os
import numpy as np
import importlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pymultinest
import corner

from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
from scipy.optimize import brentq
import scipy.stats as st
from Functions import Deg2pc,TruncSort,DenNum,RotRadii,fMAP,fCovar,Epsilon,MassRj,MassEps

from astropy import units as u
from astropy.coordinates import SkyCoord
import math
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D

#########################################################################################
quantiles = [0.16,0.5,0.84]

dir_  = os.path.expanduser('~') +"/PyAspidistra_Blanco1/"
real  = True
Ntot  = 10000  # Number of stars if synthetic (real = False)

nargs = len(sys.argv)
if nargs == 4:
	na = 0
if nargs == 3:
	na = 1
model = str(sys.argv[1-na])
Rcut  = float(sys.argv[2-na])
exte  = str(sys.argv[3-na])

if exte == "None":
	dext = "Models"
	fext = "NoCentre/"
elif exte == "Ctr":
	dext = "ModelsCtr"
	fext = "Centre/"
elif exte == "Ell":
	dext = "ModelsEll"
	fext = "Elliptic/"
elif exte == "Seg":
	dext = "ModelsSeg"
	fext = "Segregated/"
else :
	print("Extention not recognised!")
	print("Available ones are: None, Centre (Ctr), Elliptic (Ell), and Segregated (Seg)")
	sys.exit()

########### Load module ######################
mod = importlib.import_module(dext+"."+model)
##############################################

############## Directory of outputs #########################
if real :
	dir_out  = dir_+'Analysis/'+fext+model+'_'+str(int(Rcut))
else:
	dir_out  = dir_+'Analysis/Synthetic/'+fext+model+'_'+str(int(Rcut))+'_'+str(Ntot)
	fsyn     = dir_out+'/2-data.csv'
if not os.path.exists(dir_out): os.makedirs(dir_out)
################################### MODEL ######################################################

texp = 100.0 # truncation of exponential prior for exponents
sexp = 1.0  # scale of  ""      """
src  = 1.0  # scale of half-cauchy for core radius
srt  = 10.0 #   ""        """            tidal radius


if model == "EFF":
	if exte == "None" or exte =="Ctr":
		#----------------------------------
		namepar = ["$r_c$ [pc]","$\gamma$"]
		params  = [2.0,3.0]
		rng     = [[0,4],[2,4]]
	if exte == "Ell" or exte =="Seg":
		#--------- Initial parameters --------------
		namepar = ["$\phi$ [radians]","$r_{ca}$ [pc]","$r_{cb}$ [pc]","$\gamma$"]
		params  = [np.pi/4,2.0,2.0,0.0]
		rng     = [[-0.5*np.pi,0.5*np.pi],[0,4],[0,4],[2,4]]
		id_rc   = [3,4]
	hyp     = np.array([src,texp,sexp])

if model == "GDP":
	if exte == "None" or exte =="Ctr":
	#--------- Initial parameters --------------
		namepar = ["$r_c$ [pc]","$\\alpha$","$\\beta$","$\gamma$"]
		params  = [2.0,0.5,2.0,0.1]
		rng     = [[0,10],[0,2],[0,10],[0,1]]

	if exte == "Ell" or exte =="Seg":
		#--------- Initial parameters --------------
		namepar = ["$\phi$ [radians]","$r_{ca}$ [pc]","$r_{cb}$ [pc]",
					"$\\alpha$","$\\beta$","$\gamma$"]
		params  = [np.pi/4,2.0,2.0,0.5,2.0,0.0]
		rng     = [[-0.5*np.pi,0.5*np.pi],[0,10],[0,10],[0,2],[0,10],[0,2]]
		id_rc   = [3,4]
	hyp     = np.array([src,texp,sexp])

if model == "GKing":
	if exte == "None" or exte =="Ctr":
		#--------- Initial parameters --------------
		namepar = ["$r_c$ [pc]","$r_t$ [pc]","$\\alpha$","$\\beta$"]
		params  = [2.0,20.0,0.5,2.0]
		rng     = [[0,5],[10,50],[0,2],[0,5]]
		idrt    = 3

	if exte == "Ell" or exte =="Seg":

		#--------- Initial parameters --------------
		namepar = ["$\phi$ [radians]","$r_{ca}$ [pc]","$r_{ta}$ [pc]", 
					"$r_{cb}$ [pc]","$r_{tb}$ [pc]","$\\alpha$","$\\beta$"]
		params  = [np.pi/4,2.0,20.0,2.0,20.0,0.5,2.0]
		rng     = [[-0.5*np.pi,0.5*np.pi],[0,5],[10,50],[0,5],[10,50],[0,2],[0,5]]
		id_rc   = [3,5]
		idrt    = 4
	hyp     = np.array([src,srt,texp,sexp])

if model == "King":
	if exte == "None" or exte =="Ctr":
		#--------- Initial parameters --------------
		namepar = ["$r_c$ [pc]", "$r_t$ [pc]"]
		params  = [2.0,20.0]
		rng     = [[0,5],[10,70]]
		idrt    = 3
	if exte == "Ell" or exte =="Seg":
		#--------- Initial parameters --------------
		namepar = ["$\phi$ [radians]","$r_{ca}$ [pc]","$r_{ta}$ [pc]",
					 "$r_{cb}$ [pc]","$r_{tb}$ [pc]"]
		params  = [np.pi/4,2.0,20.0,2.0,20.0]
		rng     = [[-0.5*np.pi,0.5*np.pi],[0,5],[10,70],[0,5],[10,70]]
		id_rc   = [3,5]
		idrt    = 4
	hyp     = np.array([src,srt])

if model == "OGKing":
	if exte == "None" or exte =="Ctr":
		#--------- Initial parameters --------------
		namepar = ["$r_c$ [pc]","$r_t$ [pc]"]
		params  = [2.0,20.0]
		rng     = [[0,4],[10,40]]
		idrt    = 3

	if exte == "Ell" or exte =="Seg":
		#--------- Initial parameters --------------
		namepar = ["$\phi$ [radians]","$r_{ca}$ [pc]","$r_{ta}$ [pc]",
					 "$r_{cb}$ [pc]","$r_{tb}$ [pc]"]
		params  = [np.pi/4,2.0,20.0,2.0,20.0]
		rng     = [[-0.5*np.pi,0.5*np.pi],[0,4],[10,40],[0,4],[10,20]]
		id_rc   = [3,5]
		idrt    = 4
	hyp     = np.array([src,srt])

if model == "RGDP":
	if exte == "None" or exte =="Ctr":
		#--------- Initial parameters --------------
		namepar = ["$r_c$ [pc]","$\\alpha$","$\\beta$"]
		params  = [2.0,0.5,2.0]
		rng     = [[0,10],[0,2],[0,10]]

	if exte == "Ell" or exte =="Seg":
		#--------- Initial parameters --------------
		namepar = ["$\phi$ [radians]","$r_{ca}$ [pc]","$r_{cb}$ [pc]",
					"$\\alpha$","$\\beta$"]
		params  = [np.pi/4,2.0,2.0,0.5,2.0]
		rng     = [[-0.5*np.pi,0.5*np.pi],[0,10],[0,10],[0,2],[0,10]]
		id_rc   = [3,4]
	hyp     = np.array([src,texp,sexp])


######### parameters of centre ####################
Dist    = 235.9
centre  = [0.853, -29.958]

nameparCtr = ["$\\alpha_c\ \ [^\circ]$","$\\delta_c\ \ [^\circ]$"]
hypCtr    = np.array([1.0,1.0])
rngCtr    = [[centre[0]-1.0,centre[0]+1.0],[centre[1]-1.0,centre[1]+1.0]]
######### Parameter of luminosity segregation #########
nameSg    = ["$\kappa$ [pc$\cdot \\rm{mag}^{-1}$]"]
hypSg     = np.array([0,0.5]) # Uniform prior between hypSg[0] and hypSg[0]+hypSg[1]
rngSg     = [[-0.6,1.2]]
#####################################################

#------- concatenate parameters
if not exte == "None" :
	namepar = nameparCtr + namepar
	hyp     = np.append(hypCtr,hyp)
	rng     = sum([rngCtr,rng],[])

if exte == "Seg":
	namepar = namepar + nameSg
	hyp     = np.append(hyp,hypSg)
	rng     = sum([rng,rngSg],[])

##################### DATA ##############################################################

if real :
	#------- reads data ---------------
	#fdata = dir_+'Data/OnlyTycho.csv'
	#cdtsT = np.array(pd.read_csv(fdata,header=0,sep=',',usecols=[1,2,17,18,32],dtype=np.float64))#K band col=21
	#cdtsT[:,[2, 3,4]] = cdtsT[:,[4,2,3]] # Puts band and uncertainti in lasts columns
	fdata = dir_+'Data/Blanco_1.csv'
	cdtsD  = np.array(pd.read_csv(fdata,header=0,sep=',',usecols=[11,0,1,19,22],dtype=np.float64)) #K band col=8
	#cdtsD[:,[0,1,2]] = cdtsD[:,[1,2,0]] # reorder
	print("Make shure to have data in right order! (R.A., Dec. Prob., Band,e_band)")
	cdts  = np.vstack([cdtsT, cdtsD])
	#---- removes duplicateds --------
	sumc  = np.sum(cdts[:,:2],axis=1)
	idx   = np.unique(sumc,return_index=True)[1]
	cdts  = cdts[idx]
	sumc  = np.sum(cdts[:,:2],axis=1)
	if len(sumc) != len(list(set(sumc))):
		sys.exit("Duplicated entries in Coordinates!")
	##----- Select objects with J band observed and less than 19
	ido   = np.where(cdts[:,3] <= 20)[0]
	cdts  = cdts[ido]
	# print(cdts.shape)
else :
	# -------- Check if file exists -------
	if os.path.exists(fsyn):
		cdts     = np.array(pd.read_csv(fsyn,header=0,sep=','))
	else:
		# ----- Creates synthetic data ---------
		unifsyn  = np.random.uniform(low=0.0,high=1.0,size=Ntot)
		radii    = np.empty(Ntot)
		for s,st in enumerate(unifsyn):
			radii[s]= brentq(lambda x:mod.cdf(x,params,float(Rcut))-st,a=0.0,b=float(Rcut))
		theta    = np.random.uniform(low=0.0,high=2.*np.pi,size=Ntot)
		cdts     = np.empty((Ntot,3))
		cdts[:,0]= (radii/Dist)*np.cos(theta)*R2D + centre[0]
		cdts[:,1]= (radii/Dist)*np.sin(theta)*R2D + centre[1]
		cdts[:,2]= np.repeat(1,Ntot)
		radii = None
		theta = None
		np.savetxt(fsyn,cdts, delimiter=",")

################# Calculates Radii and PA ################
#------------ Load Module -------
Module  = mod.Module(cdts,Rcut,hyp,Dist,centre)

#========================================================
#--------- Dimension, walkers and inital positions -------------------------
# number of dimensions our problem has
ndim     = len(namepar)
n_params = len(namepar)

pymultinest.run(Module.LogLike,Module.Priors, n_params,resume = False, verbose = True, n_live_points=100,
 	outputfiles_basename=dir_out+'/0-',multimodal=True, max_modes=2,sampling_efficiency = 'model')

# analyse the results
ana = pymultinest.Analyzer(n_params = n_params, outputfiles_basename=dir_out+'/0-')
summary = ana.get_stats()
samples = ana.get_data()[:,2:]
np.savetxt(dir_out+'/0-foo.csv',samples, delimiter=",")

# MAP   = np.array(summary["modes"][0]["maximum"])
# MAP   = np.array(summary["modes"][0]["maximum a posterior"])
# MAP   = np.array(summary["modes"][0]["mean"])
MAP   = fMAP(samples)
# print(MAP)
print()
print("-" * 30, 'ANALYSIS', "-" * 30)
print("Global Evidence:\n\t%.15e +- %.15e" % (summary['nested sampling global log-evidence'], 
	summary['nested sampling global log-evidence error'] ))

########################## PLOTS #########################
# ----- select only 100 parameters to plot
samp = samples[np.random.choice(np.arange(len(samples)),size=100,replace=False)]


############### Stores the covariance matrix ##############
print("Finding covariance matrix around MAP ...")
covar = fCovar(samples,MAP)
np.savetxt(dir_out+"/"+model+"_covariance.txt",covar,
		fmt=str('%2.3f'),delimiter=" & ",newline=str("\\\ \n"))


# ------ establish new centre (if needed) ------
if not exte == "None":
	centre = MAP[:2]
# -------- prepare data for plots ---------
rad,thet              = Deg2pc(cdts,centre,Dist)
cdts,radii,theta,Rmax = TruncSort(cdts,rad,thet,Rcut)

if exte == "Ell":
	radii,theta       = RotRadii(cdts,centre,Dist,MAP[3])

Nr,bins,dens          = DenNum(radii,Rmax)
x                     = np.linspace(0.01,Rmax,50)

pdf = PdfPages(dir_out+"/"+model+'_fit.pdf')

plt.figure()
for s,par in enumerate(samp):
	plt.plot(x,mod.Number(x,par,Rmax,np.max(Nr)),lw=1,color="orange",alpha=0.2,zorder=1)
plt.fill_between(radii, Nr+np.sqrt(Nr), Nr-np.sqrt(Nr), facecolor='grey', alpha=0.5,zorder=2)
plt.plot(radii,Nr,lw=1,color="black",zorder=3)
plt.plot(x,mod.Number(x,MAP,Rmax,np.max(Nr)), lw=1,color="red",zorder=4)
plt.ylim((0,1.1*max(Nr)))
plt.xlim((0,1.1*Rmax))
plt.ylabel("Number of stars")
plt.xlabel("Radius [pc]")
pdf.savefig(bbox_inches='tight')  # saves the current figure into a pdf page
plt.close()


for s,par in enumerate(samp):
	plt.plot(x,mod.Density(x,par,Rmax),lw=1,color="grey",alpha=0.2,zorder=1)
plt.errorbar(bins,dens[:,0],yerr=dens[:,1],fmt="o",color="black",lw=1,ms=2,zorder=3)
plt.plot(x,mod.Density(x,MAP,Rmax), linewidth=1,color="red",zorder=2)
plt.ylabel("Density [stars $\cdot$ pc$^{-2}$]")
plt.xlabel("Radius [pc]")
plt.ylim(1e-3,0.5)
plt.yscale("log")
pdf.savefig(bbox_inches='tight')  # saves the current figure into a pdf page
plt.close()

plt.figure()
n, bins, patches = plt.hist(theta,50, normed=1, facecolor='green', alpha=0.5)
plt.xlabel('Position Angle [radians]')
plt.ylabel('Density [stars $\cdot$ radians$^{-1}$]')
pdf.savefig(bbox_inches='tight')  # saves the current figure into a pdf page
plt.close()

plt.rc('font', size=14)  

if real :
	corner.corner(samples, labels=namepar,truths=MAP,truth_color="red",range=rng,
		reverse=False,plot_datapoints=False,fill_contours=True, quantiles=quantiles, show_titles=True, title_kwargs={"fontsize": 12})
else:
	corner.corner(samples, labels=namepar,truths=params,truth_color="blue",range=rng,
		reverse=False,plot_datapoints=False,fill_contours=True, quantiles=quantiles, show_titles=True, title_kwargs={"fontsize": 12})
pdf.savefig(bbox_inches='tight')
plt.close()

if exte == "Seg":
	#------ separates data in bins -----------
	low  = 12.0
	up   = 14.0
	idl  = np.where(cdts[:,3] <= low)[0]
	idu  = np.where(cdts[:,3] >= up)[0]
	idm  = np.where(np.logical_and(cdts[:,3] > low,cdts[:,3] < up))[0]

	radii_l = radii[idl]
	radii_m = radii[idm]
	radii_u = radii[idu]

	Nr_l,bins,dens_l = DenNum(radii_l,Rmax,nbins=21)
	Nr_m,bins,dens_m = DenNum(radii_m,Rmax,nbins=21)
	Nr_u,bins,dens_u = DenNum(radii_u,Rmax,nbins=21)

	dlt_l  = np.mean(cdts[idl,3]) - Module.mode
	dlt_m  = np.mean(cdts[idm,3]) - Module.mode
	dlt_u  = np.mean(cdts[idu,3]) - Module.mode

	# plt.figure()
	# plt.errorbar(radii[idl],cdts[idl,3],yerr=cdts[idl,4],fmt="o",color="green",ecolor="grey",lw=1,ms=2)
	# plt.errorbar(radii[idm],cdts[idm,3],yerr=cdts[idm,4],fmt="o",color="cyan",ecolor="grey",lw=1,ms=2)
	# plt.errorbar(radii[idu],cdts[idu,3],yerr=cdts[idu,4],fmt="o",color="magenta",ecolor="grey",lw=1,ms=2)
	# plt.plot(x,Module.mode+(x*MAP[7]))
	# plt.xlabel("Radius [pc]")
	# plt.ylabel('J [mag]')
	# pdf.savefig(bbox_inches='tight')  # saves the current figure into a pdf page
	# plt.close()

	

	for s,par in enumerate(samp):
		plt.plot(x,mod.DenSeg(x,par,Rmax,dlt_l),
			lw=1,color="grey",alpha=0.2,zorder=1)

	plt.errorbar(bins,dens_l[:,0],yerr=dens_l[:,1],fmt="o",color="green",ecolor="grey",lw=1,ms=2,zorder=3)
	plt.plot(x,mod.DenSeg(x,MAP,Rmax,dlt_l),
			 linewidth=1,color="green",zorder=2)

	plt.errorbar(bins,dens_m[:,0],yerr=dens_m[:,1],fmt="o",color="cyan",ecolor="grey",lw=1,ms=2,zorder=3)
	plt.plot(x,mod.DenSeg(x,MAP,Rmax,dlt_m),
			 linewidth=1,color="cyan",zorder=2)

	plt.errorbar(bins,dens_u[:,0],yerr=dens_u[:,1],fmt="o",color="magenta",ecolor="grey",lw=1,ms=2,zorder=3)
	plt.plot(x,mod.DenSeg(x,MAP,Rmax,dlt_u),
			 linewidth=1,color="magenta",zorder=2)

	plt.ylabel("Density [stars $\cdot$ pc$^{-2}$]")
	plt.xlabel("Radius [pc]")
	plt.ylim(1e-3,0.5)
	plt.yscale("log")
	pdf.savefig(bbox_inches='tight')  # saves the current figure into a pdf page
	plt.close()

############################ SKY CHART ################################################
figsize = (10,10)
fig, ax = plt.subplots(figsize=figsize)
#--------------- Create grid in ra dec ----------
fk5_centre = SkyCoord(ra=centre[0],dec=centre[1],frame='icrs',unit='deg')
gal_centre = np.array([fk5_centre.galactic.l.degree,fk5_centre.galactic.b.degree])

fk5_cdts = SkyCoord(ra=cdts[:,0],dec=cdts[:,1],frame='icrs',unit='deg')

gal_cdts = np.empty((len(cdts[:,0]),2))
gal_cdts[:,0] = fk5_cdts.galactic.l.degree
gal_cdts[:,1] = fk5_cdts.galactic.b.degree

#---------- plot stars -----------------------------------
plt.scatter(gal_cdts[:,0],gal_cdts[:,1],s=10*cdts[:,3], edgecolor='grey', facecolor='none', marker='*',zorder=1)
plt.xlim(np.min(gal_cdts[:,0])-1.0,np.max(gal_cdts[:,0])+1)
plt.ylim(np.min(gal_cdts[:,1])-1.0,np.max(gal_cdts[:,1])+1)
ax.set_aspect('equal')
if exte == "Ell" or exte == "Seg":
	rca = math.degrees(MAP[3]/Dist)
	p1 = centre + 5*rca*np.array([np.cos(MAP[2]),np.sin(MAP[2])])
	# plt.scatter(p1[0],p1[1],s=20,color="red", marker='*',zorder=1)

if exte == "Ctr":
	rca = math.degrees(MAP[2]/Dist)
	rcb = rca
	p1 = centre + 5*rca


fk5_p1 = SkyCoord(ra=p1[0],dec=p1[1],frame='icrs',unit='deg')
gal_p1 = np.array([fk5_p1.galactic.l.degree,fk5_p1.galactic.b.degree])
angle = math.degrees(np.arctan2(gal_p1[1]-gal_centre[1],gal_p1[0]-gal_centre[0]))


if exte == "Ell" or exte == "Seg":
	if model in ["GKing","King","OGKing"]:
		rcb = math.degrees(MAP[5]/Dist)
		rta = math.degrees(MAP[4]/Dist)
		rtb = math.degrees(MAP[6]/Dist)
		ell  = Ellipse(gal_centre,width=2*rta,height=2*rtb,angle=angle,clip_box=ax.bbox,
			edgecolor="black",facecolor=None,fill=False,linewidth=2)
		ax.add_artist(ell)

	else:
		rcb = math.degrees(MAP[4]/Dist)

	p1 = gal_centre + 5*rca*np.array([np.cos(math.radians(angle+180)),np.sin(math.radians(angle+180))])
	p2 = gal_centre + 5*rca*np.array([np.cos(math.radians(angle)),np.sin(math.radians(angle))])
	p3 = gal_centre + 5*rcb*np.array([np.cos(math.radians(angle+90+180)),np.sin(math.radians(angle+90+180))])
	p4 = gal_centre + 5*rcb*np.array([np.cos(math.radians(angle+90)),np.sin(math.radians(angle+90))])

	line_a = Line2D([p1[0],p2[0]], [p1[1], p2[1]],
	        lw=2, color='black', axes=ax)
	line_b = Line2D([p3[0],p4[0]], [p3[1], p4[1]],
	        lw=2, color='black', axes=ax)

	ax.add_line(line_a)
	ax.add_line(line_b)

	
ell  = Ellipse(gal_centre,width=2*rca,height=2*rcb,angle=angle,clip_box=ax.bbox,
			edgecolor="black",facecolor=None,fill=False,linewidth=2)
ax.add_artist(ell)

plt.xlabel('$l$ [deg]')
plt.ylabel('$b$ [deg]')

pdf.savefig(bbox_inches='tight')  # saves the current figure into a pdf page
plt.close()

#######################  ELLIPTICITIES ##############################
if exte == "Ell" or exte == "Seg":
	eps_rc     = np.array(map(Epsilon,samples[:,id_rc]))
	kde        = st.gaussian_kde(eps_rc)
	x          = np.linspace(0,1,num=100)
	epsrc_mode = x[kde(x).argmax()]
	epsrc_low  = brentq(lambda x:kde.integrate_box_1d(0,x)-0.16,a=0.0,b=1.0)
	epsrc_up   = brentq(lambda x:kde.integrate_box_1d(0,x)-0.84,a=0.0,b=1.0)
	qepsrc     = [epsrc_low,epsrc_mode,epsrc_up]
	nerc,bins,_= plt.hist(eps_rc,50, normed=1,
					ec="black",histtype='step', linestyle='solid',label="$\epsilon_{rc}$")
	plt.vlines(qepsrc,0,kde(qepsrc),colors="grey",alpha=0.5)
	plt.text(0.6,0.8*np.max(nerc),'$\epsilon_{rc}$=[%0.2f,%0.2f,%0.2f]' % ( tuple(qepsrc) ))
	epsilons = np.array([epsrc_mode])
	plt.ylim(0,1.1*np.max(nerc))

	if model in ["GKing","King","OGKing"]:
		eps_rt      = np.array(map(Epsilon,samples[:,[4,6]]))
		kde         = st.gaussian_kde(eps_rt)
		x           = np.linspace(0,1,num=1000)
		epsrt_mode  = x[kde(x).argmax()]
		epsrt_low  = brentq(lambda x:kde.integrate_box_1d(0,x)-0.16,a=0.0,b=1.0)
		epsrt_up   = brentq(lambda x:kde.integrate_box_1d(0,x)-0.84,a=0.0,b=1.0)
		qepsrt     = [epsrt_low,epsrt_mode,epsrt_up]

		nert,bins,_ = plt.hist(eps_rt,50, normed=1, range=[0,1], 
						ec="black",histtype='step', linestyle='dashed',label="$\epsilon_{rt}$")
		plt.vlines(qepsrt,0,kde(qepsrt),colors="grey",alpha=0.5,linestyle="dashed")
		plt.text(0.6,0.7*np.max(nerc),'$\epsilon_{rt}$=[%0.2f,%0.2f,%0.2f]' % ( tuple(qepsrt) ))
		epsilons = np.array([epsrc_mode,epsrt_mode]).reshape((1,2))
		plt.ylim(0,1.1*max([np.max(nert),np.max(nerc)]))

	#---------------------------
	plt.gca().xaxis.set_major_formatter(mtick.FormatStrFormatter('%.02f')) 
	plt.xlim(0,1)
	plt.xlabel('$\epsilon$')
	plt.legend()
	pdf.savefig(bbox_inches='tight')  # saves the current figure into a pdf page
	plt.close()

	MAP = np.append(MAP,epsilons)

#--------- computes the distribution of number of stars -----
if model in ["GKing","King","OGKing"]:
	Nrt = np.empty(len(samples))
	for s,par in enumerate(samples):
		Nrt[s] = mod.Number(par[idrt],par,Rmax,np.max(Nr))
	Nrt        = Nrt[np.where(Nrt < 1e4)[0]]
	kde        = st.gaussian_kde(Nrt)
	x          = np.linspace(np.min(Nrt),np.max(Nrt),num=1000)
	Nrt_mode   = np.array(x[kde(x).argmax()]).reshape((1,))
	bins       = np.linspace(1e3,1e4,num=50)   
	nrt,_,_ = plt.hist(Nrt,bins=bins, normed=1,ec="black",histtype='step')
	nrt = np.append(nrt,0)
	plt.ylabel("Density")
	plt.xlabel("Number of stars within $r_t$")
	plt.yscale("log")
	pdf.savefig(bbox_inches='tight')  # saves the current figure into a pdf page
	plt.close()

	# print(bins.shape)

	f_e = file(dir_out+"/"+model+"_numbers.txt", 'w')
	np.savetxt(f_e, Nrt_mode,fmt=str('%2.3f'),delimiter=" ")
	np.savetxt(f_e, np.c_[bins,nrt],fmt=str('%2.5e'),delimiter=" ")
	f_e.close()
#--------- computes the distribution of total mass -----
	bins          = np.linspace(0,1e4,num=100) 
	massrj        = MassRj(samples[:,idrt])
	massrj        = massrj[np.where(massrj < 1e4)[0]]
	kde           = st.gaussian_kde(massrj)
	x             = np.linspace(np.min(massrj),np.max(massrj),num=1000)
	massrj_mode   = np.array(x[kde(x).argmax()]).reshape((1,))

	nmr,_,_       = plt.hist(massrj,bins=bins, normed=1,ec="black",ls="dashed",histtype='step',label="$M_{rt}$")
	nmr           = np.append(nmr,0)
	mass_mode     = np.array([massrj_mode]).reshape((1,1))
	nms           = np.c_[bins,nmr]

	  
	if not exte == "Ctr":
		massep        = MassEps(samples[:,idrt],eps_rt)
		massep        = massep[np.where(massep < 1e4)[0]]
		kde           = st.gaussian_kde(massep)
		x             = np.linspace(np.min(massep),np.max(massep),num=1000)
		massep_mode   = np.array(x[kde(x).argmax()]).reshape((1,))
		nme,_,_       = plt.hist(massep,bins=bins, normed=1,ec="black",histtype='step',label="$M_{\epsilon}$")
		nme           = np.append(nme,0)
		mass_mode     = np.array([massrj_mode,massep_mode]).reshape((1,2))
		nms           = np.c_[bins,nmr,nme]

	plt.legend()
	plt.ylabel("Density")
	plt.xlabel("Total mass [$M_\odot$]")
	plt.yscale("log")
	# plt.ylim(1e-5,1)
	pdf.savefig(bbox_inches='tight')  # saves the current figure into a pdf page
	plt.close()

	f_e = file(dir_out+"/"+model+"_mass.txt", 'w')
	np.savetxt(f_e, mass_mode,fmt=str('%2.3f'),delimiter=" ")
	np.savetxt(f_e, nms,fmt=str('%2.5e'),delimiter=" ")
	f_e.close()

pdf.close()

############### Stores the MAP ##############
np.savetxt(dir_out+"/"+model+"_map.txt",MAP.reshape(1,len(MAP)),
		fmt=str('%2.3f'),delimiter="\t")

print("Take a look at the pdf files in "+dir_out) 


