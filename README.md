# Master-s-Thesis

Hi everyone,

I would like to share my programming code on the Python of my final project to get a Master of Science on Advanced Astrophysics from Institut Teknologi Bandung. That is about "The Utilization of Bayesian Framework in Determining Projected Spatial Distributions of Star Clusters".

The aims are:
1. Modeling various spatial density distributions of globular and open clusters using Bayesian Evidence (Marginal Likelihood).
2. Calculating the posterior model ratio (Bayes factor) using a Bayesian Framework and determine the best model of spatial density distribution the Globular and Open Clusters.
3. Determine the total mass and predict the mass segregation of Globular and Open Clusters.


To execute the code, use the following syntaxis:

python runSpatial.py arg1 arg2 [arg3]

with arg1 being the model (profile), arg2 the maximum radius coverage (in pc), and arg3 the extension to the model.

Possible models are:
- EFF : the Elson et al. 1987
- GDP : similar to that of Lauer et al. (1995)
- King: classical King 1962 profile
- GKing : introduced in Olivares et al. (2017b)
- OGKing : the optimised GKing 
- RGDP : the restricted GDP

Possible extensions are:
- Ctr : the centre of the cluster is inferred as parameter
- Ell : in addition to the centre, the models have ellipticity
- Seg : in addition to the centre and ellipticity the models are luminosity segregated.


## Requirements
In order to run this code you need the following libraries:
- pandas
- numpy
- scipy
- pymultinest
- importlib
- corner
- matplotlib

##Disclaimer
The code from Javier Olivares has been tested and improved.
Everyone now can use this code for the other star clusters. 

If you experience crashes, please let us know at yudarhd@gmail.com or javier.olivares@univ-grenoble-alpes.fr

Happy Coding!
