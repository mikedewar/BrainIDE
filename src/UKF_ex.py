import pylab as pb
import numpy as np
from UKF import *

#-------------field--------------------
Massdensity=1#.5#2#1
field_basis_separation=2#.5#3#4#3
field_basis_width=2#1#2
field_width=5#2#10#20 use 5 for estimation trial
dimension=2
f_centers=field_centers(field_width,Massdensity,field_basis_separation,field_basis_width)
#f_centers=[pb.matrix([[0],[0]])]
nx=len(f_centers)
f_widths=[pb.matrix([[field_basis_width,0],[0,field_basis_width]])]*nx
f_weights=[1]*nx
#spacestep=1./Massdensity
spacestep=1


#f_space=pb.arange(-field_width/2.,(field_width/2.)+spacestep,spacestep)# the step size should in a way that we have (0,0) in our kernel as the center
f_space=pb.arange(-5,5+spacestep,spacestep)
f=Field(f_weights,f_centers,f_widths,dimension,f_space,nx,spacestep)
f.plot(f_centers)

#_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _Connectivity Kernel properties_ _ _ _ _ __ _ _ _ _ __ _ _ _ _ __ _ _ _ _ __ _ _ _ _ 
# Centers of the connectivity basis functions are placed at origin (0,0)                                                |
# Weights should be tunned to get stable kernel                                                                         |
#Spatial domain of the kernel is twice wider than the field for the fast simulation and the kernel must have center and |
#it must be located at (0,0), therefore len(k_space) must be odd.                                                       |
#_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _|

k_centers=[pb.matrix([[0],[0]]),pb.matrix([[0],[0]]),pb.matrix([[0],[0]])]
k_weights =[1e-5,-.8e-5,0.05e-5]
#k_weights =[10,-8,.5]
k_widths=[pb.matrix([[4**2,0],[0,4**2]]),pb.matrix([[6**2,0],[0,6**2]]),pb.matrix([[15**2,0],[0,15**2]])]
#k_space=pb.arange(-field_width,(field_width)+spacestep,spacestep)
k_space=pb.arange(-10,10+spacestep,spacestep)
k=Kernel(k_weights,k_centers,k_widths,k_space,dimension)
#k.plot(k_space)


#-------Brain----------------
alpha=100
field_noise_variance=800000
act_func=ActivationFunction(threshold=6,nu=1,beta=0.56)
#----------observations--------------------------
Sensorwidth =2#.36    # equals to 1mm 
SensorSpacing = 4*spacestep     # mm factor of spacestep
BoundryEffectWidth = .5 #mm 

#Sensorwidth = [3]     # equals to 1mm 
#SensorSpacing = 10     # mm factor of spacestep
#BoundryEffectWidth =4 #mm 

observation_centers=gen_obs_locations(field_width,Sensorwidth,SensorSpacing,BoundryEffectWidth)
#obs_locns =gen_obs_lattice(observation_centers)
obs_locns=f_centers
[circle(cent,Sensorwidth) for cent in obs_locns]
#pb.title('Sensors locations')
pb.show()
ny=len(obs_locns)
widths=[pb.matrix([[Sensorwidth,0],[0,Sensorwidth]])]
EEG_signals=EEG(obs_locns,widths,dimension,f_space,ny,spacestep)


obs_noise_covariance =.00001*pb.matrix(np.eye(len(obs_locns),len(obs_locns)))
	
#----------Field initialasation----------------------------
mean=[0]*nx
Initial_field_covariance=10*pb.eye(len(mean))
init_field=pb.matrix(pb.multivariate_normal(mean,Initial_field_covariance,[1])).T

# -------Sampling properties-------------
Fs = 1e3   #sampling rate                                       
Ts = 1/Fs   #sampling period, second
t_end = .2  # seconds

NSamples = t_end*Fs;
T = pb.linspace(0,t_end,NSamples);
	
#--------------model and simulation------------------
model=IDE(k,f,EEG_signals, act_func,alpha,field_noise_variance,obs_noise_covariance,Sensorwidth,obs_locns,spacestep,Ts)
model.gen_ssmodel() 
X,Y=model.simulate(init_field,T)
ukfilter=UKF(model)
#plot_field(X[2],model.field.fbases,f_space)








def observation_locations_shift(obs_locations,origin,spacestep):

	'''This is to find appropriate shifts to translate observation locations to spatial field origin
	in order to calculate the observation convolution 

	Arguments
	---------
	obs_locations: list of matrix
			each element is the 2x1 observation location 	
		the list contains first the outter centers and then the inner centers of the sensors:
		outter centers:the first element of the list is the bottom left corner then it goes to the top left corner (moving along the y-axis),
		 then increment x and again goes along the y-axis till it reaches the top right corner, then it goes exactly the same way
		for the inner centers of the observation locations.

	origin: matrix
			2x1 matrix containing the origin of the spatial field, bottom left corner of the spatial filed

	spacestep: float
			spatial step


	Returns
	---------
	list of matrix: 
		each 2x1 matrix is the appropriate shift needed to translate the observation location to the spatial field origin (bottom left corner)

	'''

	w=[(1./spacestep) *(obs_locations[i]-origin) for i in range(len(obs_locations))]
	return w

