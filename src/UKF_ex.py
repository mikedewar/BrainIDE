import pylab as pb
import numpy as np
from UKF import *

#-------------field--------------------
Massdensity=1#.5#2#1
field_basis_separation=2#.5#3#4#3
field_basis_width=1#1#2
field_width=5#2#10#20 
dimension=2
f_centers=field_centers(field_width,Massdensity,field_basis_separation,field_basis_width)
#f_centers=[pb.matrix([[0],[0]])]
nx=len(f_centers)
f_widths=[pb.matrix([[field_basis_width,0],[0,field_basis_width]])]*nx
f_weights=[1]*nx
#spacestep=1./Massdensity
spacestep=1


#f_space=pb.arange(-field_width/2.,(field_width/2.)+spacestep,spacestep)# the step size should in a way that we have (0,0) in our kernel as the center
#for field_width=10 take the f_space= -8,8
f_space=pb.arange(-5,5+spacestep,spacestep) #for field_width=5
#f_space=pb.arange(-8,8+spacestep,spacestep) #for field_width=10 
#f_space=pb.arange(-12,12+spacestep,spacestep) #for field_width=20 
#f_space=pb.arange(-23,23+spacestep,spacestep) #for field_width=40 

#f_space=pb.arange(-5,5+spacestep,spacestep)#field_width=20,time-1.5 hrs
f=Field(f_weights,f_centers,f_widths,dimension,f_space,nx,spacestep)
f.plot(f_centers)

#_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _Connectivity Kernel properties_ _ _ _ _ __ _ _ _ _ __ _ _ _ _ __ _ _ _ _ __ _ _ _ _ 
# Centers of the connectivity basis functions are placed at origin (0,0)                                                |
# Weights should be tunned to get stable kernel                                                                         |
#Spatial domain of the kernel is twice wider than the field for the fast simulation and the kernel must have center and |
#it must be located at (0,0), therefore len(k_space) must be odd.                                                       |
#_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _|

k_centers=[pb.matrix([[0],[0]]),pb.matrix([[0],[0]]),pb.matrix([[0],[0]])]
#k_centers=[pb.matrix([[0],[0]])]
#k_weights =[2.5]
#k_widths=[pb.matrix([[4**2,0],[0,4**2]])]
k_weights =[1e1,-.8e1,0.05e1]
#k_weights =[10,-8,.5]
k_widths=[pb.matrix([[4**2,0],[0,4**2]]),pb.matrix([[6**2,0],[0,6**2]]),pb.matrix([[15**2,0],[0,15**2]])]
#k_space=pb.arange(-field_width,(field_width)+spacestep,spacestep)
k_space=pb.arange(-10,10+spacestep,spacestep)
k=Kernel(k_weights,k_centers,k_widths,k_space,dimension)
#k.plot(k_space)
#------Field noise----------
field_noise_variance=150000
field_cov_function=FieldCovarianceFunction(pb.matrix([0,0]),pb.matrix([[.1,0],[0,.1]]),2)


#-------Brain----------------
alpha=100
#for field_width=5, field_width=10 and field_width=20 :field_noise_variance=250000

act_func=ActivationFunction(threshold=0.25,nu=20,beta=.8)
#act_func=ActivationFunction(threshold=6,nu=1,beta=0.56)
#for the different field decomposition keep the kernel, beta, nu as it is and play with the threshold.
#nu=10 and threshold=0.25 for field_width=5
#nu=10 and threshold=2,beta=.8 for field_width=10
#nu=10 and threshold=2,beta=.8 for field_width=20

#act_func=ActivationFunction(threshold=2,nu=10,beta=.8)
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
#[circle(cent,Sensorwidth) for cent in obs_locns]
#pb.title('Sensors locations')
#pb.show()
ny=len(obs_locns)
widths=[pb.matrix([[Sensorwidth,0],[0,Sensorwidth]])]
EEG_signals=EEG(obs_locns,widths,dimension,f_space,ny,spacestep)

#for field_width=5, field_width=10 and field_width=20 :obs_noise_covariance =.00001*pb.matrix(np.eye(len(obs_locns),len(obs_locns)))

obs_noise_covariance =.0001*pb.matrix(np.eye(len(obs_locns),len(obs_locns)))
	
#----------Field initialasation----------------------------
mean=[0]*nx
initial_field_covariance=10*pb.eye(len(mean))
init_field=pb.matrix(pb.multivariate_normal(mean,initial_field_covariance,[1])).T

# -------Sampling properties-------------
Fs = 1e3   #sampling rate                                       
Ts = 1/Fs   #sampling period, second
t_end = .05  # seconds
NSamples = t_end*Fs;
T = pb.linspace(0,t_end,NSamples);
#Ts = 1
#T = range(0,201,);
	
#--------------model and simulation------------------
model=IDE(k,f,EEG_signals, act_func,alpha,field_cov_function,field_noise_variance,obs_noise_covariance,Sensorwidth,obs_locns,spacestep,init_field,initial_field_covariance,Ts)
X,Y=model.simulate(T)
ukfilter=ukf(model)
#plot_field(X[2],model.field.fbases,f_space)
#t0=time.time()
#xhat,phat=ukfilter._filter(Y)
#xhat,phat=ukfilter._filter(Y)
#xb,pb,xhat,phat=ukfilter.rtssmooth(Y)
t0=time.time()
#xb,pb,xhat,phat=ukfilter.rtssmooth(Y)
#print 'elapsed time is', time.time()-t0
ps_estimate=para_state_estimation(model)
ps_estimate.itr_est(Y,1)
print "Elapsed time in seconds is", time.time()-t0








