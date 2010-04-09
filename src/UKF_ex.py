import pylab as pb
import numpy as np
from UKF import *

#-------------field--------------------
Massdensity=1
field_basis_width=12#3.6
field_width=20
dimension=2
spacestep=1
S=pb.linspace(-10,10,9)
#S=pb.arange(-10,10+2,2)
f_centers=gen_spatial_lattice(S)
nx=len(f_centers)
f_widths=[pb.matrix([[field_basis_width,0],[0,field_basis_width]])]*nx
f_weights=[1]*nx



f_space=pb.arange(-field_width/2.,(field_width/2.)+spacestep,spacestep)# the step size should in a way that we have (0,0) in our kernel as the center
f=Field(f_weights,f_centers,f_widths,dimension,f_space,nx,spacestep)


#_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _Connectivity Kernel properties_ _ _ _ _ __ _ _ _ _ __ _ _ _ _ __ _ _ _ _ __ _ _ _ _ 
# Centers of the connectivity basis functions are placed at origin (0,0)                                                |
# Weights should be tunned to get stable kernel                                                                         |
#Spatial domain of the kernel is twice wider than the field for the fast simulation and the kernel must have center and |
#it must be located at (0,0), therefore len(k_space) must be odd.                                                       |
#_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _|

k_centers=[pb.matrix([[0],[0]]),pb.matrix([[0],[0]]),pb.matrix([[0],[0]])]
k_weights =[1e1,-.8e1,0.05e1]
k_widths=[pb.matrix([[1.8**2,0],[0,1.8**2]]),pb.matrix([[2.4**2,0],[0,2.4**2]]),pb.matrix([[6**2,0],[0,6**2]])]
k_space=pb.arange(-field_width,(field_width)+spacestep,spacestep)
k=Kernel(k_weights,k_centers,k_widths,k_space,dimension)
#k.plot(k_space)
#------Field noise----------
field_noise_variance=0.01
field_cov_function=FieldCovarianceFunction(pb.matrix([0,0]),pb.matrix([[1.3**2,0],[0,1.3**2]]),2)
#-------Brain----------------
alpha=100
act_func=ActivationFunction(threshold=2,nu=20,beta=.8)
#----------observations--------------------------
Sensorwidth =1.2**2 #equals to 1mm #1.2**2 # equals to 1.99mm 
S_obs=pb.linspace(-10,10,10)
obs_locns=gen_spatial_lattice(S_obs)

#------------------------------------------------

l1=[circle(cent,2*pb.sqrt(pb.log(2),)*pb.sqrt(Sensorwidth)/2,'b') for cent in obs_locns]
l2=[circle(cent,2*pb.sqrt(pb.log(2),)*pb.sqrt(f_widths[0][0,0])/2,'r') for cent in f_centers]
pb.legend((l1, l2), ('r:field basis functions', 'b:Sensors'), loc='best')
pb.axis([-field_width/2.,field_width/2.,-field_width/2,field_width/2])
pb.show()
ny=len(obs_locns)
widths=[pb.matrix([[Sensorwidth,0],[0,Sensorwidth]])]
EEG_signals=EEG(obs_locns,widths,dimension,f_space,ny,spacestep)
obs_noise_covariance =.1*pb.matrix(np.eye(len(obs_locns),len(obs_locns)))
	
#----------Field initialasation----------------------------
mean=[0]*nx
initial_field_covariance=10*pb.eye(len(mean))
init_field=pb.matrix(pb.multivariate_normal(mean,initial_field_covariance,[1])).T

# -------Sampling properties-------------
Fs = 1e3   #sampling rate                                       
Ts = 1/Fs   #sampling period, second
t_end= 0.2  # seconds
NSamples = t_end*Fs;
T = pb.linspace(0,t_end,NSamples);
#--------------model and simulation------------------
model=IDE(k,f,EEG_signals, act_func,alpha,field_cov_function,field_noise_variance,obs_noise_covariance,Sensorwidth,obs_locns,spacestep,init_field,initial_field_covariance,Ts)
model.gen_ssmodel(sim=1)
#X,Y=model.simulate(T)
#ukfilter=ukf(model)
#plot_field(X[2],model.field.fbases,f_space)
#t0=time.time()
#xhat,phat=ukfilter._filter(Y)
#xhat,phat=ukfilter._filter(Y)
#xb,pb,xhat,phat=ukfilter.rtssmooth(Y)
#t0=time.time()
#xb,pb,xhat,phat=ukfilter.rtssmooth(Y)
#print 'elapsed time is', time.time()-t0
#ps_estimate=para_state_estimation(model)
#ps_estimate.itr_est(Y,1)
#print "Elapsed time in seconds is", time.time()-t0








