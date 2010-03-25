import pylab as pb
import numpy as np
from ODE2 import *
import quickio
#-------------field--------------------
field_width=20#2#10#20 
dimension=2
spacestep=1
f_space=pb.arange(-(field_width)/2.,(field_width)/2+spacestep,spacestep)# the step size should in a way that we have (0,0) in our kernel as the center
#field_covariance_function=beta(pb.matrix([0,0]),pb.matrix([[.1**2,0],[0,.1**2]]),2)
beta_variance=3**2
spatial_location_num=(len(f_space))**2

#_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _Connectivity Kernel properties_ _ _ _ _ __ _ _ _ _ __ _ _ _ _ __ _ _ _ _ __ _ _ _ _ 
# Centers of the connectivity basis functions are placed at origin (0,0)                                                |
# Weights should be tunned to get stable kernel                                                                         |
#Spatial domain of the kernel is twice wider than the field for the fast simulation and the kernel must have center and |
#it must be located at (0,0), therefore len(k_space) must be odd.                                                       |
#_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _|
k_centers=[pb.matrix([[0],[0]]),pb.matrix([[0],[0]]),pb.matrix([[0],[0]])]
k_weights =[1e1,-.8e1,0.05e1]
k_widths=[pb.matrix([[1.8**2,0],[0,1.8**2]]),pb.matrix([[2.4**2,0],[0,2.4**2]]),pb.matrix([[6**2,0],[0,6**2]])]
k_space=pb.arange(-(field_width/2.),((field_width/2.))+spacestep,spacestep)
k=Kernel(k_weights,k_centers,k_widths,k_space,dimension)
#k.plot(k_space)


#-------Brain----------------
alpha=100
#for field_width=5, field_width=10 and field_width=20 :field_noise_variance=250000
field_noise_variance=0.01
#act_func=ActivationFunction(threshold=6,nu=1,beta=0.56)
act_func=ActivationFunction(threshold=2,nu=20,beta=.8)
#----------observations--------------------------
Sensorwidth =1.8**2 #equals to 3mm
SensorSpacing = 2.6*spacestep     # mm factor of spacestep
BoundryEffectWidth = 0 #mm 
#obs_locns=gen_spatial_lattice(-5,4,-5,4,4.7)
S= pb.linspace(-10,10,9)
#obs_locns=gen_obs_locations(field_width,Sensorwidth,SensorSpacing,BoundryEffectWidth)
obs_locns=gen_spatial_lattice(S)
ny=len(obs_locns)
widths=[pb.matrix([[Sensorwidth,0],[0,Sensorwidth]])]
EEG_signals=EEG(obs_locns,widths,dimension,f_space,ny,spacestep)

obs_noise_covariance =.1*pb.matrix(np.eye(len(obs_locns),len(obs_locns)))
[circle(cent,2*pb.sqrt(pb.log(2),)*pb.sqrt(Sensorwidth)/2,'b') for cent in obs_locns]
pb.show()
#----------Field initialasation----------------------------
#init_field=quickio.read('init_field')
#init_field=init_field['var0']
mean=[0]*spatial_location_num
initial_field_covariance=10*pb.eye(len(mean))
init_field=pb.matrix(pb.multivariate_normal(mean,initial_field_covariance)).T
#init_field_temp=pb.hstack(pb.split(init_field,pb.sqrt(spatial_location_num)))
#init_field_temp[:,0]=init_field_temp[:,-1]
#init_field_temp[:,1]=init_field_temp[:,-2]
#init_field_temp[0,:]=init_field_temp[-1,:]
#init_field_temp[1,:]=init_field_temp[-2,:]
#init_field=pb.matrix(pb.ravel(init_field_temp.T)).T
# -------Sampling properties-------------
Fs = 1e3   #sampling rate                                       
Ts = 1/Fs   #sampling period, second
t_end = .2 # seconds
NSamples = t_end*Fs;
T = pb.linspace(0,t_end,NSamples);
#Ts = 1
#T = range(0,201,);

#--------------model and simulation------------------
model=IDE(k,f_space,EEG_signals, act_func,alpha,field_noise_variance,beta_variance,obs_noise_covariance,spacestep,init_field,initial_field_covariance,Ts)
model.gen_ssmodel()
V_vector1,V_matrix1,Y1=model.simulate(T)
#anim_ode_field(V_matrix,f_space)








