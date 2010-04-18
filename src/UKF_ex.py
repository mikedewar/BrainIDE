import pylab as pb
import numpy as np
from UKF import *
import quickio
#-------------reading parameters--------------------
parameters=quickio.read('parameters')
parameters=parameters['var0']

field_width=(parameters['field_width'])/2.
dimension=parameters['dimension']
k_centers=parameters['k_centers']
k_weights=parameters['k_weights']
k_widths=parameters['k_widths']
alpha=parameters['alpha']
field_noise_variance=parameters['field_noise_variance']
beta_variance=parameters['beta_variance']
threshold=parameters['threshold']
nu=parameters['nu']
beta=parameters['beta']
Sensorwidth=parameters['Sensorwidth']
obs_locns=parameters['obs_locns']
obs_noise_covariance=parameters['obs_noise_covariance']
Fs=parameters['Fs']
t_end=parameters['t_end']
#-------------field--------------------
field_basis_width=3.6
spacestep=1
S=pb.linspace(-10,10,9)
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
k_space=pb.arange(-field_width,(field_width)+spacestep,spacestep)
k=Kernel(k_weights,k_centers,k_widths,k_space,dimension)
#k.plot(k_space)
#------Field noise----------
field_cov_function=FieldCovarianceFunction(pb.matrix([0,0]),pb.matrix([[beta_variance,0],[0,beta_variance]]),2)
#-------Brain----------------
act_func=ActivationFunction(threshold,nu,beta)
#----------observations--------------------------
l1=[circle(cent,2*pb.sqrt(pb.log(2),)*pb.sqrt(Sensorwidth)/2,'b') for cent in obs_locns]
l2=[circle(cent,2*pb.sqrt(pb.log(2),)*pb.sqrt(f_widths[0][0,0])/2,'r') for cent in f_centers]
pb.legend((l1, l2), ('r:field basis functions', 'b:Sensors'), loc='best')
pb.axis([-field_width/2.,field_width/2.,-field_width/2,field_width/2])
pb.show()
ny=len(obs_locns)
widths=[pb.matrix([[Sensorwidth,0],[0,Sensorwidth]])]
EEG_signals=EEG(obs_locns,widths,dimension,f_space,ny,spacestep)

#----------Field initialasation----------------------------
mean=[0]*nx
initial_field_covariance=10*pb.eye(len(mean))
init_field=pb.matrix(pb.multivariate_normal(mean,initial_field_covariance,[1])).T

# -------Sampling properties-------------
Ts = 1/Fs   #sampling period, second
NSamples = t_end*Fs;
T = pb.linspace(0,t_end,NSamples);
#--------------model and simulation------------------
model=IDE(k,f,EEG_signals, act_func,alpha,field_cov_function,field_noise_variance,obs_noise_covariance,Sensorwidth,obs_locns,spacestep,init_field,initial_field_covariance,Ts)
model.gen_ssmodel(sim=1)
Y=quickio.read('Y')
Y=Y['var0']
t0=time.time()
ps_estimate=para_state_estimation(model)
ps_estimate.itr_est(Y,5)
print 'elapsed time is', time.time()-t0
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








