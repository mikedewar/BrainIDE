import pylab as pb
import numpy as np
from scipy import io
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
field_basis_width=2.5#3.6
spacestep=1
S=pb.linspace(-10,10,11)
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
ps_estimate.itr_est(Y[4:],5)
print 'elapsed time is', time.time()-t0
#Saving Kernel and 1/synaptic time constant in mat format
alpha_mat={}
kernel_mat={}
alpha_mat['alpha']=ps_estimate.alpha_est 
kernel_mat['kernel']= pb.squeeze(ps_estimate.kernel_weights_est)
io.savemat('alpha',alpha_mat)
io.savemat('kernel',kernel_mat)
#Saving filtered states in mat format
X_f={}
X_f['X_f']= pb.squeeze(ps_estimate.Xhat).T  #each column is a state vector
io.savemat('X_f',X_f)
#Uncomment if you use smoother otherwise it gives error
#saving smooth state in mat format
#X_b={}
#X_b['var0']=pb.squeeze(ps_estimate.Xb).T
#io.savemat('X_b',X_b)








