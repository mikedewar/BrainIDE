from __future__ import division
import pylab as pb
import numpy as np
from scipy import io
from UKF import *
import quickio

#-------------reading parameters--------------------
parameters=quickio.read('parameters')
parameters=parameters['var0']
estimation_field_width=(parameters['field_width'])/2.     # need to change to estimation field width
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
field_basis_width=2.5			# sigma^2_b, width of field basis functions
spacestep=0.5					# space step for estimator
NBasisFunction_x_or_y_dir = 8	# the total number basis functions is this value squared (9=>81 basis functions)

# in this part of the code we make sure the basis functions are symetric with the spatial discretisation
steps_in_estimated_field = estimation_field_width/spacestep + 1;								#in space indexes
print "steps (indexes) in estimated field is", steps_in_estimated_field
Dist_between_basis_func = pb.floor(steps_in_estimated_field / (NBasisFunction_x_or_y_dir - 1) ) 		#in space indexes
print "distance between basis functions (in space index) is", Dist_between_basis_func
twice_offset = steps_in_estimated_field - Dist_between_basis_func * (NBasisFunction_x_or_y_dir-1) #in space indexes
basis_func_offset = twice_offset/2.

# we need to make sure that twice_offset/2 is an integer
# if twice_offset/2 is not an integer the basis functions will be assymetrically placed in the field we are trying to estimate
print "offset (in index) = " ,basis_func_offset
assert pb.mod(basis_func_offset,.5)==0, "offset for positioning basis functions not an int. Choose at different number of basis functions"
  
print "basis function offset ok"
S_index = pb.arange(basis_func_offset,steps_in_estimated_field-basis_func_offset+Dist_between_basis_func,Dist_between_basis_func)
S = (S_index - steps_in_estimated_field/2.) * spacestep
# S=pb.linspace(-estimation_field_width/2+basis_func_offset*spacestep,estimation_field_width/2-basis_func_offset*spacestep,NBasisFunction_x_or_y_dir)			# location of field basis functions

print "basis function indexes:", S_index
print "basis function centers:", S
# reset the edge of the estimated field such that the edge cuts a basis function in half
# estimation_field_width = estimation_field_width - twice_offset*spacestep

f_centers=gen_spatial_lattice(S)
nx=len(f_centers)
f_widths=[pb.matrix([[field_basis_width,0],[0,field_basis_width]])]*nx
f_weights=[1]*nx
f_space=pb.arange(-estimation_field_width/2.,(estimation_field_width/2.)+spacestep,spacestep)# the step size should in a way that we have (0,0) in our kernel as the center
f=Field(f_weights,f_centers,f_widths,dimension,f_space,nx,spacestep)


#_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _Connectivity Kernel properties_ _ _ _ _ __ _ _ _ _ __ _ _ _ _ __ _ _ _ _ __ _ _ _ _ 
# Centers of the connectivity basis functions are placed at origin (0,0)                                                |
# Weights should be tunned to get stable kernel                                                                         |
#Spatial domain of the kernel is twice wider than the field for the fast simulation and the kernel must have center and |
#it must be located at (0,0), therefore len(k_space) must be odd.                                                       |
#_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _|
k_space=pb.arange(-estimation_field_width,(estimation_field_width)+spacestep,spacestep)
k=Kernel(k_weights,k_centers,k_widths,k_space,dimension)
#k.plot(k_space)
#------Field noise----------
field_cov_function=FieldCovarianceFunction(pb.matrix([0,0]),pb.matrix([[beta_variance,0],[0,beta_variance]]),2)
#-------Brain----------------
act_func=ActivationFunction(threshold,nu,beta)
#----------observations--------------------------
l1=[circle(cent,2*pb.sqrt(pb.log(2),)*pb.sqrt(Sensorwidth)/2,'b') for cent in obs_locns]
l2=[circle(cent,2*pb.sqrt(pb.log(2),)*pb.sqrt(f_widths[0][0,0])/2,'r') for cent in f_centers]
# pb.legend((l1, l2), ('r:field basis functions', 'b:Sensors'), loc='best')
pb.axis([-estimation_field_width/2.,estimation_field_width/2.,-estimation_field_width/2,estimation_field_width/2])
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
#absolute value of the estimation field starting point 
Estimated_field_startingpoint=10

#--------------model and simulation------------------
model=IDE(k,f,EEG_signals, act_func,alpha,field_cov_function,field_noise_variance,obs_noise_covariance,Sensorwidth,obs_locns,spacestep,init_field,initial_field_covariance,Ts,Estimated_field_startingpoint)
model.gen_ssmodel(sim=1)
Y=quickio.read('Y')
Y=Y['var0']
t0=time.time()
ps_estimate=para_state_estimation(model)
UKF_iterations = 20;
start_sample = 4;
ps_estimate.itr_est(Y[start_sample:],UKF_iterations)

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








