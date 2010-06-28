#Author Parham Aram
#Date 28-06-2010
'''This module simulates the full neural model and uses the data
in the state space model to estimate the states, 
the connectivity kernel parameters and the synaptic dynamics'''

#Standard library imports
from __future__ import division
import pylab as pb
import numpy as np
import scipy as sp

#My modules
from Bases import *
from Data_generator import *
from IDE_Analytic import *

#space properties

dimension=2
#spatial step size
spacestep=0.5 # the step size should in a way that we have (0,0) in our kernel as the center
#field width
field_width=20
#spatial range
field_space=pb.arange(-(field_width)/2.,(field_width)/2.+spacestep,spacestep)

#time properties

#sampling rate  
Fs = 1e3                                       
#sampling period
Ts = 1/Fs   
t_end= 0.5 
NSamples = t_end*Fs;
T = pb.linspace(0,t_end,NSamples);

#observation and field basis function locations

#distance between sensors
Delta_s = 1.5 
observation_locs_mm=pb.arange(-field_width/2.+(spacestep/2.),field_width/2.-(spacestep/2.)+Delta_s,Delta_s)
print 'Sensor centers:',observation_locs_mm
#generate observation locations lattice
obs_locns=gen_spatial_lattice(observation_locs_mm)
#Define basis functions locations
NBasisFunction_x_or_y_dir=9
phi_centers_x_y=pb.linspace(-field_width/2.,field_width/2.,NBasisFunction_x_or_y_dir)
print 'field basis centers:',phi_centers_x_y
#generate field basis functions lattice
phi_centers=gen_spatial_lattice(phi_centers_x_y)


#Define connectivity kernel basis functions

#centers
psi0_center=pb.matrix([[0],[0]]);psi1_center=pb.matrix([[0],[0]]);psi2_center=pb.matrix([[0],[0]])
#widths
psi0_width=1.8**2;psi1_width=2.4**2;psi2_width=6.**2
#weights
psi0_weight=10;psi1_weight=-8;psi2_weight=0.5
#define each kernel basis functions
psi0=basis(psi0_center,psi0_width,dimension)
psi1=basis(psi1_center,psi1_width,dimension)
psi2=basis(psi2_center,psi2_width,dimension)

Connectivity_kernel_basis_functions=[psi0,psi1,psi2]
Connectivity_kernel_weights=[psi0_weight,psi1_weight,psi2_weight]
NF_Connectivity_kernel=NF_Kernel(Connectivity_kernel_basis_functions,Connectivity_kernel_weights)
IDE_Connectivity_kernel=IDE_Kernel(Connectivity_kernel_basis_functions,Connectivity_kernel_weights)

#Define basis functions

phi_widths=2.5 
#place field basis functions in n_x by 1 array
Phi=pb.array([basis(center,phi_widths,dimension) for center in phi_centers],ndmin=2).T
IDE_field=IDE_Field(Phi)

#Define field covariance function and observation noise

#Field noise
gamma_center=pb.matrix([[0],[0]])
gamma_width=1.3**2 
gamma_weight=0.1
gamma=basis(gamma_center,gamma_width,dimension)
#Observation noise
varepsilon=0.1
Sigma_varepsilon =varepsilon*pb.matrix(np.eye(len(obs_locns),len(obs_locns)))

#Define sensor geometry
sensor_center=pb.matrix([[0],[0]])
sensor_width=0.9**2 
sensor_kernel=basis(sensor_center,sensor_width,dimension)

#Define sigmoidal activation function and inverse synaptic time constant
fmax=10
v0=2
varsigma=0.8
act_fun=ActivationFunction(v0,fmax,varsigma)
#inverse synaptic time constant
zeta=100
#define field initialasation and number of iterations for estimation
mean=[0]*len(Phi)
P0=10*pb.eye(len(mean))
x0=pb.matrix(pb.multivariate_normal(mean,P0,[1])).T
number_of_iterations=10
#ignore first 100 observations allowing the model's initial transients to die out
First_n_observations=100

#populate the model
NF_model=NF(NF_Connectivity_kernel,sensor_kernel,obs_locns,observation_locs_mm,gamma,gamma_weight,Sigma_varepsilon,act_fun,zeta,Ts,field_space,spacestep)
IDE_model=IDE(IDE_Connectivity_kernel,IDE_field,sensor_kernel,obs_locns,gamma,gamma_weight,Sigma_varepsilon,act_fun,x0,P0,zeta,Ts,field_space,spacestep)
#generate the Neural Field model
NF_model.gen_ssmodel()
V,Y=NF_model.simulate(T)
#generate the reduced model (state space model)
IDE_model.gen_ssmodel()
#estimate the states, the connectivity kernel parameters and the synaptic dynamics
ps_estimate=para_state_estimation(IDE_model)
ps_estimate.itrerative_state_parameter_estimation(Y[First_n_observations:],number_of_iterations)
