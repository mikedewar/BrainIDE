#Author Parham Aram
#Date 21-11-2010
'''This module simulates the  neural model and uses the data
in the state space model to estimate the states, 
the connectivity kernel parameters and the synaptic dynamics'''

#Standard library imports
#~~~~~~~~~~~~~~~~~~~~~~~~
from __future__ import division
import pylab as pb
import numpy as np
import scipy as sp
#My modules
#~~~~~~~~~~~~~~~~~~~~
from Bases2D import *
from NF import *
from IDEModel import *
import IDEComponents
import ActivationFunction


#space properties
#~~~~~~~~~~~~~~~~
dimension=2
#spatial step size
spacestep=0.5 
#field width
field_width=20
#spatial range
field_space_x_y=pb.arange(-(field_width)/2.,(field_width)/2.+spacestep,spacestep)
#time properties
#~~~~~~~~~~~~~~~~
#sampling rate  
Fs = 1e3                                       
#sampling period
Ts = 1/Fs   
t_end= .5
NSamples = t_end*Fs
T = pb.linspace(0,t_end,NSamples)

#Define observation locations
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#distance between sensors
Delta_s = 1.5 
observation_locs_mm=pb.arange(-field_width/2.+(spacestep/2.),field_width/2.-(spacestep/2.)+Delta_s,Delta_s)
obs_locns_x,obs_locns_y=pb.meshgrid(observation_locs_mm,observation_locs_mm)
obs_locns=pb.array(zip(obs_locns_x.flatten(),obs_locns_y.flatten()))

#Define connectivity kernel basis functions
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#centers
psi0_center=pb.matrix([[0],[0]]);psi1_center=pb.matrix([[0],[0]]);psi2_center=pb.matrix([[0],[0]])
#widths
psi0_width=1.8**2;psi1_width=2.4**2;psi2_width=6.**2
#weights
psi0_weight=100;psi1_weight=-80;psi2_weight=5
#Define each kernel basis functions
psi0=basis(psi0_center,psi0_width,dimension)
psi1=basis(psi1_center,psi1_width,dimension)
psi2=basis(psi2_center,psi2_width,dimension)

NF_Connectivity_kernel_basis_functions=[psi0,psi1,psi2]
NF_Connectivity_kernel_weights=pb.array([psi0_weight,psi1_weight,psi2_weight])

IDE_Connectivity_kernel_basis_functions=[psi0,psi1,psi2]
IDE_Connectivity_kernel_weights=pb.array([psi0_weight,psi1_weight,psi2_weight])

NF_Connectivity_kernel=IDEComponents.Kernel(NF_Connectivity_kernel_basis_functions,NF_Connectivity_kernel_weights)
IDE_Connectivity_kernel=IDEComponents.Kernel(IDE_Connectivity_kernel_basis_functions,IDE_Connectivity_kernel_weights)

#Define basis functions locations
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
NBasisFunction_x_or_y_dir=9
S=pb.linspace(-field_width/2.,field_width/2.,NBasisFunction_x_or_y_dir)
phi_centers_x,phi_centers_y=pb.meshgrid(S,S)
phi_centers=zip(phi_centers_x.flatten(),phi_centers_y.flatten())

#Define basis functions
#~~~~~~~~~~~~~~~~~~~~~~
phi_widths=2.5  #must be float
Phi=pb.array([basis(center,phi_widths,dimension) for center in phi_centers],ndmin=2).T
field=IDEComponents.Field(Phi)

#Define field covariance function and observation noise
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Field noise
gamma_center=pb.matrix([[0],[0]])
gamma_width=1.3**2 
gamma_weight=0.1
gamma=basis(gamma_center,gamma_width,dimension)

#Observation noise
varepsilon=0.1
Sigma_varepsilon =varepsilon*np.eye(len(obs_locns),len(obs_locns))

#Define sensor geometry
sensor_center=pb.matrix([[0],[0]])
sensor_width=0.9**2 
sensor_kernel=basis(sensor_center,sensor_width,dimension)


#Define sigmoidal activation function and inverse synaptic time constant
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
fmax=1
v0=1.8
varsigma=0.56
act_fun=ActivationFunction.ActivationFunction(v0,fmax,varsigma)
#inverse synaptic time constant
zeta=100
#define field initialasation and number of iterations for estimation
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mean=[0]*len(Phi)
P0=10*pb.eye(len(mean))
x0=pb.multivariate_normal(mean,P0,[1]).T
number_of_iterations=10
#ignore first 100 observations allowing the model's initial transients to die out
First_n_observations=100
#populate the model
NF_model=NF(NF_Connectivity_kernel,sensor_kernel,obs_locns,gamma,gamma_weight,Sigma_varepsilon,act_fun,zeta,Ts,field_space_x_y,spacestep)
IDE_model=IDE(IDE_Connectivity_kernel,field,sensor_kernel,obs_locns,gamma,gamma_weight,Sigma_varepsilon,act_fun,x0,P0,zeta,Ts,field_space_x_y,spacestep)
#generate Neural Field model
NF_model.gen_ssmodel()
V,Y=NF_model.simulate(T)
#generate the reduced model (state space model)
IDE_model.gen_ssmodel()
#estimate the states, the connectivity kernel parameters and the synaptic dynamics
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
ps_estimate=para_state_estimation(IDE_model)
ps_estimate.itrerative_state_parameter_estimation(Y[First_n_observations:],number_of_iterations)

