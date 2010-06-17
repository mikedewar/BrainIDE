#Built-in modules
from __future__ import division
import pylab as pb
import numpy as np
import scipy as sp
from scipy import io
#My modules
from Bases import *
from Data_generator import *
from IDE_Analytic import *
import quickio

#simulation properties
#----------------------------------------------------------------------------------------------------------------------------------------------
dimension=2
#step size
simulation_field_spacestep=0.5 # the step size should in a way that we have (0,0) in our kernel as the center
estimation_field_spacestep=simulation_field_spacestep
#field width
simulation_field_width=20
estimation_field_width=simulation_field_width
#spatial range
simulation_field_space=pb.arange(-(simulation_field_width)/2.,(simulation_field_width)/2.+simulation_field_spacestep,simulation_field_spacestep)
estimation_field_space=pb.arange(-(estimation_field_width)/2.,(estimation_field_width)/2.+estimation_field_spacestep,estimation_field_spacestep)
# Define Sampling properties
Fs = 1e3   #sampling rate                                       
Ts = 1/Fs   #sampling period, second
t_end= 0.5 # seconds
NSamples = t_end*Fs;
T = pb.linspace(0,t_end,NSamples);
#-----------------------------------------------------------------------------------------------------------------------------------------------
#Define observation locations and basis function locations
Delta_s = 1.5 #mm distance between sensors in mm
#observation_locs_mm =observation_locns(2*simulation_field_spacestep,estimation_field_width,Delta_s)
observation_locs_mm =observation_locns(2*simulation_field_spacestep,20,Delta_s)
print 'Sensor centers:',observation_locs_mm
obs_locns=gen_spatial_lattice(observation_locs_mm)
#Define basis functions locations
NBasisFunction_x_or_y_dir=9
S=pb.linspace(-estimation_field_width/2.,estimation_field_width/2.,NBasisFunction_x_or_y_dir)
phi_centers=gen_spatial_lattice(S)
#------------------------------------------------------------------------------------------------------------------------------------------------
#Define connectivity kernel basis functions
#Connectivity kernel basis functions' centers
psi1_center=pb.matrix([[0],[0]])
psi2_center=pb.matrix([[0],[0]])
psi3_center=pb.matrix([[0],[0]])
#Kernel basis functions' widths
psi1_width=1.8**2
psi2_width=2.4**2
psi3_width=6.**2
#Kernel basis functions' widths
psi1_weight=10
psi2_weight=-8
psi3_weight=0.5
psi1=basis(psi1_center,psi1_width,dimension)
psi2=basis(psi2_center,psi2_width,dimension)
psi3=basis(psi3_center,psi3_width,dimension)
#create a list of connectivity kernel basis functions
Connectivity_kernel_basis_functions=[psi1,psi2,psi3]
Connectivity_kernel_weights=[psi1_weight,psi2_weight,psi3_weight]
ODE_Connectivity_kernel=ODE_Kernel(Connectivity_kernel_basis_functions,Connectivity_kernel_weights)
IDE_Connectivity_kernel=IDE_Kernel(Connectivity_kernel_basis_functions,Connectivity_kernel_weights)
#------------------------------------------------------------------------------------------------------------------------------------------------
#Define basis functions
phi_widths=2.5 #must be float
#place field basis functions in an array in the form of n_x*1
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
#Define sensor geometry
sensor_center=pb.matrix([[0],[0]])
sensor_width=0.9**2 
sensor_kernel=basis(sensor_center,sensor_width,dimension)
Sensor_vector=ODE_Sensor(obs_locns,sensor_width,dimension) #this is to calculate the sensors at each spatial locations
#plot sensors and basis functions
l1=[circle(cent,2*pb.sqrt(pb.log(2),)*pb.sqrt(sensor_width)/2,'b') for cent in obs_locns]
l2=[circle(cent,2*pb.sqrt(pb.log(2),)*pb.sqrt(phi_widths)/2,'r') for cent in phi_centers]
pb.axis([-estimation_field_width/2.,estimation_field_width/2.,-estimation_field_width/2,estimation_field_width/2])
pb.show()
#Define sigmoidal activation function and zeta
#------------------------------------
fmax=10
v0=2
varsigma=0.8
act_fun=ActivationFunction(v0,fmax,varsigma)
zeta=100
#----------Field initialasation and number of iterations for estimation----------------------------
mean=[0]*len(Phi)
P0=10*pb.eye(len(mean))
x0=pb.matrix(pb.multivariate_normal(mean,P0,[1])).T
number_of_iterations=10
#----------Ignoring first few observation
First_n_observations=100
#saving parameters
parameters={}
parameters['dimension']=dimension
parameters['simulation_field_spacestep']=simulation_field_spacestep
parameters['estimation_field_width']=estimation_field_width
parameters['obs_locns']=obs_locns
parameters['observation_locs_mm']=observation_locs_mm
parameters['psi1_center']=psi1_center
parameters['psi2_center']=psi2_center
parameters['psi3_center']=psi3_center
parameters['psi1_width']=psi1_width
parameters['psi2_width']=psi2_width
parameters['psi3_width']=psi3_width
parameters['psi1_weight']=psi1_weight
parameters['psi2_weight']=psi2_weight
parameters['psi3_weight']=psi3_weight
parameters['gamma_center']=gamma_center
parameters['gamma_width']=gamma_width
parameters['gamma_weight']=gamma_weight
parameters['varepsilon']=varepsilon
parameters['sensor_center']=sensor_center
parameters['sensor_width']=sensor_width
parameters['fmax']=fmax
parameters['v0']=v0
parameters['varsigma']=varsigma
parameters['zeta']=zeta
parameters['Fs']=Fs
parameters['t_end']=t_end
quickio.write('parameters',parameters,'w', overwrite=True)

#populate the model
ODE_model=ODE(ODE_Connectivity_kernel,sensor_kernel,Sensor_vector,obs_locns,observation_locs_mm,gamma,gamma_weight,Sigma_varepsilon,act_fun,zeta,Ts,simulation_field_space,simulation_field_spacestep)
IDE_model=IDE(IDE_Connectivity_kernel,IDE_field,sensor_kernel,obs_locns,gamma,gamma_weight,Sigma_varepsilon,act_fun,x0,P0,zeta,Ts,estimation_field_space,estimation_field_spacestep)
#Sigma_e_c=io.loadmat('Sigma_e_c01.mat')
#Sigma_e_c=Sigma_e_c['Sigma_e_c']
#ODE_model.Sigma_e_c=pb.matrix(Sigma_e_c)
ODE_model.gen_ssmodel()
V_matrix,Y=ODE_model.simulate(T)
#V_matrix_temp={};V_matrix_temp['V_matrix']=V_matrix
#Y_temp={};Y_temp['Y']=Y[First_n_observations:]
#quickio.write('V_matrix',V_matrix_temp,'w',overwrite=True)
#quickio.write('Y',Y_temp,'w',overwrite=True)
IDE_model.gen_ssmodel()
ps_estimate=para_state_estimation(IDE_model)
ps_estimate.itr_est(Y[First_n_observations:],number_of_iterations)
