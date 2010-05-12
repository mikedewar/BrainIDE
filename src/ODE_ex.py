import pylab as pb
import numpy as np
import scipy
from ODE import *
import quickio
from scipy import io

#-------------field--------------------
field_width=40;                        # -20 to 20 mm, twice the estimated field, # must be EVEN!!!
observedfieldwidth = field_width/2;    # mm, -field_width/4 to field_width/4 
dimension=2;
spacestep=0.5;
steps_in_field = field_width/spacestep + 1;

Delta = 1./spacestep;
Nspacestep_in_observed_field = Delta*observedfieldwidth+1	

observation_offest = field_width/4;     # mm
observation_offset_units = observation_offest / spacestep -1;

f_space=pb.arange(-(field_width)/2.,(field_width)/2.+spacestep,spacestep)# the step size should in a way that we have (0,0) in our kernel as the center
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
#------Field noise----------
field_noise_variance=0.01
beta_variance=1.3**2
#-------Brain----------------
alpha=100
threshold=2
nu=20
beta=30
act_func=ActivationFunction(threshold,nu,beta)
#----------observations--------------------------
Delta_s = 1.5	# mm
Delta_s_units = Delta_s/spacestep	
nonsymmetric_obs_location_units = pb.arange(1,Nspacestep_in_observed_field,Delta_s_units)
offset = ((Nspacestep_in_observed_field - nonsymmetric_obs_location_units[-1])/2.)
symmetricobslocation_units = nonsymmetric_obs_location_units + offset + observation_offset_units

observation_locs_mm = symmetricobslocation_units*spacestep - field_width/2.
print observation_locs_mm

ny= (len(observation_locs_mm))**2;

Sensorwidth =0.9**2 #1.2**2 #equals to 1.5mm
S_obs= observation_locs_mm

obs_locns=gen_spatial_lattice(S_obs)
# ny=len(obs_locns)
widths=[pb.matrix([[Sensorwidth,0],[0,Sensorwidth]])]
EEG_signals=Sensor(obs_locns,widths,dimension,f_space,ny,spacestep)
obs_noise_covariance =.1*pb.matrix(np.eye(len(obs_locns),len(obs_locns)))
[circle(cent,2*pb.sqrt(pb.log(2),)*pb.sqrt(Sensorwidth)/2,'b') for cent in obs_locns]
pb.show()


# -------Sampling properties-------------
Fs = 1e3   #sampling rate                                       
Ts = 1/Fs   #sampling period, second
t_end = 1 # seconds
NSamples = t_end*Fs;
T = pb.linspace(0,t_end,NSamples);

#--------------model and simulation------------------
model=IDE(k,f_space,EEG_signals,act_func,alpha,field_noise_variance,beta_variance,obs_noise_covariance,spacestep,Ts)
model.gen_ssmodel()
V_matrix,V_filtered,Y=model.simulate(T)

quickio.writed('V_matrix','w',V_matrix)
quickio.writed('Y','w',Y)
quickio.writed('V_filtered','w',V_filtered)
#-------------save parameters---------------------------
parameters={}
parameters['field_width']=field_width
parameters['dimension']=dimension
parameters['spacestep']=spacestep
parameters['f_space']=f_space
parameters['k_centers']=k_centers
parameters['k_weights']=k_weights
parameters['k_widths']=k_widths
parameters['alpha']=alpha
parameters['field_noise_variance']=field_noise_variance
parameters['beta_variance']=beta_variance
parameters['threshold']=threshold
parameters['nu']=nu
parameters['beta']=beta
parameters['Sensorwidth']=Sensorwidth 
parameters['obs_locns']=obs_locns
parameters['Sensorwidth']=Sensorwidth
parameters['obs_noise_covariance']=obs_noise_covariance
parameters['Fs']=Fs
parameters['t_end']=t_end
quickio.writed('parameters','w',parameters)
#---------------save results in mat format---------------
V_matrix_dic={}
V_filtered_dic={}
Y_dic={}
V_matrix_dic['V_matrix']=V_matrix
V_filtered_dic['V_filtered']=V_filtered
Y_dic['Y']=Y
io.savemat('V_filtered',V_filtered_dic)
io.savemat('Y',Y_dic)
io.savemat('V_matrix',V_matrix_dic)

