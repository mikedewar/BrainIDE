import pylab as pb
import numpy as np
from ODE import *
import quickio


def ex_1():
	#-------------field--------------------
	field_width=40
	dimension=2
	spacestep=4
	f_space=pb.arange(-(field_width)/2.,(field_width)/2+spacestep,spacestep)# the step size should in a way that we have (0,0) in our kernel as the center
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

	#k_centers=[pb.matrix([[0],[0]])]
	#k_weights =[1]
	#k_widths=[pb.matrix([[8**2,0],[0,8**2]])]

	k_space=pb.arange(-(field_width/2.),((field_width/2.))+spacestep,spacestep)
	k=Kernel(k_weights,k_centers,k_widths,k_space,dimension)
	#------Field noise----------
	field_noise_variance=0.01
	beta_variance=1.3**2
	#-------Brain----------------
	alpha=100
	act_func=ActivationFunction(threshold=2,nu=20,beta=.8)
	#----------observations--------------------------
	Sensorwidth =1.2**2 #equals to 3mm
	S_obs= pb.linspace(-10,10,10)
	obs_locns=gen_spatial_lattice(S_obs)
	ny=len(obs_locns)
	widths=[pb.matrix([[Sensorwidth,0],[0,Sensorwidth]])]
	EEG_signals=Sensor(obs_locns,widths,dimension,f_space,ny,spacestep)
	obs_noise_covariance =.1*pb.matrix(np.eye(len(obs_locns),len(obs_locns)))
	# plotting
	plot_sensor = lambda c: circle(
		center = c, 
		radius = 2*pb.sqrt(pb.log(2),)*pb.sqrt(Sensorwidth)/2,
		color = 'b'
	)
	#[plot_sensor(cent) for cent in obs_locns]
	#pb.show()


	# -------Sampling properties-------------
	Fs = 1e3   #sampling rate                                       
	Ts = 1/Fs   #sampling period, second
	t_end = .2 # seconds
	NSamples = t_end*Fs;
	T = pb.linspace(0,t_end,NSamples);
	#--------------model and simulation------------------
	model=IDE(k,f_space,EEG_signals,act_func,alpha,field_noise_variance,beta_variance,obs_noise_covariance,spacestep,Ts)
	model.gen_ssmodel()
	V_matrix,V_filtered,Y=model.simulate(T)
	quickio.writed('V_matrix','w',V_matrix)
	quickio.writed('Y','w',Y)
	quickio.writed('V_filtered','w',V_filtered)

	#----------Field initialasation----------------------------
	#mean=[0]*spatial_location_num
	#initial_field_covariance=10*pb.eye(len(mean))
	#init_field=pb.matrix(pb.multivariate_normal(mean,initial_field_covariance)).T

if __name__ == "__main__":
	import profile
	profile.run('ex_1()')
	
	