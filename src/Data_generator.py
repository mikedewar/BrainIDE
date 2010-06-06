#Built-in modules
from __future__ import division
import pylab as pb
import numpy as np
import time
import os
from scipy import signal
import scipy as sp
from Bases import basis


class ODE():

	"""class defining a non-linear, Integro-Difference, discrete-time state space model.

	Arguments
	----------
	kernel: 
		IDE kernel.
	field : 
		spatial field.
	act_fun : 
		Sigmoidal activation function.
	alpha : Integer
		alpha = 1/tau, post-synaptic time constant, (Wendling, 2002, 100 for excitatory, 500 for inhibitory), Schiff = 3

	beta : field covariance function

	field_noise_variance: float
		The process noise variance, which drives the dynamic system
	obs_noise_covariance: matrix
		Observation noise covariance
	Sensorwidth: float
		Sensor diameter in mm
	obs_locns: List of matrix
		Sensors locations
	spacestep: float
		Spatial step size for descritization in mm
	Ts: float
		sampling time, seconds
	Attributes
	----------
	"""

	def __init__(self,kernel,sensor_kernel,Sensor_vector,obs_locns,observation_locs_mm,gamma,gamma_weight,Sigma_varepsilon,act_fun,zeta,Ts,simulation_field_space,simulation_field_spacestep):

		self.kernel = kernel
		self.sensor_kernel=sensor_kernel
		self.Sensor_vector=Sensor_vector
		self.obs_locns=obs_locns
		self.observation_locs_mm=observation_locs_mm
		self.gamma=gamma
		self.gamma_weight=gamma_weight
		self.Sigma_varepsilon=Sigma_varepsilon
		self.act_fun = act_fun
		self.zeta=zeta
		self.Ts=Ts
		self.simulation_field_spacestep=simulation_field_spacestep
		self.simulation_field_space=simulation_field_space
		self.ny=len(self.obs_locns)
		self.xi=1-(self.Ts*self.zeta)

	def gen_ssmodel(self):

		"""
		generates nonlinear IDE
		Arguments
		---------
		if sim=1 it generates K (sum of kernels which is used in simulation), the reason is finding K is faster
		    than finding each kernel individually and for the simulation we dont need to find each kernel individually
		Returns
		----------
		K: ndarray
			matrix of kernel values over the spatial domain of the kernel

		Sw: matrix
			covariance of the field
		Swc: matrix
			cholesky decomposiotion of Sw
		Svc: matrix
			cholesky decomposiotion of the observation noise covariance
			

		"""
		print "generating nonlinear IDE"

		t_K=time.time()
		K = pb.empty((len(self.simulation_field_space),len(self.simulation_field_space)))
		#S=pb.empty((len(self.simulation_field_space),len(self.simulation_field_space)))
		#sensors_values_temp=[]
		for i in range(len(self.simulation_field_space)):
			for j in range(len(self.simulation_field_space)):
				K[i,j]=self.kernel(pb.matrix([[self.simulation_field_space[i]],[self.simulation_field_space[j]]]))
				#sensors_values_temp.append(self.Sensor_vector(pb.matrix([[self.simulation_field_space[i]],[self.simulation_field_space[j]]])))
				#S[i,j]=self.sensor_kernel(pb.matrix([[self.simulation_field_space[i]],[self.simulation_field_space[j]]]))

		self.K=pb.matrix(K)  #K is matrix
		#self.S=pb.matrix(S)
		#we squeeze the result to have a matrix in a form of
		#[m1(s1) m1(s2) ...m1(sl);m2(s1) m2(s2) ... m2(sl);...;mny(s1) mny(s2) ... mny(sl)]
		#where sl is the number of spatial points after discritization
		#self.Sensors_vector_values=pb.matrix(pb.squeeze(sensors_values_temp).T)
		print 'elapsed time to generate the Kernel matrix in seconds is',time.time()-t_K


		#Calculating Sw
		print "calculating Covariance matrix"
		t0=time.time()
		gamma_space=pb.empty((self.simulation_field_space.size**2,2),dtype=float)
		l=0
		for i in self.simulation_field_space:
			for j in self.simulation_field_space:
				gamma_space[l]=[i,j]
				l+=1
		N1,D1 = gamma_space.shape
		diff = gamma_space.reshape(N1,1,D1) - gamma_space.reshape(1,N1,D1)
		Sigma_e_temp=self.gamma_weight*np.exp(-np.sum(np.square(diff),-1)*(1./self.gamma.width))
		self.Sigma_e=pb.matrix(Sigma_e_temp)
		print "Elapsed time to calculate Sigma_e in seconds is", time.time()-t0
		t0=time.time()

		if hasattr(self,'Sigma_e_c'):
			pass
		else:
			self.Sigma_e_c=pb.matrix(sp.linalg.cholesky(self.Sigma_e)).T
			print "Elapsed time to find cholesky of Sigma_e in seconds is", time.time()-t0

		Sigma_varepsilon_c=pb.matrix(sp.linalg.cholesky(self.Sigma_varepsilon)).T
		self.Sigma_varepsilon_c=Sigma_varepsilon_c

		#Calculating observation matrix
		print "calculating sensors vector"
		t0=time.time()
		sensor_space=pb.empty((self.observation_locs_mm.size**2,2),dtype=float)
		l=0
		for i in self.observation_locs_mm:
			for j in self.observation_locs_mm:
				sensor_space[l]=[i,j]
				l+=1
		N2,D2 = sensor_space.shape
		diff = sensor_space.reshape(N2,1,D2) - gamma_space.reshape(1,N1,D1)
		#diff is in the form of: n_y by number of spatial points
		#[(r'_1-r_1) (r'_1-r_2)...(r'_1-r_s);(r'_2-r_1) (r'_2-r_2)...(r'_2-r_s);(r'_ny-r_1) (r'_ny-r_2)...(r'_ny-r_s)] s:number of spatial points
		# Here we find the gaussian at each point in diff, sum(np.square(r'_1-r_1)) is equivalent to (r'_1-r_1).T* (r'_1-r_1)
		C=np.exp(-np.sum(np.square(diff),-1)*(1./self.sensor_kernel.width))
		self.C=pb.matrix(C)
		print 'elapsed time to generate  the sensors vector in seconds is',time.time()-t0




	def simulate(self,T):

		Y=[]
		V_matrix=[]  #I don't save initial field


		#*********This just works for the stepsize of factor of 0.5 because of the floating point problem
		#width_temp=self.estimation_field_width/2. #Find the half width of the estimation width
		#starting_index=pb.where(self.simulation_field_space==-width_temp)[0][0] #finds the index of the starting point
		#ending_index=pb.where(self.simulation_field_space==width_temp)[0][0]    #finds the index of the ending point


		spatial_location_num=(len(self.simulation_field_space))**2
		sim_field_space_len=len(self.simulation_field_space) #length of simulation field space
		v0=self.Sigma_e_c*pb.matrix(np.random.randn(spatial_location_num,1)) #initial field
		v_membrane=pb.reshape(v0,(sim_field_space_len,sim_field_space_len))
		#v_membrane is len(self.simulation_field_space) by len(self.simulation_field_space) matrix

		spiking_rate=self.act_fun.fmax/(1.+pb.exp(self.act_fun.varsigma*(self.act_fun.v0-v_membrane)))
		#spiking rate is len(self.simulation_field_space) by len(self.simulation_field_space) matrix

		for t in T[1:]:

			v = self.Sigma_varepsilon_c*pb.matrix(np.random.randn(len(self.obs_locns),1)) #ny by one matrix
			w = pb.reshape(self.Sigma_e_c*pb.matrix(np.random.randn(spatial_location_num,1)),(sim_field_space_len,sim_field_space_len)) #w is len(self.simulation_field_space) by len(self.simulation_field_space) matrix
			print "simulation at time",t

			g=signal.fftconvolve(self.K,spiking_rate,mode='same') 
			# g (average_action_potentials) is len(self.simulation_field_space) by len(self.simulation_field_space) ndarray
			g*=(self.simulation_field_spacestep**2)
			v_membrane=self.Ts*pb.matrix(g) +self.xi*v_membrane+w
			spiking_rate=self.act_fun.fmax/(1.+pb.exp(self.act_fun.varsigma*(self.act_fun.v0-v_membrane)))


			#Observation
			Y.append((self.simulation_field_spacestep**2)*(self.C*pb.reshape(v_membrane,(sim_field_space_len**2,1)))+v)

			#Free Boundary Condition
			# we simulate over a bigger field to avoid boundary conditions and then we look at the middle part of
			# the field
			
			#v_membrane_estimation=v_membrane[starting_index:ending_index+1,starting_index:ending_index+1]
			#V_matrix.append(v_membrane_estimation)
			V_matrix.append(v_membrane)

		return V_matrix,Y


class ODE_Kernel():

	"""class defining the Connectivity kernel of the brain.

	Arguments
	----------

	Psi:list
		list of connectivity kernel basis functions; must be of class Bases
	weights: list
		list of connectivity kernel weights	


	Attributes
	----------
	__call__: 
		evaluate the kernel at a given spatial location
	"""
	
	def __init__(self,Psi,weights):
		
		self.Psi=Psi
		self.weights = weights

	def __call__(self,s):

		"""
		evaluates the kernel at a given spatial location

		Arguments
		----------
		s : matrix
			spatail location, it must be in a form of dimension x 1

		"""
		return self.weights[0]*self.Psi[0](s)+self.weights[1]*self.Psi[1](s)+self.weights[2]*self.Psi[2](s)


class ODE_Sensor():

	def __init__(self,centers,width,dimension):
		
		
		self.centers=centers
		self.width=width
		self.dimension=dimension

	def __call__(self,s):

		"""
		generates vector of sensors at a given spatial location

		Arguments
		----------
		s : float
			spatail location
		Returns
		----------
		vector of observation kernels: matrix
			matrix of ny x 1 dimension [m_1(s) m_2(s) ... m_ny(s)].T
		"""
		return pb.matrix([basis(cen,self.width,self.dimension)(s) for cen in self.centers]).T


class ActivationFunction():

	"""class defining the sigmoidal activation function .

	Arguments
	----------
	v0: float
		firing threshold, mV                  (Wendling, 2002, 6 mV), (Schiff, 2007, threshold = 0.24 (Heaviside))
	fmax: float
		maximum firing rate, spikes/s         (Wendling, 2002, 2*e0 = 5, or nu = 5
	varsigma: float
		slope of sigmoid, spikes/mV           (Wendling, 2002, 0.56 mV^-1)

	v: float
		Presynaptic potential

	Returns
	----------
	average firing rate
	"""	
	def __init__(self,v0,fmax,varsigma):
		
		self.v0 = v0
		self.fmax = fmax
		self.varsigma = varsigma
		
	def __call__(self,v):

		return float(self.fmax/(1.+pb.exp(self.varsigma*(self.v0-v))))


	def plot(self,plot_range):
		u=pb.linspace(-plot_range,plot_range,1000)
		z=pb.zeros_like(u)
		for i,j in enumerate(u):
			z[i]=self(j)
		pb.plot(u,z)
		pb.show()





def	gen_spatial_lattice(S):
	"""generates a list of vectors, where each vector is a co-ordinate in a lattice"""
	number_of_points = len(S)**2

	space = [0 for i in range(number_of_points)]
	count = 0
	for i in S:
		for j in S:
			space[count] = pb.matrix([[i],[j]])
			count += 1
	return space



def observation_locns(spacestep,estimation_field_width,Delta_s):
	'''Define the center of sensors along x and y
		
		Atguments:
		----------
			spacestep: the spatial step in the simulated field
			observedwidthfield: the width of the observed field
			Delta_s: distance between sensors in mm

		Returns
		-------
			observation_locns_mm:
				the observation location along x or y directions in mm'''
		

	steps_in_field = (2*estimation_field_width)/spacestep + 1;
	inv_spacestep = 1./spacestep;						
	Nspacestep_in_observed_field = inv_spacestep*estimation_field_width+1	

	observation_offest = estimation_field_width/2;     # mm
	observation_offset_units = observation_offest / spacestep -1;
	field_space=pb.arange(-estimation_field_width,estimation_field_width+spacestep,spacestep)
	spatial_location_num=(len(field_space))**2
	Delta_s_units = Delta_s/spacestep	
	nonsymmetric_obs_location_units = pb.arange(1,Nspacestep_in_observed_field,Delta_s_units)
	offset = ((Nspacestep_in_observed_field - nonsymmetric_obs_location_units[-1])/2.)
	symmetricobslocation_units = nonsymmetric_obs_location_units + offset + observation_offset_units

	observation_locs_mm = symmetricobslocation_units*spacestep - estimation_field_width
	return observation_locs_mm


def circle(center,radius,color):
	"""
		plot a circle with given center and radius

		Arguments
		----------
		center : matrix or ndarray
			it should be 2x1 ndarray or matrix
		radius:  float
			masses per mm
	"""
	u=pb.linspace(0,2*np.pi,200)
	x0=pb.zeros_like(u)
	y0=pb.zeros_like(u)
	for i,j in enumerate(u):
		x0[i]=radius*pb.sin(j)+center[0,0]
		y0[i]=radius*pb.cos(j)+center[1,0]
	pb.plot(x0,y0,color)



def avi_field(V_matrix,f_space,spacestep,filename,play=0):

	"""
		creat avi for the neural field

		Arguments
		----------
		V_matrix: list of matrix
			field at each time 

		f_space: array
			x,y of the spatial locations

		filename: 	
		Returns
		---------
		avi of the neural field

	"""
	fig_hndl =pb.figure()
	files=[]
	filename=" "+filename


	for t in range(len(V_matrix)):

		pb.figure()	
		pb.imshow(V_matrix[t],animated=True,origin='lower',aspect=1, alpha=1,extent=[min(f_space),max(f_space),min(f_space),max(f_space)])
		#pb.imshow(pb.hstack([V_matrix[t]]*2),animated=True,origin='lower',aspect=1, alpha=1,extent=[min(f_space),max(f_space),min(f_space),max(f_space)])

		pb.colorbar()

		fname = '_tmp%05d.jpg'%t
		pb.savefig(fname,format='jpg')
		pb.close()
		files.append(fname)
	os.system("ffmpeg -r 5 -i _tmp%05d.jpg -y -an"+filename+".avi")
	# cleanup
	for fname in files: os.remove(fname)
	if play: os.system("vlc"+filename+".avi")
