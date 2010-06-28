#Author Parham Aram
#Date 28-06-2010
"""
This module provides the full neural field class, which describes a non-linear
integro-difference equation model and methods to simulate
the model
"""

#Standard library imports
from __future__ import division
import pylab as pb
import numpy as np
import time
import os
from scipy import signal
import scipy as sp
from bases import Basis


class NF():

	"""class defining full neural model.

	Arguments
	----------
	kernel: 
			Connectivity kernel.

	sensor_kernel:

			output kernel, governs the sensor pick-up geometry
		
	obs_locns: List of matrix
				Sensors locations

	observation_locs_mm: ndarray
				center of sensors along x and y


	gamma : Gaussian basis function
				covariance function of the field disturbance
	gamma_weight: float
				amplitude of the field disturbance covariance function

	Sigma_varepsilon: matrix
				Observation noise covariance
	act_fun : 
				Sigmoidal activation function

	zeta: float
				Inverse synaptic time constant
	Ts: float
				Sampling time

	field_space: ndarray
				Spatiel field

	spacestep: float
			Spatial step size for descritization 	
	Attributes
	----------
	ny: int
		Number of sensors

	xi: float
		Time constant parameter

	gen_ssmodel:
		Generate the full neural field model

	simulate:
		Simulate the full neural field model
	"""

	def __init__(self,kernel,sensor_kernel,obs_locns,observation_locs_mm,gamma,gamma_weight,Sigma_varepsilon,act_fun,zeta,Ts,field_space,spacestep):

		self.kernel = kernel
		self.sensor_kernel=sensor_kernel
		self.obs_locns=obs_locns
		self.observation_locs_mm=observation_locs_mm
		self.gamma=gamma
		self.gamma_weight=gamma_weight
		self.Sigma_varepsilon=Sigma_varepsilon
		self.act_fun = act_fun
		self.zeta=zeta
		self.Ts=Ts
		self.spacestep=spacestep
		self.field_space=field_space
		self.ny=len(self.obs_locns)
		self.xi=1-(self.Ts*self.zeta)

	def gen_ssmodel(self):

		"""
		generates full neural model

		Attributes:
		----------
		K: matrix
			matrix of connectivity kernel evaluated over the spatial domain of the kernel

		Sigma_e: matrix
			field disturbance covariance matrix
		Sigma_e_c: matrix
			cholesky decomposiotion of field disturbance covariance matrix
		Sigma_varepsilon_c: matrix
			cholesky decomposiotion of observation noise covariance matrix
		C: matrix
			matrix of sensors evaluated at each spatial location, it's not the same as C in the IDE model	

		"""
		print "generating full neural model"


		K = pb.empty((len(self.field_space),len(self.field_space)))
		for i in range(len(self.field_space)):
			for j in range(len(self.field_space)):
				K[i,j]=self.kernel(pb.matrix([[self.field_space[i]],[self.field_space[j]]]))

		self.K=pb.matrix(K)


		#calculate field disturbance covariance matrix and its Cholesky decomposition
		gamma_space=pb.empty((self.field_space.size**2,2),dtype=float)
		l=0
		for i in self.field_space:
			for j in self.field_space:
				gamma_space[l]=[i,j]
				l+=1
		N1,D1 = gamma_space.shape
		diff = gamma_space.reshape(N1,1,D1) - gamma_space.reshape(1,N1,D1)
		Sigma_e_temp=self.gamma_weight*np.exp(-np.sum(np.square(diff),-1)*(1./self.gamma.width))
		self.Sigma_e=pb.matrix(Sigma_e_temp)

		if hasattr(self,'Sigma_e_c'):
			pass
		else:
			self.Sigma_e_c=pb.matrix(sp.linalg.cholesky(self.Sigma_e)).T

		#calculate Cholesky decomposition of observation noise covariance matrix
		Sigma_varepsilon_c=pb.matrix(sp.linalg.cholesky(self.Sigma_varepsilon)).T
		self.Sigma_varepsilon_c=Sigma_varepsilon_c

		#Calculate sensors at each spatial locations, it's not the same as C in the IDE model	
		t0=time.time()
		sensor_space=pb.empty((self.observation_locs_mm.size**2,2),dtype=float)
		l=0
		for i in self.observation_locs_mm:
			for j in self.observation_locs_mm:
				sensor_space[l]=[i,j]
				l+=1
		N2,D2 = sensor_space.shape
		diff = sensor_space.reshape(N2,1,D2) - gamma_space.reshape(1,N1,D1)
		C=np.exp(-np.sum(np.square(diff),-1)*(1./self.sensor_kernel.width))
		self.C=pb.matrix(C)





	def simulate(self,T):

		"""Simulates the full neural field model

		Arguments
		----------

		T: ndarray
				simulation time instants
		Returns
		----------
		V: list of matrix
			each matrix is the neural field at a time instant

		Y: list of matrix
			each matrix is the observation vector corrupted with noise at a time instant
		"""

		Y=[]
		V=[]  

		spatial_location_num=(len(self.field_space))**2
		sim_field_space_len=len(self.field_space) 

		#initial field
		v0=self.Sigma_e_c*pb.matrix(np.random.randn(spatial_location_num,1)) 
		v_membrane=pb.reshape(v0,(sim_field_space_len,sim_field_space_len))
		firing_rate=self.act_fun.fmax/(1.+pb.exp(self.act_fun.varsigma*(self.act_fun.v0-v_membrane)))


		for t in T[1:]:

			v = self.Sigma_varepsilon_c*pb.matrix(np.random.randn(len(self.obs_locns),1)) 
			w = pb.reshape(self.Sigma_e_c*pb.matrix(np.random.randn(spatial_location_num,1)),(sim_field_space_len,sim_field_space_len))

			print "simulation at time",t
			g=signal.fftconvolve(self.K,firing_rate,mode='same') 
			g*=(self.spacestep**2)
			v_membrane=self.Ts*pb.matrix(g) +self.xi*v_membrane+w
			firing_rate=self.act_fun.fmax/(1.+pb.exp(self.act_fun.varsigma*(self.act_fun.v0-v_membrane)))
			#Observation
			Y.append((self.spacestep**2)*(self.C*pb.reshape(v_membrane,(sim_field_space_len**2,1)))+v)
			V.append(v_membrane)

		return V,Y


class NF_Kernel():

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

		evaluates the kernel at a given spatial location
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




class ActivationFunction():

	"""class defining the sigmoidal activation function .

	Arguments
	----------
	v0: float
		firing threshold
	fmax: float
		maximum firing rate
	varsigma: float
		slope of sigmoid

	v: float
		Presynaptic potential

	Attributes:
	-----------
		plot:
			plots the sigmoidal activation function

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



def gen_spatial_lattice(S):

		"""generates a list of vectors, where each vector is a co-ordinate in a lattice

		Arguments:
		----------
		S: ndarray

		Returns:
		--------
		spatial_lattice: list of vectors, where each vector is a co-ordinate in a lattice
		"""


		number_of_points = len(S)**2

		spatial_lattice = [0 for i in range(number_of_points)]
		count = 0
		for i in S:
			for j in S:
				spatial_lattice[count] = pb.matrix([[i],[j]])
				count += 1
		return spatial_lattice



