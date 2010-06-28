#Author Parham Aram
#Date 28-06-2010

"""
This module provides the IDE class, which describes a non-linear
integro-difference equation model and methods to generate,
interrogate and estimate the model from data.
"""

#Standard library imports
from __future__ import division
import pylab as pb
import numpy as np
import time
import os
from scipy import signal
import scipy as sp


class IDE():

	"""class defining a non-linear, Integro-Difference, discrete-time state space model.

	Arguments
	----------
	kernel: 
			IDE kernel.
	field : 
			spatial field.
	sensor_kernel:

			output kernel, governs the sensor pick-up geometry
	
	obs_locns: List of matrix
				Sensors locations

	gamma : Gaussian basis function
				covariance function of the field disturbance
	gamma_weight: float
				amplitude of the field disturbance covariance function

	Sigma_varepsilon: matrix
				Observation noise covariance
	act_fun : 
				Sigmoidal activation function
	x0: matrix
				Initial state
	P0: matrix
				Initial state covariance matrix
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
	nx: int
		Number of basis functions
	ny: int
		Number of sensors

	xi: float
		Time constant parameter

	gen_ssmodel:
		Generate the state space model

	simulate:
		Simulate the state space model

	state_equation:
		Evaluate state transition function at a given state

	"""

	def __init__(self,kernel,field,sensor_kernel,obs_locns,gamma,gamma_weight,Sigma_varepsilon,act_fun,x0,P0,zeta,Ts,field_space,spacestep):

		self.kernel = kernel
		self.field = field
		self.sensor_kernel=sensor_kernel
		self.obs_locns=obs_locns
		self.gamma=gamma
		self.gamma_weight=gamma_weight
		self.Sigma_varepsilon=Sigma_varepsilon
		self.act_fun = act_fun
		self.x0=x0
		self.P0=P0
		self.zeta=zeta
		self.Ts=Ts
		self.spacestep=spacestep
		self.field_space=field_space
		self.nx=len(self.field.Phi)
		self.ny=len(self.obs_locns)
		self.xi=1-(self.Ts*self.zeta)

	def gen_ssmodel(self):

		'''Generates non-linear, Integro-Difference, discrete-time state space model.

		Atributes:
		----------
		Gamma: matrix
			Inner product of field basis functions
		Sigma_e: matrix
			Covariance matrix of the field disturbance
		Sigma_e_c: matrix
			cholesky decomposiotion of field disturbance covariance matrix
		
		Sigma_varepsilon_c: matrix
			cholesky decomposiotion of observation noise covariance matrix

		Phi_values: matrix
			field basis functions values over the spatial field

		psi0_conv_Phi_values: list
			convolution of the central excitatory connectivity kernel basis function with the field basis functions
			evaluate over space

		psi1_conv_Phi_values: list
			convolution of the surround inhibition connectivity kernel basis function with the field basis functions
			evaluate over space

		psi2_conv_Phi_values: list
			convolution of the longer range excitatory connectivity kernel basis function with the field basis functions
			evaluate over space

		C: matrix
			Observation matrix
		'''


		print 'Generating state space model'
		# unpack the kernel
 		psi0,psi1,psi2=self.kernel.Psi[0],self.kernel.Psi[1],self.kernel.Psi[2]
		psi0_weight,psi1_weight,psi2_weight=self.kernel.weights[0],self.kernel.weights[1],self.kernel.weights[2]
		#centers
		psi0_center=psi0.center
		psi1_center=psi1.center
		psi2_center=psi2.center
		#widths
		psi0_width=psi0.width
		psi1_width=psi1.width
		psi2_width=psi2.width


		#Generate Gamma
		if hasattr(self,'Gamma'):
			pass
		else:

			#calculate Gamma=PhixPhi.T; inner product of the field basis functions
			Gamma=pb.matrix(self.field.Phi*self.field.Phi.T,dtype=float)
			Gamma_inv=Gamma.I
			self.Gamma=Gamma
			self.Gamma_inv=Gamma_inv

		#Generate field covariance matrix and its Cholesky decomposition
		if hasattr(self,'Sigma_e'):
			pass
		else:

			gamma_convolution_vecrorized=pb.vectorize(self.gamma.conv)
			gamma_conv_Phi=gamma_convolution_vecrorized(self.field.Phi).T 
			Pi=pb.matrix(self.field.Phi*gamma_conv_Phi,dtype=float) #nx by nx matrix
			Sigma_e=self.gamma_weight*Gamma_inv*Pi*Gamma_inv.T
			Sigma_e_c=pb.matrix(sp.linalg.cholesky(Sigma_e)).T
			self.Sigma_e=Sigma_e
			self.Sigma_e_c=Sigma_e_c
			#calculate Cholesky decomposition of observation noise covariance matrix
			Sigma_varepsilon_c=pb.matrix(sp.linalg.cholesky(self.Sigma_varepsilon)).T
			self.Sigma_varepsilon_c=Sigma_varepsilon_c


		if hasattr(self,'Phi_values'):
			pass
		else:

		

			#vectorizing convolution function of the kernel basis functions
			psi0_convolution_vectorized=pb.vectorize(psi0.conv) 	
			psi1_convolution_vectorized=pb.vectorize(psi1.conv) 	
			psi2_convolution_vectorized=pb.vectorize(psi2.conv) 	
			#convolving each kernel basis function with the field basis functions
			psi0_conv_Phi=psi0_convolution_vectorized(self.field.Phi) 
			psi1_conv_Phi=psi1_convolution_vectorized(self.field.Phi) 
			psi2_conv_Phi=psi2_convolution_vectorized(self.field.Phi)

			Psi_conv_Phi=pb.hstack((psi0_conv_Phi,psi1_conv_Phi,psi2_conv_Phi))
	
			#Finding the convolution of the kernel basis functions with field basis functions at discritized spatial points
			psi0_conv_Phi_values=[]
			psi1_conv_Phi_values=[]
			psi2_conv_Phi_values=[]
			Phi_values=[]

			psi0_conv_Phi_values_temp=[]
			psi1_conv_Phi_values_temp=[]
			psi2_conv_Phi_values_temp=[]
			Phi_values_temp=[]

			t_Gamma_inv_Psi_sep=time.time()
			for m in range(self.nx):
				for i in self.field_space:
					for j in self.field_space:
						psi0_conv_Phi_values_temp.append(psi0_conv_Phi[m,0](pb.matrix([[i],[j]])))
						psi1_conv_Phi_values_temp.append(psi1_conv_Phi[m,0](pb.matrix([[i],[j]])))
						psi2_conv_Phi_values_temp.append(psi2_conv_Phi[m,0](pb.matrix([[i],[j]])))


						Phi_values_temp.append(self.field.Phi[m,0](pb.matrix([[i],[j]])))


				psi0_conv_Phi_values.append(psi0_conv_Phi_values_temp)
				psi1_conv_Phi_values.append(psi1_conv_Phi_values_temp)
				psi2_conv_Phi_values.append(psi2_conv_Phi_values_temp)
				Phi_values.append(Phi_values_temp)

				psi0_conv_Phi_values_temp=[]
				psi1_conv_Phi_values_temp=[]
				psi2_conv_Phi_values_temp=[]
				Phi_values_temp=[]


			Phi_values=pb.matrix(Phi_values)

			self.psi0_conv_Phi_values=psi0_conv_Phi_values
			self.psi1_conv_Phi_values=psi1_conv_Phi_values
			self.psi2_conv_Phi_values=psi2_conv_Phi_values		


			self.Phi_values=Phi_values 
			Gamma_inv_psi0_conv_Phi=self.Gamma_inv*psi0_conv_Phi_values 
			Gamma_inv_psi1_conv_Phi=self.Gamma_inv*psi1_conv_Phi_values 
			Gamma_inv_psi2_conv_Phi=self.Gamma_inv*psi2_conv_Phi_values  			

			self.Gamma_inv_psi0_conv_Phi=Gamma_inv_psi0_conv_Phi
			self.Gamma_inv_psi1_conv_Phi=Gamma_inv_psi1_conv_Phi
			self.Gamma_inv_psi2_conv_Phi=Gamma_inv_psi2_conv_Phi



		#Calculate observation matrix
			
		if hasattr(self,'C'):
			pass
		else:
			sensor_kernel_convolution_vecrorized=pb.vectorize(self.sensor_kernel.conv)
			sensor_kernel_conv_Phi=sensor_kernel_convolution_vecrorized(self.field.Phi).T 
			sensor_kernel_conv_Phi_values_temp=[]
			sensor_kernel_conv_Phi_values=[]
			for m in range(self.nx):
				for n in self.obs_locns:
					sensor_kernel_conv_Phi_values_temp.append(sensor_kernel_conv_Phi[0,m](n))
				sensor_kernel_conv_Phi_values.append(sensor_kernel_conv_Phi_values_temp)
				sensor_kernel_conv_Phi_values_temp=[]
			C=pb.matrix(pb.squeeze(sensor_kernel_conv_Phi_values).T)
			self.C=C


	
		#Finding the convolution of the kernel  with field basis functions at discritized spatial points
		self.Gamma_inv_Psi_conv_Phi=psi0_weight*self.Gamma_inv_psi0_conv_Phi+ \
		psi1_weight*self.Gamma_inv_psi1_conv_Phi+psi2_weight*self.Gamma_inv_psi2_conv_Phi



	def simulate(self,T):

		"""Simulates the nonlinear state space IDE

		Arguments
		----------

		T: ndarray
				simulation time instants
		Returns
		----------
		X: list of matrix
			each matrix is the state vector at a time instant

		Y: list of matrix
			each matrix is the observation vector corrupted with noise at a time instant
		"""
		Y = []		
		X = []		
		x=self.x0
		firing_rate_temp=x.T*self.Phi_values
 		firing_rate=pb.array(self.act_fun.fmax/(1.+pb.exp(self.act_fun.varsigma*(self.act_fun.v0-firing_rate_temp)))).T		

		print "iterating"
		for t in T[1:]:
			w = self.Sigma_e_c*pb.matrix(np.random.randn(self.nx,1))
			v = self.Sigma_varepsilon_c*pb.matrix(np.random.randn(len(self.obs_locns),1))
			print "simulation at time",t
			Gamma_inv_Psi_conv_Phi=pb.array(self.Gamma_inv_Psi_conv_Phi)
			g=pb.matrix(pb.dot(Gamma_inv_Psi_conv_Phi,firing_rate))
			g *=(self.spacestep**2)
			x=self.Ts*g+self.xi*x+w

			X.append(x)
			Y.append(self.C*x+v)
			firing_rate_temp=x.T*self.Phi_values
			firing_rate=pb.array(self.act_fun.fmax/(1.+pb.exp(self.act_fun.varsigma*(self.act_fun.v0-firing_rate_temp)))).T		

		return X,Y



	def  state_equation(self,x):

		'''state transition function 
		Arguments:
		----------
		x: matrix
			state vector
		Returns:
		--------
		Q: matrix
			Result of the state transition function at x
			'''
		firing_rate_temp=x.T*self.Phi_values
 		firing_rate=pb.array(self.act_fun.fmax/(1.+pb.exp(self.act_fun.varsigma*(self.act_fun.v0-firing_rate_temp)))).T		
		Gamma_inv_Psi_conv_Phi=pb.array(self.Gamma_inv_Psi_conv_Phi)
		g=pb.matrix(pb.dot(Gamma_inv_Psi_conv_Phi,firing_rate))
		g *=(self.spacestep**2)
		Q=self.Ts*g+self.xi*x
		return Q

			


class IDE_Kernel():

	"""class defining the connectivity kernel of the brain.

	Arguments
	----------

	Psi:list
		list of connectivity kernel basis functions; must be of class Bases
	weights: list
		list of connectivity kernel weights	
	"""
	
	def __init__(self,Psi,weights):
		
		self.Psi=Psi
		self.weights = weights



class IDE_Field():	

	"""class defining the spatial field."""

	def __init__(self, Phi):

		self.Phi=Phi


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

class ukf():

	"""class defining the Unscented Kalman filter.

	Arguments
	----------
	model: IDE instance
		Nonlinear IDE state space model
	kappa: float
		secondary scaling parameter.
	alpha_sigma_points: 
		Determines the spread of sigma points around x.
	beta_sigma_points:
		Incorporates prior knowledge of the distribution of x
		beta=2 is optimal for Gaussian distribution
	

	Attributes
	----------
	L: int
		state vector dimension
	lamda: float
		Scaling parameter
	gamma_sigma_points: float
		Scaling parameter
	rtssmooth:
		implements the unscented Rauch-Tung-Striebel smoother
	"""

	def __init__(self,model,kappa=0.0,alpha_sigma_points=1e-3,beta_sigma_points=2):

		self.model=model
		self.alpha_sigma_points=alpha_sigma_points
		self.beta_sigma_points=beta_sigma_points
		self.L=self.model.nx
		self.kappa=3-self.L
		self.lamda=self.alpha_sigma_points**2*(self.L+self.kappa)-self.L    
		self.gamma_sigma_points=pb.sqrt(self.L+self.lamda) 	

	def sigma_vectors(self,x,P):

		"""
		generates sigma vectors

		Arguments
		----------
		x : matrix
			state at time instant t
		P:  matrix
			state covariance matrix at time instant t

		Returns
		----------
		Chi : matrix
			matrix of sigma points
		"""
		State_covariance_cholesky=sp.linalg.cholesky(P).T
		State_covariance_cholesky_product=self.gamma_sigma_points*State_covariance_cholesky		
		chi_plus=[]
		chi_minus=[]
		for i in range(self.L):
			chi_plus.append(x+State_covariance_cholesky_product[:,i].reshape(self.L,1)) 
			chi_minus.append(x-State_covariance_cholesky_product[:,i].reshape(self.L,1)) 

		Chi=pb.hstack((x,pb.hstack((pb.hstack(chi_plus),pb.hstack(chi_minus))))) 
		return pb.matrix(Chi)



	def sigma_vectors_weights(self):

		"""
		generates  sigma vector weights

		Returns
		----------
		Wm_i : ndarray
			array of sigma points' weights 
		Wc_i : ndarray
			array of sigma points' weights 
		"""
		Wm0=[self.lamda/(self.lamda+self.L)]
		Wc0=[(self.lamda/(self.lamda+self.L))+1-self.alpha_sigma_points**2+self.beta_sigma_points]
		Wmc=[1./(2*(self.L+self.lamda))]
		Wm_i=pb.concatenate((Wm0,2*self.L*Wmc)) 
		Wc_i=pb.concatenate((Wc0,2*self.L*Wmc)) 
		return Wm_i,Wc_i


	def rtssmooth(self,Y):

		''' RTS smoother

		Arguments:
		----------
		Y: list of matrices
			observation vectors
		Returns:
		--------
		xb:list of matrices
			Backward posterior state estimates
		Pb:list of matrices
			Backward posterior covariabce matrices
		xhat:list of matrices
			Forward posterior state estimates
		Phat:list of matrices
			Forward posterior covariabce matrices

		'''

		# initialise
		P=self.model.P0 
		xf=self.model.x0
		# filter quantities
		xfStore =[]
		PfStore=[]

		#calculate the sigma vector weights
		Wm_i,Wc_i=self.sigma_vectors_weights()



		for y in Y:
			#calculate the sigma points matrix
			Chi=self.sigma_vectors(xf,P)
			# update sigma vectors
			Chi_update=pb.matrix(pb.empty_like(Chi))
			for i in range(Chi.shape[1]):
				Chi_update[:,i]=self.model.state_equation(Chi[:,i])	
			#calculate forward prior state estimate
			xf_=pb.sum(pb.multiply(Wm_i,Chi_update),1)
			#perturbation
			Chi_perturbation=Chi_update-xf_
			#weighting
			weighted_Chi_perturbation=pb.multiply(Wc_i,Chi_perturbation)
			#calculate forward prior covariance estimate
			Pf_=Chi_perturbation*weighted_Chi_perturbation.T+self.model.Sigma_e
			#measurement update equation
			Pyy=self.model.C*Pf_*self.model.C.T+self.model.Sigma_varepsilon 
			Pxy=Pf_*self.model.C.T
			K=Pxy*(Pyy.I)
			yhat_=self.model.C*xf_
			#calculate forward posterior state and covariance estimates
			xf=xf_+K*(y-yhat_)
			Pf=(pb.eye(self.model.nx)-K*self.model.C)*Pf_
			#store
			xfStore.append(xf)
			PfStore.append(Pf)

		# initialise the smoother
		T=len(Y)
		xb = [None]*T
		Pb = [None]*T

		xb[-1], Pb[-1] = xfStore[-1], PfStore[-1]

		## smoother
		for t in range(T-2,-1,-1):
			#calculate the sigma points matrix from filterd states
			Chi_smooth=self.sigma_vectors(xfStore[t],PfStore[t]) 
			Chi_smooth_update=pb.matrix(pb.empty_like(Chi))
			for i in range(Chi_smooth.shape[1]):
				Chi_smooth_update[:,i]=self.model.state_equation(Chi_smooth[:,i])
			
			#calculate backward prior state estimate
			xb_=pb.sum(pb.multiply(Wm_i,Chi_smooth_update),1) 
			#perturbation
			Chi_smooth_perturbation=Chi_smooth-xfStore[t] 
			Chi_smooth_update_perturbation=Chi_smooth_update-xb_ 
			#weighting
			weighted_Chi_smooth_perturbation=pb.multiply(Wc_i,Chi_smooth_perturbation) 
			weighted_Chi_smooth_update_perturbation=pb.multiply(Wc_i,Chi_smooth_update_perturbation)
			#calculate backward prior covariance
			Pb_=Chi_smooth_update_perturbation*weighted_Chi_smooth_update_perturbation.T+self.model.Sigma_e
			#calculate cross-covariance matrix
			M=weighted_Chi_smooth_perturbation*Chi_smooth_update_perturbation.T
			#calculate smoother gain
			S=M*Pb_.I
			#calculate backward posterior state and covariance estimates
			xb[t]=xfStore[t]+S*(xb[t+1]-xb_)
			Pb[t]=PfStore[t]+S*(Pb[t+1]-Pb_)*S.T

			
		return xb,Pb,xfStore,PfStore


class para_state_estimation():

	def __init__(self,model):

		self.model=model


	def q_calc(self,X):

		"""
			calculates q (L by n_theta) matrix at each time step
	
			Arguments
			----------
			X: list of matrix
				state vectors

			Returns
			---------
			q_list : list of matrix, each matrix is L by n_theta
		"""


		q_list=[]	
		T=len(X)
		for t in range(T):

			firing_rate_temp=X[t].T*self.model.Phi_values
			firing_rate=pb.array(self.model.act_fun.fmax/(1.+pb.exp(self.model.act_fun.varsigma*(self.model.act_fun.v0-firing_rate_temp)))).T		

			Gamma_inv_psi0_conv_Phi=pb.array(self.model.Gamma_inv_psi0_conv_Phi)
			Gamma_inv_psi1_conv_Phi=pb.array(self.model.Gamma_inv_psi1_conv_Phi)
			Gamma_inv_psi2_conv_Phi=pb.array(self.model.Gamma_inv_psi2_conv_Phi)

			sum0_s=pb.matrix(pb.dot(Gamma_inv_psi0_conv_Phi,firing_rate))
			sum1_s=pb.matrix(pb.dot(Gamma_inv_psi1_conv_Phi,firing_rate))
			sum2_s=pb.matrix(pb.dot(Gamma_inv_psi2_conv_Phi,firing_rate))

			temp=pb.hstack((sum0_s,sum1_s,sum2_s))

			temp *=(self.model.spacestep**2)	
			q=self.model.Ts*temp
			q_list.append(q)		
		return q_list

	def LS(self,X):

		"""
			estimate the connectivity kernel parameters and the time constant parameter using Least Square method
	
			Arguments
			----------
			X: list of matrix
				state vectors

			Returns
			---------
			Least Square estimation of the the connectivity kernel parameters and the time constant parameter 
		"""
		q=self.q_calc(X)
		Z=pb.vstack(X[1:])
		X_t_1=pb.vstack(X[:-1])
		q_t_1=pb.vstack(q[:-1])
		X_ls=pb.hstack((q_t_1,X_t_1))
		W=(X_ls.T*X_ls).I*X_ls.T*Z
		return [float( W[0]),float(W[1]),float(W[2]),float(W[3])]
 




	def itrerative_state_parameter_estimation(self,Y,max_it):

		"""Two part iterative algorithm, consisting of a state estimation step followed by a
		parameter estimation step
		
		Arguments:
		---------
		Y: list of matrices
			Observation vectors
		max_it: int
			maximum number of iterations """


		xi_est=[]
		kernel_weights_est=[]
		# generate a random state sequence
		Xb= [pb.matrix(np.random.rand(self.model.nx,1)) for t in Y]
		# iterate
		keep_going = 1
		it_count = 0
		print " Estimatiing IDE's kernel, the time constant parameter and field weights"
		t0=time.time()
		while keep_going:
			
			temp=self.LS(Xb)
			xi_est.append(float(temp[-1]))
			kernel_weights_est.append(temp[0:-1])
			self.model.kernel.weights,self.model.xi=temp[0:-1],float(temp[-1])
			self.model.gen_ssmodel()
			_filter=getattr(ukf(self.model),'rtssmooth')
			Xb,Pb,Xf,Pf=_filter(Y)
			self.model.Xb=Xb
			self.model.Pb=Pb
			self.model.Xf=Xf
			self.model.Pf=Pf
			self.model.xi_est=xi_est
			self.model.kernel_weights_est=kernel_weights_est
			self.model.x0=Xb[0]
			self.model.P0=Pb[0]
			print it_count, " Kernel current estimate: ", self.model.kernel.weights, "xi", self.model.xi
			if it_count == max_it:
				keep_going = 0
			it_count += 1
		print "Elapsed time in seconds is", time.time()-t0


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






