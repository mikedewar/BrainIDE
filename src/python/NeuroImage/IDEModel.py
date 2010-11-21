#Author Parham Aram
#Date 21-11-2010

#Built-in modules
from __future__ import division
import pylab as pb
import numpy as np
import time
from scipy import signal
import scipy as sp
#My modules
import UKF
import LS
class IDE():

	"""class defining a non-linear, Integro-Difference, discrete-time state space model.

	Arguments
	----------
	kernel: IDEComponents.Kernel instance
			IDE kernel.
	field : IDEComponents.Field instance
			spatial field.
	sensor_kernel: Bases2D.basis instance
			output kernel, governs the sensor pick-up geometry.	
	observation_locs_mm:ndarray
				Sensors locations along x or y axes

	gamma : Bases2D.basis instance
				covariance function of the field disturbance
	gamma_weight: float
				amplitude of the field disturbance covariance function

	Sigma_varepsilon: ndarry
				Observation noise covariance
	act_fun :ActivationFunction.ActivationFunction instance
				Sigmoidal activation function
	x0: ndarray
				Initial state
	P0: ndarray
				Initial state covariance matrix
	zeta: float
				Inverse synaptic time constant
	Ts: float
				Sampling time

	estimation_space_x_y: ndarray
				Spatiel field

	spacestep: float
			Spatial step size for descritization 
	
	Attributes
	----------
	nx: int
		Number of basis functions
	ny: int
		Number of sensors
	n_theta:int
		Number of connectivity kernel basis functions

	xi: float
		Time constant parameter

	gen_ssmodel:
		Generate the state space model

	simulate:
		Simulate the state space model

	state_equation:
		Evaluate state transition function at a given state

	"""


	def __init__(self,kernel,field,sensor_kernel,obs_locns,gamma,gamma_weight,Sigma_varepsilon,act_fun,x0,P0,zeta,Ts,estimation_space_x_y,spacestep):

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
		self.estimation_space_x_y=estimation_space_x_y
		self.spacestep=spacestep
		self.nx=len(self.field.Phi)
		self.ny=len(obs_locns)
		self.n_theta=len(self.kernel.Psi)
		self.xi=1-(self.Ts*self.zeta)

	def gen_ssmodel(self):

		'''Generates non-linear, Integro-Difference, discrete-time state space model.

		Atributes:
		----------
		Gamma: ndarray
			Inner product of field basis functions

		Gamma_inv: ndarray
			Inverse of Gamma

		Sigma_e: ndarray
			Covariance matrix of the field disturbance

		Sigma_e_inv: ndarray
			Inverse of Sigma_e

		Sigma_e_c: ndarray
			cholesky decomposiotion of field disturbance covariance matrix
		
		Sigma_varepsilon_c: ndarray
			cholesky decomposiotion of observation noise covariance matrix

		Phi_values: ndarray  nx by number of spatial locations
			field basis functions values over the spatial field

		psi_conv_Phi_values: ndarray ; each ndarray is nx by number of spatial locations
			convolution of the  connectivity kernel basis functions with the field basis functions
			evaluate over space


		Gamma_inv_psi_conv_Phi:ndarray; each ndarray is nx by number of spatial locations
			the product of inverse gamme withe psi0_conv_Phi



		C: ndarray
			Observation matrix

		Gamma_inv_Psi_conv_Phi: ndarray nx by number of spatial locations
			the convolution of the kernel  with field basis functions at discritized spatial points

		'''

		#Generate Gamma
		if hasattr(self,'Gamma'):
			pass
		else:
			t_total=time.time()
			#calculate Gamma=PhixPhi.T; inner product of the field basis functions
			Gamma=pb.dot(self.field.Phi,self.field.Phi.T)
			Gamma_inv=pb.inv(Gamma)
			self.Gamma=Gamma.astype('float')
			self.Gamma_inv=Gamma_inv


		#Generate field covariance matrix
		if hasattr(self,'Sigma_e'):
			pass
		else:

			gamma_convolution_vecrorized=pb.vectorize(self.gamma.conv)
			gamma_conv_Phi=gamma_convolution_vecrorized(self.field.Phi).T 
			#[gamma*phi1 gamma*phi2 ... gamma*phin] 1 by nx
			Pi=pb.dot(self.field.Phi,gamma_conv_Phi) #nx by nx ndarray
			Pi=Pi.astype('float')
			Sigma_e=pb.dot(pb.dot(self.gamma_weight*Gamma_inv,Pi),Gamma_inv.T)
			Sigma_e_c=sp.linalg.cholesky(Sigma_e,lower=1)
			self.Sigma_e=Sigma_e
			self.Sigma_e_inv=pb.inv(Sigma_e)
			self.Sigma_e_c=Sigma_e_c
			Sigma_varepsilon_c=sp.linalg.cholesky(self.Sigma_varepsilon,lower=1)
			self.Sigma_varepsilon_c=Sigma_varepsilon_c



		if hasattr(self,'Phi_values'):
			pass
		else:

		
			#Generate field meshgrid
			estimation_field_space_x,estimation_field_space_y=pb.meshgrid(self.estimation_space_x_y,self.estimation_space_x_y)
			estimation_field_space_x=estimation_field_space_x.ravel()
			estimation_field_space_y=estimation_field_space_y.ravel()

			#calculate Phi_values
			#Phi_values is like [[phi1(r1) phi1(r2)...phi1(rn_r)],[phi2(r1) phi2(r2)...phi2(rn_r)],..[phiL(r1) phiL(r2)...phiL(rn_r)]]
			Phi_values=[self.field.Phi[i,0](estimation_field_space_x,estimation_field_space_y) for i in range(self.nx)]
			self.Phi_values=pb.squeeze(Phi_values) #it's nx by number of apatial points

			#vectorizing kernel convolution method
			psi_convolution_vectorized=pb.empty((self.n_theta,1),dtype=object)
			for i in range(self.n_theta):
				psi_convolution_vectorized[i,0]=pb.vectorize(self.kernel.Psi[i].conv)

			#find convolution between kernel and field basis functions analytically
			psi_conv_Phi=pb.empty((self.nx,self.n_theta),dtype=object)#nx by n_theta
			for i in range(self.n_theta):
				psi_conv_Phi[:,i]=psi_convolution_vectorized[i,0](self.field.Phi).ravel()


			#ecaluate convolution between kernel and field basis functions at spatial locations
			psi_conv_Phi_values=pb.empty((self.n_theta,self.nx,len(self.estimation_space_x_y)**2),dtype=float)
			for i in range(self.n_theta):
				for j in range(self.nx):
					psi_conv_Phi_values[i,j,:]=psi_conv_Phi[j,i](estimation_field_space_x,estimation_field_space_y)
			self.psi_conv_Phi_values=psi_conv_Phi_values



			self.Gamma_inv_psi_conv_Phi=pb.dot(self.Gamma_inv,self.psi_conv_Phi_values)
				

		


		#Calculate observation matrix
			
		if hasattr(self,'C'):
			pass
		else:
			#Generate Observation locations grid
			obs_locns_x=self.obs_locns[:,0]
			obs_locns_y=self.obs_locns[:,1]
			sensor_kernel_convolution_vecrorized=pb.vectorize(self.sensor_kernel.conv)
			sensor_kernel_conv_Phi=sensor_kernel_convolution_vecrorized(self.field.Phi).T #first row 
			#[m*phi_1 m*phi2 ... m*phin]
			C=pb.empty(([self.ny,self.nx]),dtype=float)
			for i in range(self.nx):
				C[:,i]=sensor_kernel_conv_Phi[0,i](obs_locns_x,obs_locns_y)

			self.C=C
			print 'Elapsed time in seconds to generate the stat-space model',time.time()-t_total


		#We need to calculate this bit at each iteration	
		#Finding the convolution of the kernel  with field basis functions at discritized spatial points
		self.Gamma_inv_Psi_conv_Phi=pb.sum(self.kernel.weights[pb.newaxis,:,pb.newaxis]*self.Gamma_inv_psi_conv_Phi,axis=1)



	def simulate(self,T):

		"""
		generates nonlinear IDE


		Arguments
		----------
		T: ndarray
				simulation time instants
		Returns
		----------
		X: list of ndarray
			each ndarray is the state vector at a time instant

		Y: list of ndarray
			each ndarray is the observation vector corrupted with noise at a time instant
		"""


		Y = []		
		X = []		
		x=self.x0
		firing_rate_temp=pb.dot(x.T,self.Phi_values)
 		firing_rate=self.act_fun.fmax/(1.+pb.exp(self.act_fun.varsigma*(self.act_fun.v0-firing_rate_temp))).T		
		print "iterating"
		for t in T[1:]:
			w = pb.dot(self.Sigma_e_c,np.random.randn(self.nx,1))
			v = pb.dot(self.Sigma_varepsilon_c,np.random.randn(len(self.obs_locns),1))

			g=pb.dot(self.Gamma_inv_Psi_conv_Phi,firing_rate)
			g *=(self.spacestep**2)
			x=self.Ts*g+self.xi*x+w

			X.append(x)
			Y.append(pb.dot(self.C,x)+v)
			firing_rate_temp=pb.dot(x.T,self.Phi_values)
			firing_rate=self.act_fun.fmax/(1.+pb.exp(self.act_fun.varsigma*(self.act_fun.v0-firing_rate_temp))).T		

		return X,Y


	def  state_equation(self,x):

		'''state equation for sigma points propogation '''
		firing_rate_temp=pb.dot(x.T,self.Phi_values)
 		firing_rate=self.act_fun.fmax/(1.+pb.exp(self.act_fun.varsigma*(self.act_fun.v0-firing_rate_temp))).T	
		g=pb.dot(self.Gamma_inv_Psi_conv_Phi,firing_rate)
		g *=(self.spacestep**2)
		x=self.Ts*g+self.xi*x
		return x



class para_state_estimation():

	def __init__(self,model):

		'''this is to estimate state and connectivity kernel parameters

		Arguments:
		----------
			model: IDE instance
		'''

		self.model=model





	def itrerative_state_parameter_estimation(self,Y,max_it):

		"""Two part iterative algorithm, consisting of a state estimation step followed by a
		parameter estimation step
		
		Arguments:
		---------
		Y: list of ndarray
			Observation vectors
		max_it: int
			maximum number of iterations """


		xi_est=[]
		kernel_weights_est=[]
		# generate a random state sequence
		Xb= [np.random.rand(self.model.nx,1) for t in Y]
		# iterate
		keep_going = 1
		it_count = 0
		print " Estimatiing IDE's kernel and the time constant parameter"
		t0=time.time()
		while keep_going:
			Ls_instance=LS. para_state_estimation(self.model)
			temp=Ls_instance.estimate_kernel(Xb)
			xi_est.append(float(temp[-1]))
			kernel_weights_est.append(temp[0:-1])
			self.model.kernel.weights,self.model.xi=pb.array(temp[0:-1]),float(temp[-1])
			self.model.gen_ssmodel()
			ukf_instance=UKF.ukf(self.model.nx,self.model.x0,self.model.P0,self.model.C,self.model.Sigma_e,self.model.Sigma_varepsilon,self.model.state_equation,kappa=0.0,alpha_sigma_points=1e-3,beta_sigma_points=2)
			Xb,Pb=ukf_instance.rtssmooth(Y)
			self.model.Xb=Xb
			self.model.Pb=Pb
			self.model.xi_est=xi_est
			self.model.kernel_weights_est=kernel_weights_est
			self.model.x0=Xb[0]
			self.model.P0=Pb[0]
			print it_count, " Kernel current estimate: ", self.model.kernel.weights, "xi", self.model.xi
			if it_count == max_it:
				keep_going = 0
			it_count += 1
		print "Elapsed time in seconds is", time.time()-t0

def dots(*args):
	lastItem = 1.
	for arg in args:
		lastItem = pb.dot(lastItem, arg)
	return lastItem
