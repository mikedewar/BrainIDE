#Built-in modules
from __future__ import division
import pylab as pb
import numpy as np
import matplotlib.axes3d
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


	
		# unpack the kernel
 		psi1,psi2,psi3=self.kernel.Psi[0],self.kernel.Psi[1],self.kernel.Psi[2]
		psi1_weight,psi2_weight,psi3_weight=self.kernel.weights[0],self.kernel.weights[1],self.kernel.weights[2]
		#Kernel basis functions' centers
		psi1_center=psi1.center
		psi2_center=psi2.center
		psi3_center=psi3.center
		#Kernel basis functions' widths
		psi1_width=psi1.width
		psi2_width=psi2.width
		psi3_width=psi3.width


		#Generate Gamma
		if hasattr(self,'Gamma'):
			pass
		else:
			t_total=time.time()
			t_Gamma=time.time()
			#calculate Gamma=PhixPhi.T; inner product of the field basis functions
			Gamma=pb.matrix(self.field.Phi*self.field.Phi.T,dtype=float)
			Gamma_inv=Gamma.I
			self.Gamma=Gamma
			self.Gamma_inv=Gamma_inv
			print 'Elapsed time in seconds to calculate Gamma inverse is',time.time()-t_Gamma

		#Generate field covariance matrix
		if hasattr(self,'Sigma_e'):
			pass
		else:
			print "calculating field noise covariances"
			t_Sigma_e_c=time.time()
			gamma_convolution_vecrorized=pb.vectorize(self.gamma.conv)
			gamma_conv_Phi=gamma_convolution_vecrorized(self.field.Phi).T 
			#[gamma*phi1 gamma*phi2 ... gamma*phin] 1 by nx
			Pi=pb.matrix(self.field.Phi*gamma_conv_Phi,dtype=float) #nx by nx matrix
			Sigma_e=self.gamma_weight*Gamma_inv*Pi*Gamma_inv.T
			Sigma_e_c=pb.matrix(sp.linalg.cholesky(Sigma_e)).T
			self.Sigma_e=Sigma_e
			self.Sigma_e_c=Sigma_e_c
			Sigma_varepsilon_c=pb.matrix(sp.linalg.cholesky(self.Sigma_varepsilon)).T
			self.Sigma_varepsilon_c=Sigma_varepsilon_c

			print 'Elapsed time in seconds to calculate Cholesky decomposition of the noise covariance matrix',time.time()-t_Sigma_e_c


		if hasattr(self,'Phi_values'):
			pass
		else:

		

			#vectorizing convolution function of the kernel basis functions
			psi1_convolution_vectorized=pb.vectorize(psi1.conv) 	
			psi2_convolution_vectorized=pb.vectorize(psi2.conv) 	
			psi3_convolution_vectorized=pb.vectorize(psi3.conv) 	
			#convolving each kernel basis functions with the field basis functions
			psi1_conv_Phi=psi1_convolution_vectorized(self.field.Phi) #[psi1*phi1 psi1*phi2 ... psi1*phiL].T
			psi2_conv_Phi=psi2_convolution_vectorized(self.field.Phi) #[psi2*phi1 psi2*phi2 ... psi2*phiL].T
			psi3_conv_Phi=psi3_convolution_vectorized(self.field.Phi) #[psi3*phi1 psi3*phi2 ... psi3*phiL].T

			#place convolution of the kernel basis functions with field basis functions in a matrix of (n_x X n_theta) dimension
			#[psi1*phi1 psi2*phi1 psi3*phi1;psi1*phi2 psi2*phi2 psi3*phi2;... psi1*phiL psi2*phiL psi3*phiL]
			Psi_conv_Phi=pb.hstack((psi1_conv_Phi,psi2_conv_Phi,psi3_conv_Phi))
	
			##Finding the convolution of the kernel basis functions with field basis functions at discritized spatial points
			psi1_conv_Phi_values=[]
			psi2_conv_Phi_values=[]
			psi3_conv_Phi_values=[]
			#Psi_conv_Phi_values=[]
			Phi_values=[]

			psi1_conv_Phi_values_temp=[]
			psi2_conv_Phi_values_temp=[]
			psi3_conv_Phi_values_temp=[]
			#Psi_conv_Phi_values_temp=[]
			Phi_values_temp=[]

			t_Gamma_inv_Psi_sep=time.time()
			for m in range(self.nx):
				for i in self.field_space:
					for j in self.field_space:
						psi1_conv_Phi_values_temp.append(psi1_conv_Phi[m,0](pb.matrix([[i],[j]])))
						psi2_conv_Phi_values_temp.append(psi2_conv_Phi[m,0](pb.matrix([[i],[j]])))
						psi3_conv_Phi_values_temp.append(psi3_conv_Phi[m,0](pb.matrix([[i],[j]])))
						#Psi_conv_Phi_values_temp.append(psi1_weight*psi1_conv_Phi_values_temp[-1]+ \
						#psi2_weight*psi2_conv_Phi_values_temp[-1]+psi3_weight*psi3_conv_Phi_values_temp[-1])


						Phi_values_temp.append(self.field.Phi[m,0](pb.matrix([[i],[j]])))


				psi1_conv_Phi_values.append(psi1_conv_Phi_values_temp)
				psi2_conv_Phi_values.append(psi2_conv_Phi_values_temp)
				psi3_conv_Phi_values.append(psi3_conv_Phi_values_temp)
				#Psi_conv_Phi_values.append(Psi_conv_Phi_values_temp)
				#Phi_values is like [[phi1(s1) phi1(s2)...phi1(sn)],[phi2(s1) phi2(s2)...phi2(sn)],..[phiL(s1) phiL(s2)...phiL(sn)]]
				Phi_values.append(Phi_values_temp)

				psi1_conv_Phi_values_temp=[]
				psi2_conv_Phi_values_temp=[]
				psi3_conv_Phi_values_temp=[]
				#Psi_conv_Phi_values_temp=[]
				Phi_values_temp=[]


			Phi_values=pb.matrix(Phi_values)

			self.psi1_conv_Phi_values=psi1_conv_Phi_values
			self.psi2_conv_Phi_values=psi2_conv_Phi_values
			self.psi3_conv_Phi_values=psi3_conv_Phi_values		
			#self.Psi_conv_Phi_values=Psi_conv_Phi_values

			self.Phi_values=Phi_values #its a L by n (number of spatial points matrix):
			# [[phi1(s1) phi1(s2)...phi1(sn)],[phi2(s1) phi2(s2)...phi2(sn)],..[phiL(s1) phiL(s2)...phiL(sn)]]
			#Here psi1_conv_Phi_values is a list but behaves as a matrix of the form of [[psi1*phi1(s0) psi1*phi1(s1) ... psi1*phi1(sn)];[psi1*phi2(s0) psi1*phi2(s1) ... psi1*phi2(sn)]...;[psi1*phin(s0) psi1*phin(s1) ... psi1*phin(sn)]]
			Gamma_inv_psi1_conv_Phi=self.Gamma_inv*psi1_conv_Phi_values #matrix in a form of n_x X number of spatiol points
			Gamma_inv_psi2_conv_Phi=self.Gamma_inv*psi2_conv_Phi_values #matrix in a form of n_x X number of spatiol points
			Gamma_inv_psi3_conv_Phi=self.Gamma_inv*psi3_conv_Phi_values #matrix in a form of n_x X number of spatiol points
			#Gamma_inv_Psi_conv_Phi=self.Gamma_inv*Psi_conv_Phi_values   #matrix in a form of n_x X number of spatiol points
			self.Gamma_inv_psi1_conv_Phi=Gamma_inv_psi1_conv_Phi
			self.Gamma_inv_psi2_conv_Phi=Gamma_inv_psi2_conv_Phi
			self.Gamma_inv_psi3_conv_Phi=Gamma_inv_psi3_conv_Phi
			#self.Gamma_inv_Psi_conv_Phi=Gamma_inv_Psi_conv_Phi
			print 'Elapsed time in seconds to calculate the convolution and their products with Gamma for each kernel basis functions is', time.time()-t_Gamma_inv_Psi_sep



			#Calculate observation matrix
			
		if hasattr(self,'C'):
			pass
		else:
			sensor_kernel_convolution_vecrorized=pb.vectorize(self.sensor_kernel.conv)
			sensor_kernel_conv_Phi=sensor_kernel_convolution_vecrorized(self.field.Phi).T #first row
			#[m*phi_1 m*phi2 ... m*phin]
			t_observation_matrix=time.time()
			sensor_kernel_conv_Phi_values_temp=[]
			sensor_kernel_conv_Phi_values=[]
			for m in range(self.nx):
				for n in self.obs_locns:
					sensor_kernel_conv_Phi_values_temp.append(sensor_kernel_conv_Phi[0,m](n))
				sensor_kernel_conv_Phi_values.append(sensor_kernel_conv_Phi_values_temp)
				sensor_kernel_conv_Phi_values_temp=[] 
			#sensor_kernel_conv_Phi_values=[[m*phi_1(r1) m*phi_1(r2)...m*phi_1(rn)],[m*phi_2(r1) m*phi_2(r2)...m*phi_2(rn)]...
			#[m*phi_n(r1) m*phi_n(r2)...m*phi_n(rn)]] when we squeeze it we get an array in a form of
			#[m*phi_1(r1) m*phi_1(r2)...m*phi_1(rn);m*phi_2(r1) m*phi_2(r2)...m*phi_2(rn);...;m*phi_n(r1) m*phi_n(r2)...m*phi_n(rn)]
			#after doing transpose we get C
			#[m*phi_1(r1) m*phi_2(r1) ...m*phi_n(r1);m*phi_1(r2) m*phi_2(r2) m*phi_n(r2);m*phi_1(rn) m*phi_2(rn) m*phi_n(rn)]
			C=pb.matrix(pb.squeeze(sensor_kernel_conv_Phi_values).T)
			self.C=C
			print 'Elapsed time in seconds to calculate observation matrix C is',time.time()-t_observation_matrix	
			print 'Elapsed time in seconds to generate the model',time.time()-t_total



		#We need to calculate this bit at each iteration	
		##Finding the convolution of the kernel  with field basis functions at discritized spatial points
		self.Gamma_inv_Psi_conv_Phi=psi1_weight*self.Gamma_inv_psi1_conv_Phi+ \
		psi2_weight*self.Gamma_inv_psi2_conv_Phi+psi3_weight*self.Gamma_inv_psi3_conv_Phi












	def simulate(self,T):

		"""
		generates nonlinear IDE


		Arguments
		----------
		init_field: matrix
				initial state in a form of nx x 1
		T: ndarray
				simulation time instants
		Returns
		----------
		X: list of matrix
			each matrix is the state vector at a time instant

		Y: list of matrix
			each matrix is the observation vector corrupted with noise at a time instant
		"""


		Y = []		#at t=0 we don't have observation
		X = []		#I don't save the initial state and X starts at t=1
		x=self.x0
		firing_rate_temp=x.T*self.Phi_values
		#firing_rate=pb.array([self.act_fun(firing_rate_temp[0,i]) for i in range(len(self.Phi_values_array))],ndmin=2).T
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
			#firing_rate=pb.array([self.act_fun(firing_rate_temp[0,i]) for i in range(len(self.Phi_values_array))],ndmin=2).T
			firing_rate=pb.array(self.act_fun.fmax/(1.+pb.exp(self.act_fun.varsigma*(self.act_fun.v0-firing_rate_temp)))).T		

		return X,Y




	def f(self,x):
		'''state sequence'''
		firing_rate_temp=x.T*self.Phi_values
		firing_rate=pb.array([self.act_fun(firing_rate_temp[0,i]) for i in range(len(self.Phi_values_array))],ndmin=2).T
		Gamma_inv_Psi_conv_Phi=pb.array(self.Gamma_inv_Psi_conv_Phi)
		g=pb.matrix(pb.dot(Gamma_inv_Psi_conv_Phi,firing_rate))
		g *=(self.spacestep**2)
		x=self.Ts*g+self.xi*x
		return x



	def  state_equation(self,x):
		'''state sequence very fast '''
		#t0=time.time()
		firing_rate_temp=x.T*self.Phi_values
 		firing_rate=pb.array(self.act_fun.fmax/(1.+pb.exp(self.act_fun.varsigma*(self.act_fun.v0-firing_rate_temp)))).T		
		Gamma_inv_Psi_conv_Phi=pb.array(self.Gamma_inv_Psi_conv_Phi)
		g=pb.matrix(pb.dot(Gamma_inv_Psi_conv_Phi,firing_rate))
		g *=(self.spacestep**2)
		x=self.Ts*g+self.xi*x
		#print time.time()-t0
		return x



	def  state_equation_trapz(self,x):
		'''state sequence very fast '''
		#t0=time.time()
		firing_rate_temp=x.T*self.Phi_values
 		firing_rate=pb.array(self.act_fun.fmax/(1.+pb.exp(self.act_fun.varsigma*(self.act_fun.v0-firing_rate_temp))))	
		Gamma_inv_Psi_conv_Phi=pb.array(self.Gamma_inv_Psi_conv_Phi)
		g=pb.matrix(pb.trapz(firing_rate*Gamma_inv_Psi_conv_Phi,dx=self.spacestep**2,axis=-1)).T
		x=self.Ts*g+self.xi*x
		#print time.time()-t0
		return x
			


class IDE_Kernel():

	"""class defining the Connectivity kernel of the brain.

	Arguments
	----------

	Psi:list
		list of connectivity kernel basis functions; must be of class Bases
	weights: list
		list of connectivity kernel weights	


	Attributes
	----------
	evaluate: 
		evaluate the kernel at a given spatial location
	"""
	
	def __init__(self,Psi,weights):
		
		self.Psi=Psi
		self.weights = weights



class IDE_Field():	

	"""class defining the field.

	Arguments
	----------

	Psi:list
		list of field basis functions; must be of class Bases

	Attributes
	----------
	evaluate: 
		evaluate the kernel at a given spatial location
	"""

	def __init__(self, Phi):

		self.Phi=Phi

	def plot(self,centers,color):

		# It is calculated based on Full width at half maximum,divided by 2 to give the radus
		radius=(2*pb.sqrt(pb.log(2))*(pb.sqrt(self.Phi[0,0].width)))/2 
		for center in centers:
			circle(center,radius,color)


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





class ukf():

	"""class defining the Unscented Kalman filter for nonlinear estimation.

	Arguments
	----------
	model: IDE instance
		Nonlinear IDE state space model
	kappa: float
		secondary scaling parameter, see A New Approach for Filtering Nonlinear Systems by Julier.
	alpha_sigma_points: 
		Determines the spread of sigma points around xbar.
	beta_sigma_points:
		is used to incorporate prior knowledge of the distribution of x
		beta=2 is optimal for Gaussian distribution
	

	Attributes
	----------
	L: int
		the dimension of the states
	"""

	def __init__(self,model,kappa=0.0,alpha_sigma_points=1e-3,beta_sigma_points=2):

		self.model=model
		self.alpha_sigma_points=alpha_sigma_points
		self.beta_sigma_points=beta_sigma_points
		self.L=self.model.nx
		self.kappa=3-self.L
		self.lamda=self.alpha_sigma_points**2*(self.L+self.kappa)-self.L    #lambda is a scaling parameter
		self.gamma_sigma_points=pb.sqrt(self.L+self.lamda) #gamma is the composite scaling parameter	

	def sigma_vectors(self,x,P):

		"""
		generator for the sigma vectors

		Arguments
		----------
		x : ndarray
			state at time instant t
		P:  ndarray
			state covariance matrix at time instant t

		Returns
		----------
		Chi : matrix
			matrix of sigma points, each column is a sigma vector: [x0 x0+ x0-]
		"""
		State_covariance_cholesky=sp.linalg.cholesky(P).T
		State_covariance_cholesky_product=self.gamma_sigma_points*State_covariance_cholesky		
		chi_plus=[]
		chi_minus=[]
		for i in range(self.L):
			chi_plus.append(x+State_covariance_cholesky_product[:,i].reshape(self.L,1)) #list of matrix with length nx
			chi_minus.append(x-State_covariance_cholesky_product[:,i].reshape(self.L,1)) #list of matrix with length nx

		Chi=pb.hstack((x,pb.hstack((pb.hstack(chi_plus),pb.hstack(chi_minus))))) #matrix nx by 2nx+1
		return pb.matrix(Chi)



	def sigma_vectors_weights(self):

		"""
		generator for the sigma vectors' weights

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
		Wm_i=pb.concatenate((Wm0,2*self.L*Wmc)) #ndarray 2n_x+1
		Wc_i=pb.concatenate((Wc0,2*self.L*Wmc)) #ndarray 2n_x+1
		return Wm_i,Wc_i

	def _filter(self,Y):

		## initialise
		mean=[0]*self.L
		xhat=self.model.x0
		P=self.model.P0
		#xhat=pb.multivariate_normal(mean,P).reshape(self.L,1)
		  
		# filter quantities
		xhatStore =[]
		PhatStore=[]


		# initialise the filter

		#calculate the weights
		Wm_i,Wc_i=self.sigma_vectors_weights()



		for y in Y:
			#calculate the sigma points matrix, each column is a sigma vector
			Chi=self.sigma_vectors(xhat,P)
			# update sigma points
			Chi_update=pb.matrix(pb.empty_like(Chi))
			for i in range(Chi.shape[1]):
				Chi_update[:,i]=self.model.state_equation(Chi[:,i])	
			#pointwise multiply by weights and sum along y-axis
			xhat_=pb.sum(pb.multiply(Wm_i,Chi_update),1)
			Chi_purturbation=Chi_update-xhat_

			weighted_Chi_purturbation=pb.multiply(Wc_i,Chi_purturbation)
			P_=Chi_purturbation*weighted_Chi_purturbation.T+self.model.Sigma_e

			
			#measurement update equation
			Pyy=self.model.C*P_*self.model.C.T+self.model.Sigma_varepsilon 
			Pxy=P_*self.model.C.T
			K=Pxy*(Pyy.I)
			yhat_=self.model.C*xhat_
			xhat=xhat_+K*(y-yhat_)
			P=(pb.eye(self.model.nx)-K*self.model.C)*P_
			xhatStore.append(xhat)
			PhatStore.append(P)

		return xhatStore,PhatStore

	def rtssmooth(self,Y):
		## initialise
		mean=[0]*self.L
		P=self.model.P0 
		xhat=self.model.x0
		#xhat=pb.multivariate_normal(mean,P).reshape(self.L,1)
		 
		# filter quantities


		xhatStore =[]
		PhatStore=[]
		# initialise the filter

		#calculate the weights
		Wm_i,Wc_i=self.sigma_vectors_weights()



		for y in Y:
			#calculate the sigma points matrix, each column is a sigma vector
			Chi=self.sigma_vectors(xhat,P)
			# update sima points
			Chi_update=pb.matrix(pb.empty_like(Chi))
			for i in range(Chi.shape[1]):
				Chi_update[:,i]=self.model.state_equation(Chi[:,i])	
			#pointwise multiply by weights and sum along y-axis
			xhat_=pb.sum(pb.multiply(Wm_i,Chi_update),1)
			#xhatPredStore.append(xhat_)
			Chi_purturbation=Chi_update-xhat_

			weighted_Chi_purturbation=pb.multiply(Wc_i,Chi_purturbation)
			P_=Chi_purturbation*weighted_Chi_purturbation.T+self.model.Sigma_e

						
			#measurement update equation
			Pyy=self.model.C*P_*self.model.C.T+self.model.Sigma_varepsilon 
			Pxy=P_*self.model.C.T
			K=Pxy*(Pyy.I)
			yhat_=self.model.C*xhat_
			xhat=xhat_+K*(y-yhat_)
			P=(pb.eye(self.model.nx)-K*self.model.C)*P_
			xhatStore.append(xhat)
			PhatStore.append(P)

		# initialise the smoother
		T=len(Y)
		xb = [None]*T
		Pb = [None]*T

		xb[-1], Pb[-1] = xhatStore[-1], PhatStore[-1]

		## smooth
		for t in range(T-2,-1,-1):
			#calculate the sigma points matrix from filterd states, each column is a sigma vector
			Chi_smooth=self.sigma_vectors(xhatStore[t],PhatStore[t]) #X_k
			Chi_smooth_update=pb.matrix(pb.empty_like(Chi))	#X_k+1	
			for i in range(Chi_smooth.shape[1]):
				Chi_smooth_update[:,i]=self.model.state_equation(Chi_smooth[:,i])
			
			#pointwise multiply by weights and sum along y-axis
			xhat_smooth_=pb.sum(pb.multiply(Wm_i,Chi_smooth_update),1) #m_k+1_
			#purturbation
			Chi_smooth_purturbation=Chi_smooth-xhatStore[t] #X_k-m_k
			Chi_smooth_update_purturbation=Chi_smooth_update-xhat_smooth_ #X_k+1-m_k+1_
			#weighting
			weighted_Chi_smooth_purturbation=pb.multiply(Wc_i,Chi_smooth_purturbation) #W_ci*(X_k-m_k)
			weighted_Chi_smooth_update_purturbation=pb.multiply(Wc_i,Chi_smooth_update_purturbation)#W_ci*(X_k+1-m_k+1_)

			P_smooth_=Chi_smooth_update_purturbation*weighted_Chi_smooth_update_purturbation.T+self.model.Sigma_e
			#(X_k+1-m_k+1_)*W_ci*(X_k+1-m_k+1_).T
			C_smooth=weighted_Chi_smooth_purturbation*Chi_smooth_update_purturbation.T
			#W_ci*(X_k-m_k)*(X_k+1-m_k+1_).T
			D=C_smooth*P_smooth_.I
			xb[t]=xhatStore[t]+D*(xb[t+1]-xhat_smooth_)
			Pb[t]=PhatStore[t]+D*(Pb[t+1]-P_smooth_)*D.T

			
		return xb,Pb,xhatStore,PhatStore






class para_state_estimation():

	def __init__(self,model):

		self.model=model


	def Q_calc(self,X):

		"""
			calculates Q (n_x by n_theta) matrix of the IDE model at a each time step
	
			Arguments
			----------
			X: list of matrix
				state vectors

			Returns
			---------
			Q : list of matrix (n_x by n_theta)
		"""


		Q=[]	
		T=len(X)
		for t in range(T):

			firing_rate_temp=X[t].T*self.model.Phi_values
			firing_rate=pb.array(self.model.act_fun.fmax/(1.+pb.exp(self.model.act_fun.varsigma*(self.model.act_fun.v0-firing_rate_temp)))).T		

			Gamma_inv_psi1_conv_Phi=pb.array(self.model.Gamma_inv_psi1_conv_Phi)
			Gamma_inv_psi2_conv_Phi=pb.array(self.model.Gamma_inv_psi2_conv_Phi)
			Gamma_inv_psi3_conv_Phi=pb.array(self.model.Gamma_inv_psi3_conv_Phi)

			sum1_s=pb.matrix(pb.dot(Gamma_inv_psi1_conv_Phi,firing_rate))
			sum2_s=pb.matrix(pb.dot(Gamma_inv_psi2_conv_Phi,firing_rate))
			sum3_s=pb.matrix(pb.dot(Gamma_inv_psi3_conv_Phi,firing_rate))

			g=pb.hstack((sum1_s,sum2_s,sum3_s))

			g *=(self.model.spacestep**2)	
			q=self.model.Ts*g
			Q.append(q)		
		return Q

	def estimate_kernel(self,X):

		"""
			estimate the ide model's kernel weights using Least Square method
	
			Arguments
			----------
			X: list of matrix
				state vectors

			Returns
			---------
			Least Square estimation of the IDE parameters, see the corresponding pdf file
		"""
		#Q is already multiplied by Ts
		Q=self.Q_calc(X)
		Z=pb.vstack(X[1:])
		X_t_1=pb.vstack(X[:-1])
		Q_t_1=pb.vstack(Q[:-1])
		X_ls=pb.hstack((Q_t_1,X_t_1))
		parameters=(X_ls.T*X_ls).I*X_ls.T*Z
		return [float( parameters[0]),float(parameters[1]),float(parameters[2]),float(parameters[3])]
 




	def itr_est(self,Y,max_it):

		"""estimate the ide's kernel and field weights """
		xi_est=[]
		kernel_weights_est=[]
		# form state space model
		#self.model.gen_ssmodel()
		# generate a random state sequence
		Xb= [pb.matrix(np.random.rand(self.model.nx,1)) for t in Y]
		#Xhat= [pb.matrix(np.random.rand(self.model.nx,1)) for t in Y]
		# iterate
		keep_going = 1
		it_count = 0
		print " Estimatiing IDE's kernel and field weights"
		t0=time.time()
		while keep_going:
			
			temp=self.estimate_kernel(Xb)
			#temp=self.estimate_kernel(Xhat)
			xi_est.append(float(temp[-1]))
			kernel_weights_est.append(temp[0:-1])
			self.model.kernel.weights,self.model.xi=temp[0:-1],float(temp[-1])
			self.model.gen_ssmodel()
			#_filter=getattr(ukf(self.model),'_filter')
			_filter=getattr(ukf(self.model),'rtssmooth')
			Xb,Pb,Xhat,Phat=_filter(Y)
			#Xhat,Phat=_filter(Y)
			self.model.Xb=Xb
			self.model.Pb=Pb
			self.model.Xhat=Xhat
			self.model.Phat=Phat
			self.model.xi_est=xi_est
			self.model.kernel_weights_est=kernel_weights_est
			#self.model.x0=Xhat[0]
			#self.model.P0=Phat[0]
			self.model.x0=Xb[0]
			self.model.P0=Pb[0]
			print it_count, " Kernel current estimate: ", self.model.kernel.weights, "xi", self.model.xi
			if it_count == max_it:
				keep_going = 0
			it_count += 1
		print "Elapsed time in seconds is", time.time()-t0




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


def basis_locns(spacestep,estimation_field_width,NBasisFunction_x_or_y_dir):

	'''Define the center of basis functions along x and y
		Arguments:
		----------
			spacestep: the spatial step for the estimation field
			estimation_field_width: the width of the estimation field
			NBasisFunction_x_or_y_dir: number of basis functions along x or y axis
		Returns:
		--------
			S: center of basis functions along x and y'''

	# in this part of the code we make sure the basis functions are symetric with the spatial discretisation
	steps_in_estimated_field = estimation_field_width/spacestep + 1;								#in space indexes
	#print "steps (indexes) in estimated field is", steps_in_estimated_field
	Dist_between_basis_func = pb.floor(steps_in_estimated_field / (NBasisFunction_x_or_y_dir - 1) ) 		#in space indexes
	#print "distance between basis functions (in space index) is", Dist_between_basis_func
	twice_offset = steps_in_estimated_field - Dist_between_basis_func * (NBasisFunction_x_or_y_dir-1) #in space indexes
	basis_func_offset = twice_offset/2.

	# we need to make sure that twice_offset/2 is an integer
	# if twice_offset/2 is not an integer the basis functions will be assymetrically placed in the field we are trying to estimate
	#print "offset (in index) = " ,basis_func_offset
	assert pb.mod(basis_func_offset,.5)==0, "offset for positioning basis functions not an int. Choose at different number of basis functions"
  
	#print "basis function offset ok"
	S_index = pb.arange(basis_func_offset,steps_in_estimated_field-basis_func_offset+Dist_between_basis_func,Dist_between_basis_func)
	S = (S_index - steps_in_estimated_field/2.) * spacestep
	
	#print "basis function indexes:", S_index
	#print "basis function centers:", S
	return S








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


def plot_field(X,Phi_values,observedfieldspace,save=0,dpi=300,filename='filename'):
	"""
		plots spatial field

		Arguments
		----------
		X: matrix
			State vector at a given time instant
		fbases: list of matrix
		each entery is the vector of basis functions: matrix
			matrix of nx x 1 dimension [phi_1(s) phi_2(s) ... phi_nx(s)].T

		f_space: array
			x,y of the spatial locations	
		Returns
		---------
		gives the plot of the spatial field  at a given time instant

	"""

	z=pb.vstack(pb.split(X.T*Phi_values,pb.sqrt(Phi_values.shape[1]),axis=1))



	params = {'axes.labelsize': 20,'text.fontsize': 20,'legend.fontsize': 10,'xtick.labelsize': 20,'ytick.labelsize': 20}
	pb.rcParams.update(params) 
	pb.imshow(z,origin='lower',aspect=1, alpha=1,extent=[min(observedfieldspace),max(observedfieldspace),min(observedfieldspace),max(observedfieldspace)])	
	pb.colorbar(shrink=.55) 
	pb.show()




def avi_sub(X,Xhat,Phi_values,observedfieldspace,filename,play=0):
	"""
		creat avi for the neural field

		Arguments
		----------
		X: list of matrix
			State vectors
		fbases: list of matrix
		each entery is the vector of basis functions: matrix
			matrix of nx x 1 dimension [phi_1(s) phi_2(s) ... phi_nx(s)].T

		f_space: array
			x,y of the spatial locations

		filename: 	
		Returns
		---------
		avi of the neural field

	"""

	files=[]
	filename=" "+filename

	for t in range(len(X)):

		
		z=pb.vstack(pb.split(X[t].T*Phi_values,pb.sqrt(Phi_values.shape[1]),axis=1))
		zhat=pb.vstack(pb.split(Xhat[t].T*Phi_values,pb.sqrt(Phi_values.shape[1]),axis=1))

		fig=pb.figure()		
		ax=fig.add_subplot(121)
		pb.imshow(z,origin='lower',aspect=1, alpha=1,vmin=z.min(),vmax=z.max(),extent=[min(observedfieldspace),max(observedfieldspace),min(observedfieldspace),max(observedfieldspace)])
		pb.colorbar(shrink=.55)
		ax=fig.add_subplot(122)
		pb.imshow(zhat,origin='lower',aspect=1, alpha=1,vmin=z.min(),vmax=z.max(),extent=[min(observedfieldspace),max(observedfieldspace),min(observedfieldspace),max(observedfieldspace)])
		pb.colorbar(shrink=.55)

		
		fname = '_tmp%05d.jpg'%t
		pb.savefig(fname,format='jpg')
		pb.close()
		files.append(fname)
		
	os.system("ffmpeg -r 5 -i _tmp%05d.jpg -y -an"+filename+".avi")
	# cleanup
	for fname in files: os.remove(fname)
	if play: os.system("vlc"+filename+".avi")


def avi_sub_v(V_matrix,Xhat,Phi_values,estimation_field_space,filename,play=0):
	"""
		creat avi for the neural field

		Arguments
		----------
		X: list of matrix
			State vectors
		fbases: list of matrix
		each entery is the vector of basis functions: matrix
			matrix of nx x 1 dimension [phi_1(s) phi_2(s) ... phi_nx(s)].T

		f_space: array
			x,y of the spatial locations

		filename: 	
		Returns
		---------
		avi of the neural field

	"""

	files=[]
	filename=" "+filename

	for t in range(len(Xhat)):


		zhat=pb.vstack(pb.split(Xhat[t].T*Phi_values,pb.sqrt(Phi_values.shape[1]),axis=1))

		fig=pb.figure()		
		ax=fig.add_subplot(121)
		pb.imshow(zhat,origin='lower',aspect=1, alpha=1,vmin=V_matrix[t].min(),vmax=V_matrix[t].max(),extent=[min(estimation_field_space),max(estimation_field_space),min(estimation_field_space),max(estimation_field_space)])
		pb.colorbar(shrink=.55)
		ax=fig.add_subplot(122)
		pb.imshow(V_matrix[t],origin='lower',aspect=1, alpha=1,vmin=V_matrix[t].min(),vmax=V_matrix[t].max(),extent=[min(estimation_field_space),max(estimation_field_space),min(estimation_field_space),max(estimation_field_space)])
		pb.colorbar(shrink=.55)

		
		fname = '_tmp%05d.jpg'%t
		pb.savefig(fname,format='jpg')
		pb.close()
		files.append(fname)
		
	os.system("ffmpeg -r 5 -i _tmp%05d.jpg -y -an"+filename+".avi")
	# cleanup
	for fname in files: os.remove(fname)
	if play: os.system("vlc"+filename+".avi")


def avi(X,Phi_values,estimation_field_space,filename,play=0):
	"""
		creat avi for the neural field

		Arguments
		----------
		X: list of matrix
			State vectors
		fbases: list of matrix
		each entery is the vector of basis functions: matrix
			matrix of nx x 1 dimension [phi_1(s) phi_2(s) ... phi_nx(s)].T

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

	for t in range(len(X)):
		z=pb.vstack(pb.split(X[t].T*Phi_values,pb.sqrt(Phi_values.shape[1]),axis=1))
		pb.figure()		
		pb.imshow(z,animated=True,origin='lower',aspect=1, alpha=1,extent=[min(estimation_field_space),max(estimation_field_space),min(estimation_field_space),max(estimation_field_space)])
		pb.colorbar()

		fname = '_tmp%05d.jpg'%t
		pb.savefig(fname,format='jpg')
		pb.close()
		files.append(fname)
	
	os.system("ffmpeg -r 5 -i _tmp%05d.jpg -y -an"+filename+".avi")
	# cleanup
	for fname in files: os.remove(fname)
	if play: os.system("vlc"+filename+".avi")






def plot_smooth_states(Xreal,Xb,Xest):
	T=len(Xest)
	for i in range(len(Xest[0])):
		x1=pb.zeros(T)
		x2=pb.zeros(T)
		x3=pb.zeros(T)
		for tau in range(T):
			x1[tau]=(float(Xreal[tau][i]))
			x2[tau]=(float(Xb[tau][i]))
			x3[tau]=(float(Xest[tau][i]))
		pb.plot(range(T),x1,'k',range(T),x2,':r',range(T),x3,':b')
		pb.show()
	
def plot_kernel(theta,stepsize):
	k=lambda s,weight,width:float(weight*pb.exp(-(1./(width))*(s)**2))
	space=pb.arange(-10,10+stepsize,stepsize)
	K=[]
	Kest=[]
	for i in space:
		K.append(k(i,10,3.24)+k(i,-8,5.76)+k(i,0.5,36))
		Kest.append(k(i,theta[0],3.24)+k(i,theta[1],5.76)+k(i,theta[2],36))
	pb.plot(space,K,'k')
	pb.plot(space,Kest,'k:')
	pb.plot(space,K,'o')
	pb.show()


def plot_kernel_FT(theta,stepsize):

	k_FT=lambda s,weight,width:float(weight*pb.exp(-(1./(width))*(s)**2))
	#Kernel basis functions' widths
	psi1_width=1.8**2
	psi2_width=2.4**2
	psi3_width=6.**2  

	psi1_weight_FT=10*(pb.pi*psi1_width)
	psi2_weight_FT=-8*(pb.pi*psi2_width)
	psi3_weight_FT=0.5*(pb.pi*psi3_width)

	psi1_weight_est_FT=theta[0]*(pb.pi*psi1_width)
	psi2_weight_est_FT=theta[1]*(pb.pi*psi2_width)
	psi3_weight_est_FT=theta[2]*(pb.pi*psi3_width)


	psi1_width_FT=1./(((pb.pi)**2)*psi1_width)
	psi2_width_FT=1./(((pb.pi)**2)*psi2_width)
	psi3_width_FT=1./(((pb.pi)**2)*psi3_width)

	#Kernel basis functions' widths(pb.pi*self.width)**(self.dimension*0.5)
	F=[]
	F_est=[]
	f=pb.linspace(0,1./stepsize,800)
	for i in f:
		F.append(k_FT(i,psi1_weight_FT,psi1_width_FT)+k_FT(i,psi2_weight_FT,psi2_width_FT)+k_FT(i,psi3_weight_FT,psi3_width_FT))
		F_est.append(k_FT(i,psi1_weight_est_FT,psi1_width_FT)+k_FT(i,psi2_weight_est_FT,psi2_width_FT)+k_FT(i,psi3_weight_est_FT,psi3_width_FT))
	fig=pb.figure()
	ax=fig.add_subplot(111)	
	ax.plot(f,F,'k')
	ax.plot(f,F_est,'k:')
	ax.axis([0,0.5,-20,20])
	ax.set_xlabel('Hz')
	pb.show()
		
	


