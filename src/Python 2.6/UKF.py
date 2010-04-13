import pylab as pb
import numpy as np
# import matplotlib.axes3d
import time,warnings
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

	def __init__(self,kernel,field,EEG,act_fun,alpha,beta,field_noise_variance,obs_noise_covariance,Sensorwidth,obs_locns,spacestep,init_field,initial_field_covariance,Ts):

		self.kernel = kernel
		self.field = field
		self.EEG=EEG
		self.act_fun = act_fun
		self.alpha=alpha
		self.beta=beta
		self.field_noise_variance=field_noise_variance
		self.obs_locns=obs_locns
		self.obs_noise_covariance=obs_noise_covariance
		self.spacestep=spacestep
		self.Sensorwidth=Sensorwidth
		self.init_field=init_field
		self.initial_field_covariance=initial_field_covariance
		self.Ts=Ts
		

	def gen_ssmodel(self,sim=0):

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

		K1: ndarray
			matrix of kernel values over the spatial domain of the kernel, needed for parameter estimation
		K2: ndarray
			matrix of kernel values over the spatial domain of the kernel, needed for parameter estimation
		K3: ndarray
			matrix of kernel values over the spatial domain of the kernel, needed for parameter estimation		


		fbases: list of matrix
				each matrix is nx x 1 dimension [phi_1(s) phi_2(s) ... phi_nx(s)].T calculated at a spatial location

		Psi_x:	matrix
				inner product of the field basis functions

		Psi_xinv: matrix
				inverse of Psi_x
		Sw: matrix
			covariance of the field
		Swc: matrix
			cholesky decomposiotion of Sw
		Svc: matrix
			cholesky decomposiotion of the observation noise covariance
			

		"""
		print "generating nonlinear IDE"

		# Here we find the index of the centers
		K_center=(pb.floor(self.kernel.space.shape[0]/2.),pb.floor(self.kernel.space.shape[0]/2.))


		# This is to check if the spatial domain of the kernel is odd, it must be odd in order 
		#to have center
		if ((pb.mod(self.kernel.space.shape[0],2.)) or (pb.mod(self.kernel.space.shape[1],2.)))==0:
			warnings.warn('Kernel doesnt have center')

		# Here we find the center of the kernel and the sensor they must be lcated at (0,0)
		print 'center of kernel spatial domain is',(self.kernel.space[K_center[0]],self.kernel.space[K_center[1]])
		print "pre-calculating kernel matrices"


		if sim==1:
			K = pb.empty((len(self.kernel.space),len(self.kernel.space)),dtype=object)

		if hasattr(self,'K2'):
			pass
		else:
			K1=pb.empty((len(self.kernel.space),len(self.kernel.space)),dtype=object)
			K2=pb.empty((len(self.kernel.space),len(self.kernel.space)),dtype=object)
			K3=pb.empty((len(self.kernel.space),len(self.kernel.space)),dtype=object)


			k1_width=self.kernel.widths[0]
			k2_width=self.kernel.widths[1]
			k3_width=self.kernel.widths[2]


			k1_center=self.kernel.centers[0]
			k2_center=self.kernel.centers[1]
			k3_center=self.kernel.centers[2]


		Beta = pb.empty((len(self.kernel.space),len(self.kernel.space)),dtype=object)
		for i in range(len(self.kernel.space)):
			for j in range(len(self.kernel.space)):
				if sim==1:
					K[i,j]=self.kernel.__call__(pb.matrix([[self.kernel.space[i]],[self.kernel.space[j]]]))	
				
				if hasattr(self,'Beta'):
					pass
				else:

					Beta[i,j]=self.beta.__call__(pb.matrix([[self.kernel.space[i]],[self.kernel.space[j]]]))
	
				
				if hasattr(self,'K3'):
					pass

				else:
					K1[i,j]=gaussian(pb.matrix([[self.kernel.space[i]],[self.kernel.space[j]]]),k1_center,k1_width,self.kernel.dimension)
					K2[i,j]=gaussian(pb.matrix([[self.kernel.space[i]],[self.kernel.space[j]]]),k2_center,k2_width,self.kernel.dimension)
					K3[i,j]=gaussian(pb.matrix([[self.kernel.space[i]],[self.kernel.space[j]]]),k3_center,k3_width,self.kernel.dimension)

		if hasattr(self,'K3'):
			pass
		else:

			#Padding the kernel with zeros

			field_width=2* pb.absolute(self.field.space[-1])
			#column
			K1[:,0:field_width/2.]=0 
			K1[:,K_center[0]+1+(field_width/2.):]=0

			K2[:,0:field_width/2.]=0 
			K2[:,K_center[0]+1+(field_width/2.):]=0

			K3[:,0:field_width/2.]=0 
			K3[:,K_center[0]+1+(field_width/2.):]=0
			#rows
 			K1[0:field_width/2]=0
 			K1[K_center[0]+1+(field_width/2.):]=0

 			K2[0:field_width/2]=0
 			K2[K_center[0]+1+(field_width/2.):]=0

 			K3[0:field_width/2]=0
 			K3[K_center[0]+1+(field_width/2.):]=0

			self.K1=K1
			self.K2=K2
			self.K3=K3

		if sim==1:


			#Padding the kernel with zeros
			field_width=2* pb.absolute(self.field.space[-1])

			#column
			K[:,0:field_width/2.]=0 
			K[:,K_center[0]+1+(field_width/2.):]=0

			#rows
 			K[0:field_width/2]=0
 			K[K_center[0]+1+(field_width/2.):]=0

			self.K=K

		if hasattr(self,'Beta'):
			pass
		else:


			#Padding the kernel with zeros
			field_width=2* pb.absolute(self.field.space[-1])

			#column
			Beta[:,0:field_width/2.]=0 
			Beta[:,K_center[0]+1+(field_width/2.):]=0

			#rows
 			Beta[0:field_width/2]=0
 			Beta[K_center[0]+1+(field_width/2.):]=0


			self.Beta=Beta

		if hasattr(self.field,'fbases'):
			pass
		else:

			print "pre-calculating basis function vectors"
			t0=time.time()
			fbases=[]
			for s1 in self.field.space:
				for s2 in self.field.space:
					fbases.append(self.field.field_bases(pb.matrix([[s1],[s2]])))
			self.field.fbases=fbases
			print "Elapsed time in seconds is", time.time()-t0


		
		if hasattr(self,'observation_matrix'):
			pass
		else:

			print "calculating observation matrix"
			t0=time.time()
			ob_kernels_values=[]
			for s1 in self.EEG.space: #EEG.space equals to field_space
				for s2 in self.EEG.space:
					ob_kernels_values.append(self.EEG.observation_kernels(pb.matrix([[s1],[s2]])))
			#we squeeze the result and transpose it to have a matrix in a form of
			#[m1(s1) m1(s2) ...m1(sl);m2(s1) m2(s2) ... m2(sl);...;mny(s1) mny(s2) ... mny(sl)] 
			#where sl is the number of spatial points after discritization
			EEG.ob_kernels_values=pb.matrix(pb.squeeze(ob_kernels_values).T)
			print "Elapsed time in seconds is", time.time()-t0
			#we squeeze the result to have a matrix in a form of
			#[phi_1(s1) phi_2(s1) ...phi_n(s1);phi_1(s2) phi_2(s2) ...phi_n(s2);...;phi_1(sl) phi_2(sl) ...phi_n(sl)] 
			#where sl is the number of spatial points after discritization
			field_bases_squeezed=pb.squeeze(self.field.fbases)
			#finding observation matrix using Broadcasting, making the field_bases_squeezed last dimension to 1
			#self.observation_matrix=EEG.ob_kernels_values*field_bases_squeezed[:,:,pb.newaxis]
			self.observation_matrix=pb.hstack([EEG.ob_kernels_values*pb.matrix(field_bases_squeezed[:,i]).T for i in range(field_bases_squeezed.shape[1])])
	
			


		if hasattr(self,'Psi_x'):
			pass
		else:

			print "calculating Psi_x"		
			Psi_x=(self.spacestep**2)*sum([f*f.T for f in self.field.fbases])
			self.Psi_x=Psi_x
			print "calculating Psix_inv"
			Psi_xinv=Psi_x.I
			self.Psi_xinv=Psi_x.I
		

		if hasattr(self,'Swc'):
			pass
		else:


			print "calculating field noise covariances"
			t0=time.time()
			Pi=self.field_noise()	
			Sw=self.field_noise_variance*Psi_xinv*Pi*Psi_xinv.T
			print "Elapsed time in seconds is", time.time()-t0
			#Sw=self.field_noise_variance*self.Psi_xinv white noise
			self.Sw=Sw
			try:
				self.Swc=sp.linalg.cholesky(self.Sw).T
			except pb.LinAlgError:

				raise
			Svc=sp.linalg.cholesky(self.obs_noise_covariance)
			self.Svc=Svc.T

	


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

		self.gen_ssmodel(sim=1)

		Y = []#at t=0 we don't have observation
		X = []#I don't save the initial state and X starts at t=1	
		x=self.init_field

		K_center=(pb.floor(self.K.shape[0]/2.),pb.floor(self.K.shape[1]/2.))


		firing_rate=np.array([self.act_fun(fr.T*x) for fr in self.field.fbases])
		print "iterating"
		for t in T[1:]:

			w = self.Swc*pb.matrix(np.random.randn(self.field.nx,1))
			v = self.Svc*pb.matrix(np.random.randn(len(self.obs_locns),1))


			#print "simulation at time",t

			Kernel_convolution_at_s=[]
			for i in range(len(self.field.space)):
				K_shift_x=pb.roll(self.K,i*len(self.kernel.space)) #shift kernel along x
				for j in range(len(self.field.space)):
					K_shift_y=pb.roll(K_shift_x,j) #shift kernel along y
					K_truncate=K_shift_y[K_center[0]:,K_center[1]:]
					Kernel_convolution_at_s.append(pb.sum(firing_rate*pb.ravel(K_truncate)))	
			sum_s=pb.hstack(self.field.fbases)*pb.matrix(Kernel_convolution_at_s).T
			

			sum_s *= (self.spacestep**4)
			x=self.Ts*self.Psi_xinv*sum_s+(1-self.Ts*self.alpha)*x+w
			X.append(x)
			Y.append((self.spacestep**2)*self.observation_matrix*x+v)
			firing_rate=np.array([self.act_fun(fr.T*x) for fr in self.field.fbases])

		return X,Y


	def field_noise(self):


		'''calculate the inner inner product of the field basis functions
		    and the convolution between field basis functions and the field covariance finction
		Output:
		-------
		Pi: matrix nx X nx
			
			
		'''

		t0=time.time()
		Beta_center=(pb.floor(self.kernel.space.shape[0]/2.),pb.floor(self.kernel.space.shape[0]/2.))
		#calculate the convolution between the first basis function and beta
		#Transform the fbases to ndarray of the form of number of spatial points x number of basis functions
		#fbases=[phi_1(s1) phi_2(s1)...phi_n(s1);phi_1(s2) phi_2(s2)...phi_n(s2);...;phi_1(sp) phi_2(sp)...phi_n(sp)]
		fbases=pb.array(pb.squeeze(self.field.fbases))
		#Calculate the convolution

		Beta_fbases_convolution_at_s=[]
		for l in range(self.field.nx):
			Beta_fbases_l_convolution_at_s=[]
			for i in range(len(self.field.space)):
				Beta_shift_x=pb.roll(self.Beta,i*len(self.kernel.space)) #shift beta along x
				for j in range(len(self.field.space)):
					Beta_shift_y=pb.roll(Beta_shift_x,j) #shift beta along y
					Beta_truncate=Beta_shift_y[Beta_center[0]:,Beta_center[1]:]
					Beta_fbases_l_convolution_at_s.append(pb.sum(fbases[:,l]*pb.ravel(Beta_truncate)))
			Beta_fbases_convolution_at_s.append(Beta_fbases_l_convolution_at_s)
	

		#calculating Pi
		Pi=pb.matrix(pb.empty((self.field.nx,self.field.nx)))
		for i in range(self.field.nx):
			for j in range(self.field.nx):
				Pi[i,j]=pb.sum(fbases[:,i]*Beta_fbases_convolution_at_s[j])
		return Pi*((self.spacestep)**4)





	

	def f(self,x):

		'''state sequence'''
	
		K_center=(pb.floor(self.K.shape[0]/2.),pb.floor(self.K.shape[1]/2.))
		firing_rate=np.array([self.act_fun(fr.T*x) for fr in self.field.fbases])

		Kernel_convolution_at_s=[]
		for i in range(len(self.field.space)):
			K_shift_x=pb.roll(self.K,i*len(self.kernel.space)) #shift kernel along x
			for j in range(len(self.field.space)):
				K_shift_y=pb.roll(K_shift_x,j) #shift kernel along y
				K_truncate=K_shift_y[K_center[0]:,K_center[1]:]
				Kernel_convolution_at_s.append(pb.sum(firing_rate*pb.ravel(K_truncate)))	
		sum_s=pb.hstack(self.field.fbases)*pb.matrix(Kernel_convolution_at_s).T
		sum_s *= (self.spacestep**4)
		x=self.Ts*self.Psi_xinv*sum_s+(1-self.Ts*self.alpha)*x
		return x
		
		
	

	def h(self,x):

		'''measurement equation'''
		y=(self.spacestep**2)*self.observation_matrix*x
		return y

class EEG():

	def __init__(self,centers, widths, dimension,space,ny,spacestep):
		
		
		self.centers=centers
		self.ny=ny
		self.widths=self.ny*widths
		self.dimension=dimension
		self.space = space
		self.spacestep=spacestep

	def observation_kernels(self,s):

		"""
		generates vector of observation kernels at a given spatial location

		Arguments
		----------
		s : float
			spatail location
		Returns
		----------
		vector of observation kernels: matrix
			matrix of ny x 1 dimension [m_1(s) m_2(s) ... m_ny(s)].T
		"""
		return pb.matrix([gaussian(s,cen,wid,self.dimension) for cen,wid, in zip(self.centers,self.widths)]).T



class Field():	

	def __init__(self, weights,centers, widths, dimension,space,nx,spacestep):

		self.nx=nx
		self.weights = weights
		self.widths=self.nx*widths
		self.centers=centers
		self.dimension=dimension
		self.space = space
		self.spacestep=spacestep
		self.evaluate = lambda s: sum([w*gaussian(s,cen,wid,self.dimension) for w,cen,wid, in zip(self.weights,self.centers,self.widths)
				])

	def __call__(self,s):
		return float(self.evaluate(s))		

	def field_bases(self,s):

		"""
		generates vector of field basis functions at a given spatial location

		Arguments
		----------
		s : float
			spatail location
		Returns
		----------
		vector of basis functions: matrix
			matrix of nx x 1 dimension [phi_1(s) phi_2(s) ... phi_nx(s)].T
		"""
		return pb.matrix([gaussian(s,cen,wid,self.dimension) for cen,wid, in zip(self.centers,self.widths)]).T




	def plot(self,centers,color):
		radius=(2*pb.sqrt(pb.log(2))*(pb.sqrt(self.widths[0][0,0])))/2 # It is calculated based on Full width at half maximum,divided by 2 to give the radus
		for center in centers:
			circle(center,radius,color)



class Kernel():

	"""class defining the Connectivity kernel of the brain.

	Arguments
	----------
	weights: list
		
	centers: list of matrix
			each of the entery should be a matrix in a form of (2x1)
	widths: list of matrix
			each 2x2 matrix is the width of the corresponding basis function.

	dimension: int
			dimension of the kernel
	space: array
			spatial domain of the kernel


	Attributes
	----------
	evaluate: 
		evaluate the kernel at a given spatial location
	"""
	
	def __init__(self, weights, centers, widths, space, dimension):
		self.weights = weights
		self.centers=centers
		self.widths = widths
		self.dimension=dimension
		self.space=space
		self.evaluate = lambda s: sum([w*gaussian(s,cen,wid,dimension) for w,cen,wid, in zip(self.weights,self.centers,self.widths)
				])

	def __call__(self,s):
		"""
		evaluates the kernel at a given spatial location

		Arguments
		----------
		s : matrix
			spatail location, it must be in a form of dimension x 1

		"""
		return self.evaluate(s)


	def plot(self,space):

		y = pb.zeros((space.shape[0],space.shape[0]))
		for i in range(len(space)):
			for j in range(len(space)):
				y[i,j]=self.__call__(pb.matrix([[space[i]],[space[j]]]))			
		fig = pb.figure()
		ax = matplotlib.axes3d.Axes3D(fig)
		s1,s2=pb.meshgrid(space,space)
		params = {'axes.labelsize': 20,'text.fontsize': 20,'legend.fontsize': 10,'xtick.labelsize': 20,'ytick.labelsize': 20}
		pb.rcParams.update(params) 
		ax.plot_wireframe(s1,s2,y,color='k')
		ax.set_xlabel(r'$ s_1-r_1$',fontsize=18)
		ax.set_ylabel(r' $s_2-r_2$',fontsize=18)
		ax.set_zlabel(r' $k(s-r)$',fontsize=18)


		pb.show()

class FieldCovarianceFunction():

	"""
		defines covariance function of the field

		Arguments
		----------
			
		center:  matrix
			it should be a matrix in form of dimension x 1
		width: matrix
			*variance or *covariance matrix of the Gaussian function
		dimension: int
			dimension of the basis function
		methos
		----------
		evaluate:
			Returns the value of the covariance function at a given spatial location, it should be a matrix dimension x 1

	"""

	def __init__(self, center,width,dimension):
		self.center=center
		self.width = width
		self.dimension=dimension
		self.evaluate = lambda s: gaussian(s,center,width,dimension)
	
	def __call__(self,s):
		"""
		evaluates the covariance function at a given spatial location

		Arguments
		----------
		s : matrix
			spatail location, it must be in a form of dimension x 1

		"""
		return self.evaluate(s)
	

		
class ActivationFunction():

	"""class defining the sigmoidal activation function .

	Arguments
	----------
	threshold: float
		firing threshold, mV                  (Wendling, 2002, 6 mV), (Schiff, 2007, threshold = 0.24 (Heaviside))
	nu: float
		maximum firing rate, spikes/s         (Wendling, 2002, 2*e0 = 5, or nu = 5
	beta: float
		slope of sigmoid, spikes/mV           (Wendling, 2002, 0.56 mV^-1)

	v: float
		Presynaptic potential

	Returns
	----------
	average firing rate
	"""	
	def __init__(self,threshold=6,nu=1,beta=0.56):
		
		self.threshold = threshold
		self.nu = nu
		self.beta = beta
		
	def __call__(self,v):

		return float(self.nu/(1.+pb.exp(self.beta*(self.threshold-v))))
	def plot(self,plot_range):
		u=pb.linspace(-plot_range,plot_range,1000)
		z=pb.zeros_like(u)
		for i,j in enumerate(u):
			z[i]=self.__call__(j)
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
		self.kappa=kappa
		self.alpha_sigma_points=alpha_sigma_points
		self.beta_sigma_points=beta_sigma_points
		self.L=self.model.field.nx


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
		self.kappa=3-self.L
		lamda=self.alpha_sigma_points**2*(self.L+self.kappa)-self.L    #lambda is a scaling parameter
		self.lamda=lamda
		gamma=pb.sqrt(self.L+lamda)                     #gamma is the composite scaling parameter	
		State_covariance_cholesky=pb.cholesky(P)
		State_covariance_cholesky_product=gamma*State_covariance_cholesky		
		chi_plus=[]
		chi_minus=[]
		for i in range(self.L):
			chi_plus.append(x+State_covariance_cholesky_product[:,i].reshape(self.L,1))
			chi_minus.append(x-State_covariance_cholesky_product[:,i].reshape(self.L,1))

		Chi=pb.hstack((x,pb.hstack((pb.hstack(chi_plus),pb.hstack(chi_minus)))))
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
		lamda=(self.alpha_sigma_points**2)*(self.L+self.kappa)-self.L    #lambda is a scaling parameter
		Wm0=[lamda/(lamda+self.L)]
		Wc0=[(lamda/(lamda+self.L))+1-self.alpha_sigma_points**2+self.beta_sigma_points]
		Wmc=[1./(2*(self.L+lamda))]
		Wm_i=pb.concatenate((Wm0,2*self.L*Wmc))
		Wc_i=pb.concatenate((Wc0,2*self.L*Wmc))
		return Wm_i,Wc_i

	def _filter(self,Y):

		## initialise
		mean=[0]*self.L
		xhat=self.model.init_field
		P=self.model.initial_field_covariance
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
				Chi_update[:,i]=self.model.f(Chi[:,i])	
			#pointwise multiply by weights and sum along y-axis
			xhat_=pb.sum(pb.multiply(Wm_i,Chi_update),1)
			Chi_purturbation=Chi_update-xhat_

			weighted_Chi_purturbation=pb.multiply(Wc_i,Chi_purturbation)
			P_=Chi_purturbation*weighted_Chi_purturbation.T+self.model.Sw

			
			#measurement update equation
			Pyy=self.model.observation_matrix*P_*self.model.observation_matrix.T+self.model.obs_noise_covariance
			Pxy=P_*self.model.observation_matrix.T
			K=Pxy*(Pyy.I)
			yhat_=self.model.observation_matrix*xhat_
			xhat=xhat_+K*(y-yhat_)
			P=(pb.eye(self.model.field.nx)-K*self.model.observation_matrix)*P_
			xhatStore.append(xhat)
			PhatStore.append(P)

		return xhatStore,PhatStore

	def rtssmooth(self,Y):
		## initialise
		mean=[0]*self.L
		P=self.model.initial_field_covariance 
		xhat=self.model.init_field
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
				Chi_update[:,i]=self.model.f(Chi[:,i])	
			#pointwise multiply by weights and sum along y-axis
			xhat_=pb.sum(pb.multiply(Wm_i,Chi_update),1)
			#xhatPredStore.append(xhat_)
			Chi_purturbation=Chi_update-xhat_

			weighted_Chi_purturbation=pb.multiply(Wc_i,Chi_purturbation)
			P_=Chi_purturbation*weighted_Chi_purturbation.T+self.model.Sw

						
			#measurement update equation
			Pyy=self.model.observation_matrix*P_*self.model.observation_matrix.T+self.model.obs_noise_covariance
			Pxy=P_*self.model.observation_matrix.T
			K=Pxy*(Pyy.I)
			yhat_=self.model.observation_matrix*xhat_
			xhat=xhat_+K*(y-yhat_)
			P=(pb.eye(self.model.field.nx)-K*self.model.observation_matrix)*P_
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
				Chi_smooth_update[:,i]=self.model.f(Chi_smooth[:,i])
			
			#pointwise multiply by weights and sum along y-axis
			xhat_smooth_=pb.sum(pb.multiply(Wm_i,Chi_smooth_update),1) #m_k+1_
			#purturbation
			Chi_smooth_purturbation=Chi_smooth-xhatStore[t] #X_k-m_k
			Chi_smooth_update_purturbation=Chi_smooth_update-xhat_smooth_ #X_k+1-m_k+1_
			#weighting
			weighted_Chi_smooth_purturbation=pb.multiply(Wc_i,Chi_smooth_purturbation) #W_ci*(X_k-m_k)
			weighted_Chi_smooth_update_purturbation=pb.multiply(Wc_i,Chi_smooth_update_purturbation)#W_ci*(X_k+1-m_k+1_)

			P_smooth_=Chi_smooth_update_purturbation*weighted_Chi_smooth_update_purturbation.T+self.model.Sw
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
		K_center=(pb.floor(self.model.K1.shape[0]/2.),pb.floor(self.model.K2.shape[1]/2.))#all centers are equal


		

		T=len(X)
		for t in range(T):

			firing_rate=np.array([self.model.act_fun(fr.T*X[t]) for fr in self.model.field.fbases])
			K1_convolution_at_s=[]
			K2_convolution_at_s=[]
			K3_convolution_at_s=[]

			for i in range(len(self.model.field.space)):
				K1_shift_x=pb.roll(self.model.K1,i*len(self.model.kernel.space)) #shift kernel along x
				K2_shift_x=pb.roll(self.model.K2,i*len(self.model.kernel.space))
				K3_shift_x=pb.roll(self.model.K3,i*len(self.model.kernel.space))
				for j in range(len(self.model.field.space)):
					K1_shift_y=pb.roll(K1_shift_x,j) #shift kernel along y
					K2_shift_y=pb.roll(K2_shift_x,j)
					K3_shift_y=pb.roll(K3_shift_x,j)

					K1_truncate=K1_shift_y[K_center[0]:,K_center[1]:]
					K2_truncate=K2_shift_y[K_center[0]:,K_center[1]:]
					K3_truncate=K3_shift_y[K_center[0]:,K_center[1]:]

					K1_convolution_at_s.append(pb.sum(firing_rate*pb.ravel(K1_truncate)))
					K2_convolution_at_s.append(pb.sum(firing_rate*pb.ravel(K2_truncate)))
					K3_convolution_at_s.append(pb.sum(firing_rate*pb.ravel(K3_truncate)))



			K_convolution_at_s=pb.hstack((pb.matrix(K1_convolution_at_s).T,pb.matrix(K2_convolution_at_s).T,pb.matrix(K3_convolution_at_s).T))
			sum_s=pb.hstack(self.model.field.fbases)*K_convolution_at_s
			sum_s *= (self.model.spacestep**4) #2 for convolution and two for multiplying by phi and integrating
			q=self.model.Ts*self.model.Psi_xinv*sum_s
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
		Q=self.Q_calc(X)
		X_t=pb.vstack(X[1:])
		X_t_1=pb.vstack(X[:-1])
		Q_t_1=pb.vstack(Q[:-1])
		Phi=pb.hstack((Q_t_1,-X_t_1*self.model.Ts))
		return (Phi.T*Phi).I*Phi.T*(X_t-X_t_1)



	def itr_est(self,Y,max_it):

		"""estimate the ide's kernel and field weights """
		# form state soace model
		self.model.gen_ssmodel()
		# generate a random state sequence
		#Xb= [pb.matrix(np.random.rand(self.model.field.nx,1)) for t in Y]
		Xhat= [pb.matrix(np.random.rand(self.model.field.nx,1)) for t in Y]
		# iterate
		keep_going = 1
		it_count = 0
		print " Estimatiing IDE's kernel and field weights"
		t0=time.time()
		while keep_going:
			#temp=self.estimate_kernel(Xb)
			temp=self.estimate_kernel(Xhat)
			self.model.kernel.weights,self.model.alpha=temp[0:-1],float(temp[-1])
			_filter=getattr(ukf(self.model),'_filter')
			#_filter=getattr(ukf(self.model),'rtssmooth')
			#Xb,Pb,Xhat,Phat=_filter(Y)
			Xhat,Phat=_filter(Y)
			#self.Xb=Xb
			#self.Pb=Pb
			self.Xhat=Xhat
			self.Phat=Phat
			print it_count, " Kernel current estimate: ", self.model.kernel.weights.T, "alpha", self.model.alpha
			if it_count == max_it:
				keep_going = 0
			it_count += 1
		print "Elapsed time in seconds is", time.time()-t0

def gaussian(s,centre,width,dimension):
		"""
		defines Gaussian basis function

		Arguments
		----------
		s : matrix
			it should be dimension x 1
			
		center:  matrix
			it should be a matrix in form of dimension x 1
		width: matrix
			*variance or *covariance matrix of the Gaussian function
		dimension: int
			dimension of the basis function
		"""
		centre = pb.matrix(centre)
		centre.shape = (dimension,1)
		width = pb.matrix(width)
		return float(pb.exp(-(s-centre).T*width.I*(s-centre)))

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
	



	
def gen_obs_locations(FieldWidth,SensorWidth,SensorSpacing,BoundryEffectWidth):
	'''
	SensorWidth:  diameter in mm	
	SensorSpacing  in mm
	BoundryEffectWidth 
	'''
	ObservationCentre = pb.linspace(-FieldWidth/2. + BoundryEffectWidth,FieldWidth/2. - BoundryEffectWidth,FieldWidth/SensorSpacing)
	#ObservationCentre = pb.arange(-FieldWidth/2. + BoundryEffectWidth,FieldWidth/2. - BoundryEffectWidth+1,SensorSpacing)
	return [np.matrix([[i,j]]).T for i in ObservationCentre for j in ObservationCentre]





def gen_spatial_lattice(S):
	"""generates a list of vectors, where each vector is a co-ordinate in a lattice"""
	number_of_points = len(S)**2

	space = [0 for i in range(number_of_points)]
	count = 0
	for i in S:
		for j in S:
			space[count] = pb.matrix([[i],[j]])
			count += 1
	return space




def field_centers(FieldWidth,sep):
	x_center_outer =pb.linspace(-FieldWidth/2.,FieldWidth/2.,FieldWidth/sep);
	y_center_outer = x_center_outer
	distance_between_centers = abs(x_center_outer[1]-x_center_outer[0]);
	f_centers_outer=[np.array([[i,j]]).T for i in x_center_outer for j in y_center_outer]
	x_center_inner =pb.linspace(-FieldWidth/2.+distance_between_centers/2.,FieldWidth/2.-distance_between_centers/2.,FieldWidth/sep-1);
	y_center_inner = x_center_inner
	f_centers_inner=[np.array([[i,j]]).T for i in x_center_inner for j in y_center_inner]
	f_centers=np.concatenate([f_centers_outer,f_centers_inner])
	
	return [pb.matrix(x) for x in f_centers]

def subplot_field(X,Xhat,fbases,f_space):
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
		gives the subplot of the spatial field and its estimate at a given time instant

	"""
	fig=pb.figure()
	z=pb.zeros((len(f_space),len(f_space)))
	zhat=pb.zeros((len(f_space),len(f_space)))
	m=0
	for i in range(len(f_space)):
		for j in range(len(f_space)):
			z[i,j]=float(X.T*fbases[m]) 
			zhat[i,j]=float(Xhat.T*fbases[m])
			m+=1
	ax=fig.add_subplot(121)
	pb.imshow(z,origin='lower',aspect=1, alpha=1,vmin=z.min(),vmax=z.max(),extent=[min(f_space),max(f_space),min(f_space),max(f_space)])	
	pb.colorbar(shrink=.55) 
	ax=fig.add_subplot(122)
	pb.imshow(zhat,origin='lower',aspect=1, alpha=1,vmin=z.min(),vmax=z.max(),extent=[min(f_space),max(f_space),min(f_space),max(f_space)])	
	pb.colorbar(shrink=.55)
	pb.show()


def avi_sub(X,Xhat,fbases,f_space,filename,play=0):
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
		z=pb.zeros((len(f_space),len(f_space)))
		zhat=pb.zeros((len(f_space),len(f_space)))
		m=0
		for i in range(len(f_space)):
			for j in range(len(f_space)):
				z[i,j]=float(X[t].T*fbases[m])
				zhat[i,j]=float(Xhat[t].T*fbases[m])
				m+=1
		fig=pb.figure()		
		ax=fig.add_subplot(121)
		pb.imshow(z,origin='lower',aspect=1, alpha=1,vmin=z.min(),vmax=z.max(),extent=[min(f_space),max(f_space),min(f_space),max(f_space)])
		pb.colorbar(shrink=.55)
		ax=fig.add_subplot(122)
		pb.imshow(zhat,origin='lower',aspect=1, alpha=1,vmin=z.min(),vmax=z.max(),extent=[min(f_space),max(f_space),min(f_space),max(f_space)])	
		pb.colorbar(shrink=.55)
		
		fname = '_tmp%05d.jpg'%t
		pb.savefig(fname,format='jpg')
		pb.close()
		files.append(fname)
	
	os.system("ffmpeg -r 5 -i _tmp%05d.jpg -y -an"+filename+".avi")
	# cleanup
	for fname in files: os.remove(fname)
	if play: os.system("vlc"+filename+".avi")



def plot_field_error_V(V_matrix,Xhat,fbases,f_space,save=0,dpi=300,filename='filename'):
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
		gives the subplot of the spatial field and its estimate at a given time instant

	"""
	zhat=pb.zeros((len(f_space),len(f_space)))
	m=0
	for i in range(len(f_space)):
		for j in range(len(f_space)):
			zhat[i,j]=float(Xhat.T*fbases[m])
			m+=1
	error=V_matrix-zhat
	params = {'axes.labelsize': 20,'text.fontsize': 20,'legend.fontsize': 10,'xtick.labelsize': 20,'ytick.labelsize': 20}
	pb.rcParams.update(params) 
	pb.imshow(error,origin='lower',aspect=1, alpha=1,extent=[min(f_space),max(f_space),min(f_space),max(f_space)])	
	pb.colorbar(shrink=.55) 
	pb.xlabel(r'$ s_1$',fontsize=25)
	pb.ylabel(r' $s_2$',fontsize=25)

	if save:
		pb.savefig(filename+'.pdf',dpi=dpi)
	pb.show()	


def plot_field(X,fbases,f_space,save=0,dpi=300,filename='filename'):
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

	z=pb.zeros((len(f_space),len(f_space)))

	m=0
	for i in range(len(f_space)):
		for j in range(len(f_space)):
			z[i,j]=float(X.T*fbases[m]) 

			m+=1
	params = {'axes.labelsize': 20,'text.fontsize': 20,'legend.fontsize': 10,'xtick.labelsize': 20,'ytick.labelsize': 20}
	pb.rcParams.update(params) 
	pb.imshow(z,origin='lower',aspect=1, alpha=1,extent=[min(f_space),max(f_space),min(f_space),max(f_space)])	
	pb.colorbar(shrink=.55) 

	pb.xlabel(r'$ s_1$',fontsize=25)
	pb.ylabel(r' $s_2$',fontsize=25)
	if save:
		pb.savefig(filename+'.pdf',dpi=dpi)
	pb.show()	




def avi(X,fbases,f_space,filename,play=0):
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
		z=pb.zeros((len(f_space),len(f_space)))
		m=0
		for i in range(len(f_space)):
			for j in range(len(f_space)):
				z[i,j]=float(X[t].T*fbases[m]) 
				m+=1
		pb.figure()		
		pb.imshow(z,animated=True,origin='lower',aspect=1, alpha=1,extent=[min(f_space),max(f_space),min(f_space),max(f_space)])
		pb.colorbar()

		fname = '_tmp%05d.jpg'%t
		pb.savefig(fname,format='jpg')
		pb.close()
		files.append(fname)
	
	os.system("ffmpeg -r 5 -i _tmp%05d.jpg -y -an"+filename+".avi")
	# cleanup
	for fname in files: os.remove(fname)
	if play: os.system("vlc"+filename+".avi")

def plot_field_estimate_std(fbases,Pb,f_space,save=0,dpi=300,filename='filename'):
	'''Calculate the covariance of the field estimate at a given time

	Arguments:
	----------
	fbases: List of matrix
			each matrix is the vector of field basis functions at a spatial location
			L[0]=[phi_1(s_1) phi_2(s_1) ... phi_n(s_p)]; n:no of basis functions p:no of spatial locations
	Pb: matrix
		the covariance matrix of the state estimates from snoother at a given time
	Returns
	-------
	the covariance of the field estimate at a given time: matrix
			
	'''

	z=pb.zeros((len(f_space),len(f_space)))
	m=0
	for i in range(len(f_space)):
		for j in range(len(f_space)):
			z[i,j]=fbases[m].T*Pb*fbases[m]
			m+=1
	params = {'axes.labelsize': 20,'text.fontsize': 20,'legend.fontsize': 10,'xtick.labelsize': 20,'ytick.labelsize': 20}
	pb.rcParams.update(params) 
	pb.imshow(pb.sqrt(z),origin='lower',aspect=1, alpha=1,extent=[min(f_space),max(f_space),min(f_space),max(f_space)])	
	pb.colorbar(shrink=.55)
	pb.xlabel(r'$ s_1$',fontsize=25)
	pb.ylabel(r' $s_2$',fontsize=25)
	if save:
		pb.savefig(filename+'.pdf',dpi=dpi)
	pb.show()

def plot_states(Xreal,Xest,n):
	T=len(Xest)
	x1=[]
	x2=[]
	for i in range(len(Xreal)):
		x1.append(float(Xreal[i][n]))
		x2.append(float(Xest[i][n]))
	pb.plot(range(T),x1,'k',range(T),x2,':k')
	pb.show()

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


def plot_observations(Y):
	"""
		plots EEG signals for all channels

		Arguments
		----------
		Y: list of matrix
			obbservation vectors
		Returns
		---------
		plots EEG signals for all channels
	"""
	Y_array=pb.squeeze(Y) #each matrix in the list is placed in a raw in a ndarray Y_array[0,:]=Y[0]
	for i in range(len(Y[0])):
		pb.plot(Y_array[:,i])
	
	pb.show()

def postsynaptic_kernel(alpha,t):
	u=pb.linspace(0,t,1000)
	z=pb.zeros_like(u)
	for i,j in enumerate(u):
		z[i]=pb.exp(-alpha*j)
	pb.plot(u,z)
	pb.show()

def rmse(Xreal,Xest,n):
	Xrealn=[]
	Xestn=[]
	for i in range(len(Xreal)):
		Xrealn.append(Xreal[i][n])
		Xestn.append(Xest[i][n])
	return pb.sqrt((pb.mean(pb.array(Xrealn)-pb.array(Xestn)))**2)

def field_frequency_response(X,f_space,fbases,stepsize):
	"""
		calculate and plot average frequency response of the field

		Arguments
		----------
		X: list of matrix
			State vectors
		fbases: list of matrix
		each entery is the vector of basis functions: matrix
			matrix of nx x 1 dimension [phi_1(s) phi_2(s) ... phi_nx(s)].T

		f_space: array
			x,y of the spatial locations

		stepsize: float
			distance between field basis functions

		 	
		Returns
		---------
		average frequency response: array
		frequency range: array
		plot average frequency response
	"""
	Z=[]
	for t in range(len(X)):
		z=pb.zeros((len(f_space),len(f_space)))
		m=0
		for i in range(len(f_space)):
			for j in range(len(f_space)):
				z[i,j]=float(X[t].T*fbases[m]) 
				m+=1
		Z.append(z)
	fresponse=0
	for i in range(len(Z)):
		fresponse=fresponse+pb.absolute(pb.fft2(Z[i]))

	fresponse=fresponse/len(Z)
	freq=pb.fftfreq(fresponse.shape[0],float(stepsize)) #generates the frequency array [0,pos,neg] over which fft2 is taken
	Nyquist_freq=freq.max() #find the Nyquist frequency it is not exact (depending y.shape[0] is odd or even) but close to Nyquist frequency
	freq_range=2*Nyquist_freq #the frequency range over which fft2 is taken
	pb.imshow(fresponse,origin='lower',extent=[0,freq_range,0,freq_range],interpolation='nearest',cmap=pb.cm.gray,vmin=fresponse.min(),vmax=fresponse.max())
	pb.colorbar(shrink=.55)
	pb.show()
	return pb.flipud(fresponse.T),freq


def subplot_field_V(V_matrix,Xhat,fbases,f_space):
	"""
		plots spatial field (before discritization and after discritization)

		Arguments
		----------
		V_matrix:
			field matrix before discritization
		
		X: matrix
			State vector at a given time instant
		fbases: list of matrix
		each entery is the vector of basis functions: matrix
			matrix of nx x 1 dimension [phi_1(s) phi_2(s) ... phi_nx(s)].T

		f_space: array
			x,y of the spatial locations	
		Returns
		---------
		gives the subplot of the spatial field and its estimate at a given time instant

	"""
	fig=pb.figure()
	zhat=pb.zeros((len(f_space),len(f_space)))
	m=0
	for i in range(len(f_space)):
		for j in range(len(f_space)):
			zhat[i,j]=float(Xhat.T*fbases[m])
			m+=1
	ax=fig.add_subplot(121)
	pb.imshow(zhat,origin='lower',aspect=1, alpha=1,vmin=V_matrix.min(),vmax=V_matrix.max(),extent=[min(f_space),max(f_space),min(f_space),max(f_space)])	
	pb.colorbar(shrink=.55) 
	ax=fig.add_subplot(122)
	pb.imshow(V_matrix,aspect=1,origin='lower', alpha=1,vmin=V_matrix.min(),vmax=V_matrix.max(),extent=[min(f_space),max(f_space),min(f_space),max(f_space)])	
	pb.colorbar(shrink=.55)
	pb.show()

def plot_field_V(V,f_space,save=0,dpi=300,filename='filename'):

	"""
		plots spatial field

		Arguments
		----------
		V: matrix
			Spatial field at each time instant

		f_space: array
			x,y of the spatial locations	
		Returns
		---------
		plot the spatial field at a given time instant

	"""
	params = {'axes.labelsize': 25,'text.fontsize': 25,'legend.fontsize': 25,'xtick.labelsize': 25,'ytick.labelsize': 25}
	pb.imshow(V,aspect=1,origin='lower', alpha=1,extent=[min(f_space),max(f_space),min(f_space),max(f_space)])#interpolation='nearest',	
	pb.colorbar(shrink=.55) 
	pb.xlabel(r'$ s_1$',fontsize=25)
	pb.ylabel(r' $s_2$',fontsize=25)
	if save:
		pb.savefig(filename+'.pdf',dpi=dpi)
	pb.show()	


