import pylab as pb
import numpy as np
import matplotlib.axes3d
import time,warnings
import os, pp

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
			self.K1=K1
			self.K2=K2
			self.K3=K3
		if sim==1:
			self.K=K

		if hasattr(self,'Beta'):
			pass
		else:

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


		
		if hasattr(self.EEG,'ob_kernels_values'):
			pass
		else:

			print "pre-calculating observation kernel vectors"
			t0=time.time()
			ob_kernels_values=[]
			for s1 in self.EEG.space:
				for s2 in self.EEG.space:
					ob_kernels_values.append(self.EEG.observation_kernels(pb.matrix([[s1],[s2]])))
			#we squeeze the result to have a matrix in a form of
			#[m1(s1) m1(s2) ...m1(sl);m2(s1) m2(s2) ... m2(sl);...;mny(s1) mny(s2) ... mny(sl)]
			#where sl is the number of spatial points after discritization
			self.EEG.ob_kernels_values=pb.matrix(pb.squeeze(ob_kernels_values).T)
			print "Elapsed time in seconds is", time.time()-t0


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
				self.Swc=pb.linalg.cholesky(Sw)
			except pb.LinAlgError:

				raise
			Svc=pb.linalg.cholesky(self.obs_noise_covariance)
			self.Svc=Svc

	


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


		field_update=np.array([self.act_fun(fr.T*x) for fr in self.field.fbases])
		print "iterating"
		for t in T[1:]:

			w = self.Swc*pb.matrix(np.random.randn(self.field.nx,1))
			v = self.Svc*pb.matrix(np.random.randn(len(self.obs_locns),1))


			print "simulation at time",t

			Kernel_convolution_at_s=[]
			for i in range(len(self.field.space)):
				K_shift_x=pb.roll(self.K,i*len(self.kernel.space)) #shift kernel along x
				for j in range(len(self.field.space)):
					K_shift_y=pb.roll(K_shift_x,j) #shift kernel along y
					K_truncate=K_shift_y[K_center[0]:,K_center[1]:]
					Kernel_convolution_at_s.append(pb.sum(field_update*pb.ravel(K_truncate)))	
			sum_s=pb.hstack(self.field.fbases)*pb.matrix(Kernel_convolution_at_s).T
			

			sum_s *= (self.spacestep**4)
			x=self.Ts*self.Psi_xinv*sum_s+(1-self.Ts*self.alpha)*x+self.Ts*w
			X.append(x)
			field_update=np.array([self.act_fun(fr.T*x) for fr in self.field.fbases])
			Y.append((self.spacestep**2)*(self.EEG.ob_kernels_values*pb.matrix(field_update.reshape(len(field_update),1)))+v)

							
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


	

	def plot(self,centers):
		radius=2*pb.sqrt(pb.log(2))*(pb.sqrt(self.widths[0][0,0])) # It is calculated based on Full width at half maximum
		for center in centers:
			circle(center,radius)
		pb.title('field decomposition')	
		pb.show()


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
		ax.plot_wireframe(s1,s2,y,color='k')
		pb.title('kernel')
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


def f1(X,K,fbases,fspace,kspace,spacestep,Ts,Psi_xinv,alpha,act_fun):
	#'''state sequence'''


	Output=[]
	for i in range(X.shape[1]):
		x=X[:,i]
		K_center=(pylab.floor(K.shape[0]/2.),pylab.floor(K.shape[1]/2.))
		field_update=numpy.array([act_fun(fr.T*x) for fr in fbases])

		Kernel_convolution_at_s=[]
		for i in range(len(fspace)):
			K_shift_x=pylab.roll(K,i*len(kspace)) #shift kernel along x
			for j in range(len(fspace)):
				K_shift_y=pylab.roll(K_shift_x,j) #shift kernel along y
				K_truncate=K_shift_y[K_center[0]:,K_center[1]:]
				Kernel_convolution_at_s.append(pylab.sum(field_update*pylab.ravel(K_truncate)))	
		sum_s=pylab.hstack(fbases)*pylab.matrix(Kernel_convolution_at_s).T
		sum_s *= (spacestep**4)
		x=Ts*Psi_xinv*sum_s+(1-Ts*alpha)*x
		Output.append(x)
	return pylab.hstack(Output)

def h(X,fbases,EEG_ob_kernels_values,spacestep,act_fun):

	Y=[]

	for i in range(X.shape[1]):
		x=X[:,i]
		'''measurement equation'''
		field_update=numpy.array([act_fun(fr.T*x) for fr in fbases])
		y=(spacestep**2)*(EEG_ob_kernels_values*pylab.matrix(field_update.reshape(len(field_update),1)))
		Y.append(y)
	return  pylab.hstack(Y)



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

		#Parallel nodes
		ppservers = ()
		job_server = pp.Server(ppservers=ppservers)
		ncpus=job_server.get_ncpus()
		nSigmaVectors=2*self.model.field.nx+1
		parts=ncpus
		SigmaVectorInPart=nSigmaVectors/ncpus


		for y in Y:

			jobsy=[]
			jobsx=[]
			#calculate the sigma points matrix, each column is a sigma vector
			Chi=self.sigma_vectors(xhat,P)
			Chi_parts=pb.array_split(Chi,ncpus,axis=1)
			# update sima points



			for index in xrange(parts):
				jobsx.append(job_server.submit(f1,(Chi_parts[index],self.model.K,self.model.field.fbases,self.model.field.space,self.model.kernel.space,self.model.spacestep,self.model.Ts,self.model.Psi_xinv,self.model.alpha,self.model.act_fun),(),modules=('pylab','numpy')))
			result= [job() for job in jobsx]
			Chi_update=pb.hstack(result)


			##pointwise multiply by weights and sum along y-axis
			xhat_=pb.sum(pb.multiply(Wm_i,Chi_update),1)
			Chi_purturbation=Chi_update-xhat_

			weighted_Chi_purturbation=pb.multiply(Wc_i,Chi_purturbation)
			P_=Chi_purturbation*weighted_Chi_purturbation.T+(self.model.Ts**2)*self.model.Sw
			#redraw a new set of sigma points
			Chi_redraw=self.sigma_vectors(xhat_,P_)
			Chi_redraw_parts=pb.array_split(Chi_redraw,ncpus,axis=1)
			#calculate observations
	
			for index in xrange(parts):
				jobsy.append(job_server.submit(h,(Chi_redraw_parts[index],self.model.field.fbases,self.model.EEG.ob_kernels_values,self.model.spacestep,self.model.act_fun),(),modules=('pylab','numpy')))
			resulty= [job() for job in jobsy]
			Chi_redraw_observation_equation_update=pb.hstack(resulty)
			yhat_=pb.sum(pb.multiply(Wm_i,Chi_redraw_observation_equation_update),1)

			#measurement update equation
			measurement_purturbation=Chi_redraw_observation_equation_update-yhat_
			weighted_measurement_purturbation=pb.multiply(Wc_i,measurement_purturbation)
			Pyy=measurement_purturbation*weighted_measurement_purturbation.T+self.model.obs_noise_covariance
			Chi_redraw_purturbation=Chi_redraw-xhat_
			Pxy=Chi_redraw_purturbation*weighted_measurement_purturbation.T
			xhat=xhat_+(Pxy*Pyy.I)*(y-yhat_)
			P=P_-Pxy*(Pyy.I).T*Pxy.T

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
		xhatPredStore=[]
		PhatPredStore=[]
		PxyStore=[]
		# initialise the filter

		#calculate the weights
		Wm_i,Wc_i=self.sigma_vectors_weights()

		#Parallel nodes
		ppservers = ()
		job_server = pp.Server(ppservers=ppservers)
		ncpus=job_server.get_ncpus()
		nSigmaVectors=2*self.model.field.nx+1
		parts=ncpus
		SigmaVectorInPart=nSigmaVectors/ncpus




		for y in Y:

			jobsy=[]
			jobsx=[]
			#calculate the sigma points matrix, each column is a sigma vector
			Chi=self.sigma_vectors(xhat,P)
			Chi_parts=pb.array_split(Chi,ncpus,axis=1)
			# update sima points



			for index in xrange(parts):
				jobsx.append(job_server.submit(f1,(Chi_parts[index],self.model.K,self.model.field.fbases,self.model.field.space,self.model.kernel.space,self.model.spacestep,self.model.Ts,self.model.Psi_xinv,self.model.alpha,self.model.act_fun),(),modules=('pylab','numpy')))
			result= [job() for job in jobsx]
			Chi_update=pb.hstack(result)


			##pointwise multiply by weights and sum along y-axis
			xhat_=pb.sum(pb.multiply(Wm_i,Chi_update),1)
			Chi_purturbation=Chi_update-xhat_

			weighted_Chi_purturbation=pb.multiply(Wc_i,Chi_purturbation)
			P_=Chi_purturbation*weighted_Chi_purturbation.T+(self.model.Ts**2)*self.model.Sw
			#redraw a new set of sigma points
			Chi_redraw=self.sigma_vectors(xhat_,P_)
			Chi_redraw_parts=pb.array_split(Chi_redraw,ncpus,axis=1)
			#calculate observations
	
			for index in xrange(parts):
				jobsy.append(job_server.submit(h,(Chi_redraw_parts[index],self.model.field.fbases,self.model.EEG.ob_kernels_values,self.model.spacestep,self.model.act_fun),(),modules=('pylab','numpy')))
			resulty= [job() for job in jobsy]
			Chi_redraw_observation_equation_update=pb.hstack(resulty)
			yhat_=pb.sum(pb.multiply(Wm_i,Chi_redraw_observation_equation_update),1)

			#measurement update equation
			measurement_purturbation=Chi_redraw_observation_equation_update-yhat_
			weighted_measurement_purturbation=pb.multiply(Wc_i,measurement_purturbation)
			Pyy=measurement_purturbation*weighted_measurement_purturbation.T+self.model.obs_noise_covariance
			Chi_redraw_purturbation=Chi_redraw-xhat_
			Pxy=Chi_redraw_purturbation*weighted_measurement_purturbation.T
			xhat=xhat_+(Pxy*Pyy.I)*(y-yhat_)
			P=P_-Pxy*(Pyy.I).T*Pxy.T

			xhatStore.append(xhat)
			PhatStore.append(P)




		# initialise the smoother
		T=len(Y)
		xb = [None]*T
		Pb = [None]*T

		xb[-1], Pb[-1] = xhatStore[-1], PhatStore[-1]


		## smooth
		for t in range(T-2,-1,-1):


			jobs_smooth=[]





			#calculate the sigma points matrix from filterd states, each column is a sigma vector
			Chi_smooth=self.sigma_vectors(xhatStore[t],PhatStore[t]) #X_k
			Chi_smooth_parts=pb.array_split(Chi_smooth,ncpus,axis=1)


			for index in xrange(parts):
				jobs_smooth.append(job_server.submit(f1,(Chi_smooth_parts[index],self.model.K,self.model.field.fbases,self.model.field.space,self.model.kernel.space,self.model.spacestep,self.model.Ts,self.model.Psi_xinv,self.model.alpha,self.model.act_fun),(),modules=('pylab','numpy')))
			result= [job() for job in jobs_smooth]
			Chi_smooth_update=pb.hstack(result) #X_k+1	


			#pointwise multiply by weights and sum along y-axis
			xhat_smooth_=pb.sum(pb.multiply(Wm_i,Chi_smooth_update),1) #m_k+1_
			#purturbation
			Chi_smooth_purturbation=Chi_smooth-xhatStore[t] #X_k-m_k
			Chi_smooth_update_purturbation=Chi_smooth_update-xhat_smooth_ #X_k+1-m_k+1_
			#weighting
			weighted_Chi_smooth_purturbation=pb.multiply(Wc_i,Chi_smooth_purturbation) #W_ci*(X_k-m_k)
			weighted_Chi_smooth_update_purturbation=pb.multiply(Wc_i,Chi_smooth_update_purturbation)#W_ci*(X_k+1-m_k+1_)

			P_smooth_=Chi_smooth_update_purturbation*weighted_Chi_smooth_update_purturbation.T+(self.model.Ts**2)*self.model.Sw
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

			field_update=np.array([self.model.act_fun(fr.T*X[t]) for fr in self.model.field.fbases])
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

					K1_convolution_at_s.append(pb.sum(field_update*pb.ravel(K1_truncate)))
					K2_convolution_at_s.append(pb.sum(field_update*pb.ravel(K2_truncate)))
					K3_convolution_at_s.append(pb.sum(field_update*pb.ravel(K3_truncate)))



			K_convolution_at_s=pb.hstack((pb.matrix(K1_convolution_at_s).T,pb.matrix(K2_convolution_at_s).T,pb.matrix(K3_convolution_at_s).T))
			sum_s=pb.hstack(self.model.field.fbases)*K_convolution_at_s
			sum_s *= (self.model.spacestep**4)
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
		Xb= [pb.matrix(np.random.rand(self.model.field.nx,1)) for t in Y]

		# iterate
		keep_going = 1
		it_count = 0
		print " Estimatiing IDE's kernel and field weights"
		t0=time.time()
		while keep_going:

			self.model.kernel.weights=self.estimate_kernel(Xb)
			#_filter=getattr(ukf(self.model),'_filter')
			_filter=getattr(ukf(self.model),'rtssmooth')
			Xb,Pb,Xhat,Phat=_filter(Y)
			self.Xb=Xb
			self.Pb=Pb
			self.Xhat=Xhat
			self.Phat=Phat
			print it_count, " Kernel current estimate: ", self.model.kernel.weights.T
			#print it_count,"current estimate of Frobenius Norm: ", self.FroNorm()
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

def circle(center,radius):
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
	pb.plot(x0,y0)
	


def gen_obs_lattice(observation_centers_along_xy):
	"""
		generates spatial lattice

		Arguments
		----------
		observation_centers: array
			x,y of the centers of the Sensors
		Returns
		---------
		centers of the sensors: list of matrix
		the first element of the list is the bottom left corner of the spatial lattice
		then it goes to the top left corner (moving along the y-axis), then increment x and
		again goes along the y-axix till it reaches the top right corner of the spatial lattice

	"""

	return [np.matrix([[i,j]]).T for i in observation_centers_along_xy for j in observation_centers_along_xy]


	
def gen_obs_locations(FieldWidth,SensorWidth,SensorSpacing,BoundryEffectWidth):
	'''
	SensorWidth:  diameter in mm	
	SensorSpacing  in mm
	BoundryEffectWidth 
	'''
	ObservationCentre = pb.arange(-FieldWidth/2. + BoundryEffectWidth,FieldWidth/2. - BoundryEffectWidth+1,SensorSpacing)
	NSensors = len(ObservationCentre**2)
	RightDistance = np.abs((FieldWidth/2. - BoundryEffectWidth) - ObservationCentre[-1]);              
	ObservationCentre = ObservationCentre + RightDistance/2.;   
	return ObservationCentre



def field_centers(FieldWidth,Massdensity,field_basis_separation,field_basis_width):

	"""
		initialise space properties: finding the centers of the basis functions

		Arguments
		----------
		FieldWidth : int
			mm in each axis, should be even
		Massdensity:  int
			masses per mm

		field_basis_separation: int
			distance between each basis finction in mm

		field_basis_width:
			the width of the basis functions in mm

		Returns
		----------
		list of matrices
			centers of basis functions
	"""

	N_masses_in_width=Massdensity*FieldWidth+1
	
	x_center_outer =pb.linspace(-FieldWidth/2.,FieldWidth/2.,N_masses_in_width/(2.*field_basis_separation/pb.sqrt(2)));
	y_center_outer = x_center_outer;

	distance_between_centers = abs(x_center_outer[1]-x_center_outer[0]);
	x_center_inner = pb.linspace(-FieldWidth/2.+distance_between_centers/2.,FieldWidth/2.-distance_between_centers/2.,N_masses_in_width/		(2.*field_basis_separation/pb.sqrt(2))-1)
	y_center_inner = x_center_inner

	f_centers_inner=[np.array([[i,j]]).T for i in x_center_inner for j in y_center_inner]
	f_centers_outer=[np.array([[i,j]]).T for i in x_center_outer for j in y_center_outer]
	f_centers=np.concatenate([f_centers_outer,f_centers_inner])
	return [pb.matrix(x) for x in f_centers]


def field_cent(FieldWidth,sep):
	x_center_outer =pb.arange(-FieldWidth/2.+1,FieldWidth/2.,sep);
	y_center_outer = x_center_outer;
	f_centers_outer=[np.array([[i,j]]).T for i in x_center_outer for j in y_center_outer]
	x_center_inner =pb.arange(-FieldWidth/2.+sep/2.+1,FieldWidth/2.-sep/2.,sep);
	y_center_inner = x_center_inner
	f_centers_inner=[np.array([[i,j]]).T for i in x_center_inner for j in y_center_inner]
	f_centers=np.concatenate([f_centers_outer,f_centers_inner])
	
	return [pb.matrix(x) for x in f_centers]

def plot_field(X,fbases,f_space):
	"""
		plots spatial field

		Arguments
		----------
		X: matrix
			State vector at each time instant
		fbases: list of matrix
		each entery is the vector of basis functions: matrix
			matrix of nx x 1 dimension [phi_1(s) phi_2(s) ... phi_nx(s)].T

		f_space: array
			x,y of the spatial locations	
		Returns
		---------
		plot the spatial field at a given time instant

	"""
	z=pb.zeros((len(f_space),len(f_space)))
	m=0
	for i in range(len(f_space)):
		for j in range(len(f_space)):
			z[i,j]=float(X.T*fbases[m]) 
			m+=1
	pb.imshow(z,origin='lower',aspect=1, alpha=1,extent=[min(f_space),max(f_space),min(f_space),max(f_space)])	
	pb.colorbar(shrink=.55) 
	pb.show()

def anim_field(X,fbases,f_space):
	"""
		animates neural field

		Arguments
		----------
		X: list of matrix
			State vectors
		fbases: list of matrix
		each entery is the vector of basis functions: matrix
			matrix of nx x 1 dimension [phi_1(s) phi_2(s) ... phi_nx(s)].T

		f_space: array
			x,y of the spatial locations	
		Returns
		---------
		animates neural field

	"""
	pb.ion()
	fig_hndl =pb.figure()
	files=[]
	for t in range(len(X)):
		z=pb.zeros((len(f_space),len(f_space)))
		m=0
		for i in range(len(f_space)):
			for j in range(len(f_space)):
				z[i,j]=float(X[t].T*fbases[m]) 
				m+=1
		image=pb.imshow(z,animated=True,origin='lower',aspect=1, alpha=1,extent=[min(f_space),max(f_space),min(f_space),max(f_space)])

	pb.ioff()	
		
def plot_states(Xreal,Xest):
	T=len(Xest)
	for i in range(len(Xest[0])):
		x1=pb.zeros(T)
		x2=pb.zeros(T)
		for tau in range(T):
			x1[tau]=(float(Xreal[tau][i]))
			x2[tau]=(float(Xest[tau][i]))
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


