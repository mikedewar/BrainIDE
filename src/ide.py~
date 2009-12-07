"""
this module provides the ide class, which describes an 
integrodifference equation model and methods to generate,
interrogate and estimate the model from data.
"""
# my modules
import ssmodel, ideBase, bases
# built-ins
import pylab as pb
import numpy as np
import  warnings

class ide:
	"""class defining an integrodifference equation (ide)
	
	properties:
	(name :type - description)
	
	dimension :scalar - spatial dimension of the ide
	kernel :ideBase.kernel - a spatial mixing kernel object
	field :ideBase.field - the spatial field object
	space :list - simulated space
	time :list - time indices
	ssmodel :ssmodel - state space representation of the ide
	obs :list - observations
	obs_locns :list - observation locations
	obs_noise_covariance :numpy.matrix - covariance of the observed field
	field_noise_variance :numpy.matrix - covariance of the spatial field
	
	notes:
	For dimension > 1, the space and obs_locns properties need to be 
	lists of vectors, using numpy matrices.
	
	Check out the gen_example_x() functions (in this module) for usage examples.
	"""
	
	def __init__(self,kernel,field,space,obs_locns,obs_noise_covariance,field_noise_variance,T,report=0):
		'''notes:
		If report is set to 1, a little report about the model is generated.'''
				
		assert kernel.__class__ is ideBase.kernel, 'specified kernel is not a kernel object' 
		self.kernel = kernel
		
		assert field.__class__ is ideBase.field, 'specified field is not a field object' 
		self.field = field
		
		# dimension consistency check
		assert kernel.dimension == field.dimension, 'inconsistent dimensions'
		self.dimension = field.dimension
		
		self.space = space
		self.obs_locns = obs_locns
		
		# positive definite covariance matrix check
		#-------------------numpy.linalg.cholesky(obs_noise_covariance)---------------------
		# observation location consistency check
		
		self.obs_noise_covariance = pb.matrix(obs_noise_covariance)
		self.field_noise_variance = field_noise_variance
		assert len(obs_locns) == obs_noise_covariance.shape[0], 'inconsistent observation location size'
		
		
		# generate empty ssmodel
		nx = len(field.bases)
		ny = len(obs_locns)
		ntheta = len(kernel.bases)
		A = pb.zeros([nx,nx])
		C = pb.zeros([ny,nx])
		Sw = pb.zeros([nx,nx])
		Sv = pb.zeros([ny,ny])
		x0=pb.zeros([nx,1])
		self.T=T
		self.ssmodel = ssmodel.ssmodel(A,C,Sw,Sv,x0)
		
		# print a report
		if report:
			print 'IDE model:'
			print 'kernel with',ntheta,'basis functions'
			print 'field with',nx,'basis functions'
			print 'with',ny,'observation locations'
		
	def gen_ssmodel(self):
		"""generate a state space model using the ide object's properties"""		
		# unpack
		phi, psi = self.field.bases, self.kernel.bases		
		nx, ntheta, ny = len(phi), len(psi), len(self.obs_locns)	
		# theta is the vector containing the weights. 
		theta = pb.matrix(self.kernel.weights)
		theta.shape = ntheta, 1
		
		s = self.obs_locns
		# form gamma (this is the hard bit)
		if hasattr(self,'gamma'):
			gamma = self.gamma
		else:
			# initialise gamma as a nx x nx nested list
			gamma = [[0 for x in range(nx)] for y in range(nx)]
			for i in range(nx):
				for j in range(nx):
					g = pb.matrix(pb.zeros([ntheta,1]))
					for m in range(ntheta):
						# form the convolution of phi[j] and psi[m]
						Phi = phi[j].conv(psi[m])
						# find the inner product of the resulting basis with phi[i]
						g[m] = Phi * phi[i]
					# store each g vector in the nested list
					gamma[i][j] = g
			# store the gamma so we don't have to re-generate it each time we do this		
			self.gamma = gamma
		# form Psi_theta
		Psi_theta = pb.matrix(pb.zeros([nx,nx]))
		for i in range(nx):
			for j in range(nx):
				Psi_theta[i,j] = theta.T * gamma[i][j]
		self.Psi_theta=	Psi_theta	
		# form invPsi_x
		if hasattr(self,'invPsi_x'):
			invPsi_x = self.invPsi_x
		else:		
			Psi_x = pb.matrix(pb.zeros([nx,nx]))
			for i in range(nx):
				for j in range(nx):
					Psi_x[i,j] = phi[i] * phi[j]
			invPsi_x = Psi_x.I
			self.invPsi_x = invPsi_x
		# form the state matrix A
		self.ssmodel.A = invPsi_x * Psi_theta
		# check for stability
		for u in self.eig():
			if abs(u) > 1:
				warnings.warn('unstable system')
		# form the observation matrix
		for i in range(ny):
			for j in range(nx):
				self.ssmodel.C[i,j] = self.field.bases[j].evaluate(s[i])
		# form Sw
		self.ssmodel.Sw = self.field_noise_variance * invPsi_x;
		self.Sw = self.ssmodel.Sw
		# populate the ssmodel
		self.ssmodel.Sv = self.obs_noise_covariance
		self.ssmodel.x0 = pb.matrix(self.field.weights).T
		
	def simulate(self):
		"""simulates the model across time and space"""
		X,Y=self.ssmodel.simulate(self.T)
		return X,Y
		

	
	def estimate_kernel(self,X,P,M):
		"""estimate the ide model's kernel weights from data stored in the ide object"""
		# unpack
		nx, na = self.ssmodel.nx, len(self.kernel.weights)
		# form Xi variables
		Xi_0 = pb.matrix(pb.zeros([nx,nx]))
		Xi_1 = pb.matrix(pb.zeros([nx,nx]))
		for t in range(1,self.T):
			Xi_0 += X[t-1] * X[t].T + M[t]
			Xi_1 += X[t-1] * X[t-1].T + P[t-1]
		# form Upsilon
		Upsilon = pb.matrix(pb.zeros([na,na]))
		temp = self.invPsi_x * self.Sw.I * self.invPsi_x
		for i in range(nx):
			for j in range(nx):
				for k in range(nx):
					for m in range(nx):
						Upsilon += Xi_1[i,j] * self.gamma[k][j] * temp[k,m] * self.gamma[m][i].T
		Upsilon.I
		# form upsilon
		upsilon = pb.matrix(pb.zeros([na,1]))
		temp = Xi_0 * self.Sw.I * self.invPsi_x
		for i in range(nx):
			for j in range(nx):
				upsilon += temp[i,j] * self.gamma[j][i]
		# update the weights
		weights =  Upsilon.I * upsilon
		self.kernel.weights = [float(x) for x in weights[:,0]]
		
		
	def estimate_field(self,Y):
		'''estimates the field of the ide across time'''	
		xb, Pb, KStore, M=self.ssmodel.rtssmooth(Y)	
		return xb, Pb, KStore, M


	def em(self,Y):
		"""estimate the ide's kernel and field weights using an EM algorithm"""
		# form state soace model
		self.gen_ssmodel()
		# generate a random state sequence
		X= [pb.matrix(np.random.rand(self.ssmodel.nx,1)) for t in range(self.T)]
		P=[1000*pb.matrix(pb.ones((self.ssmodel.nx,self.ssmodel.nx)))]*self.T
		M=[1000*pb.matrix(pb.ones((self.ssmodel.nx,self.ssmodel.nx)))]*self.T
		#P=[10000*pb.matrix(pb.eye((self.ssmodel.nx)))]*self.T
		#M=[10000*pb.matrix(pb.eye((self.ssmodel.nx)))]*self.T

		# introduce the observations
		# iterate
		keep_going = 1
		it_count = 0
		max_it = 20
		while keep_going:
			self.estimate_kernel(X,P,M)
			self.gen_ssmodel()
			X, P, KStore,M=self.estimate_field(Y)
			print it_count, " current estimate: ", self.kernel.weights
			print it_count,"current estimate of Frobenius Norm: ", self.FroNorm()
			if it_count == max_it:
				keep_going = 0
			it_count += 1
	
	def eig(self):
		'''returns a list of the eigenvalues of the models A matrix'''
		u,v = pb.linalg.eig(self.ssmodel.A)
		return [i for i in u]
	
	def FroNorm(self):
		'''returns Frobenius Norm'''
		u= np.sqrt(np.trace(self.ssmodel.A.T*self.ssmodel.A))
		return u	

