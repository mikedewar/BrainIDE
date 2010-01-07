import numpy as np
import pylab as pb
from ideBase import ideBase
'''Each basis that is to be used in the integrodifference equation 
model should be defined here. 

Each basis needs to have __mul__ and __conv__ oeprators that define
inner products and convolutions, respectively.'''

class basis():
	"""class defining common aspects of every basis"""
	def isbasis(self):
		return True

class gaussianBasis(basis):
	"""class defining a Gaussian basis function
	Arguments
	---------
	dimension : int
		The dimension of the basis functions
	center : matrix
		The center of the basis function
	width: matrix
		The covariance of the basis function
	attributes
	----------
	dimension : int
		The dimension of the basis functions
	center : matrix
		The center of the basis function
	width: matrix
		The covariance of the basis function
	constant: float
		The convolution coefficient
	
	
	"""
	
	def __init__(self,dimension,centre,width,constant = 1.0):
		assert type(dimension) is int
		self.dimension = dimension
		self.centre = np.matrix(centre)
		self.centre.shape = (self.dimension,1)
		self.width = np.matrix(width)
		# constant - The convolution coefficient
		self.constant = float(constant)
		# evaluate
		#self.evaluate = lambda s: self.constant * np.exp(-(s-self.centre).T*self.width.I*(s-self.centre))
		self.evaluate = lambda s: self.constant * np.exp(-(s-self.centre).T*self.width.I*(s-self.centre))

	def __repr__(self):	
		return 'Gaussian basis: ' +'centre= '+str(self.centre) + ', width= ' + str(self.width)

	def __mul__(self,g):
		'''inner product of two gaussian basis functions'''
		assert g.__class__ is gaussianBasis, 'inner product error'
		cij=self.width.I + g.width.I
		uij=cij.I * (self.width.I*self.centre + g.width.I*g.centre)		
		#rij=(self.centre.T * self.width * self.centre) + (g.centre.T * g.width * g.centre) - (uij.T * cij * uij)
		rij=(self.centre-g.centre).T*(self.width.I+g.width.I).I*self.width.I*g.width.I*(self.centre-g.centre)
		return self.constant * g.constant * (np.pi)**(self.dimension * 0.5) * np.linalg.det( cij.I ) **(0.5) * np.exp(-rij)
		 
	def conv(self,g):
		'''convolution of two gaussian basis functions'''
		
		assert g.__class__ is gaussianBasis, 'convolution error'
		assert self.dimension == g.dimension, 'convolution error (incompatible dimension)'

		invC_mj = (self.width.I + g.width.I).I
		constant = np.pi**(0.5*self.dimension) * np.linalg.det(invC_mj)**0.5

		centre = self.centre + g.centre # the centre of the gaussian
		width = invC_mj * self.width.I * g.width.I # the covariance matrix

		h = gaussianBasis(self.dimension, centre, width.I, constant) # form a gaussian object

		return h

	def plot(self):
		if self.dimension==1:
			space=np.linspace(float((-5*np.sqrt(self.width))+self.centre),float((5*np.sqrt(self.width))+self.centre),100)		
			y = np.array([self.evaluate(s) for s in space])
			pb.plot(space,y[:,0])
			pb.show()
		if self.dimension==2:
			u1=np.linspace(float((-5*np.sqrt(self.width[0,0]))+self.centre[0]),float((5*np.sqrt(self.width[0,0]))+self.centre[0]),100)
			u2=np.linspace(float((-5*np.sqrt(self.width[1,1]))+self.centre[1]),float((5*np.sqrt(self.width[1,1]))+self.centre[1]),100)
			y=np.zeros((len(u1),len(u2)))
			for i in range(len(u1)):
				for j in range(len(u2)):
					y[i,j]=self.evaluate(np.matrix([[u1[i]],[u2[j]]]))
			self.y=y
			pb.imshow(y)
			pb.colorbar(shrink=.55)
			pb.show()

		
