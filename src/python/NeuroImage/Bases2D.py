#Author Parham Aram
#Date 21-11-2010
from __future__ import division
import pylab as pb
import numpy as np

class basis():

	'''This is a class to define Gaussian basis functions'''

	def __init__(self,mu,width,dimension=2,constant=1):

		'''
		Arguments
		---------
			mu: matrix dimension X 1
				center of the gaussian	
			width: covariance of the gaussian sigma^2
				float
			dimension: int
					
			constant: float
				must be one, this is just to calculate the covariance of two gaussian basis functions'''


		self.dimension=dimension
		if not self.dimension==2:raise ValueError('dimension must be 2') 	
		self.mu=pb.matrix(mu)
		self.mu.shape=(self.dimension,1)		
		self.width=width
		assert type(width) is float,'width must be float type'
		self.constant=constant		
			
		



	def __call__(self,X,Y):

		'''calculates the 2D Gaussian at a given spatial location:(x,y), to speed up X,Y can be the output
		of the pb.meshgrid

		Arguments:
		---------
			X: float or ndarry for speed
			Y: float or ndarray for speed
		Returns:
		--------
			Z: float or ndarray
				the value of the two dimensional Gaussian at a given location
			'''

		mu_x=self.mu[0,0]
		mu_y=self.mu[1,0]
		X_mu_x_pow2=(X-mu_x)**2
		Y_mu_y_pow2=(Y-mu_y)**2
		Z=self.constant*pb.exp(-1/(self.width)*(X_mu_x_pow2+Y_mu_y_pow2))
		return Z

	def plot(self,X,Y):

		'''plot 2D-Gaussian over the grid X,Y

		Arguments:
		---------
			X: ndarry, output of the pb.meshgrid
			Y: ndarry, output of the pb.meshgrid'''


		mu_x=self.mu[0,0]
		mu_y=self.mu[1,0]
		X_mu_x_pow2=(X-mu_x)**2
		Y_mu_y_pow2=(Y-mu_y)**2
		Z=self.constant*pb.exp(-1/(self.width)*(X_mu_x_pow2+Y_mu_y_pow2))
		pb.imshow(Z,extent=[X.min(),X.max(),Y.min(),Y.max()],origin='lower')
		pb.show()

	
	def __repr__(self):
		return 'Gaussian basis function;'+'mu='+str(self.mu) +',width='+str(self.width) 


	def conv(self,In):

		'''calculates the convolution of two gaussian basis functions
			Argument
			--------
			In: gaussian basis function
			Returns
			--------
			convolution of two gaussian basis functions'''


		if not isinstance(In,basis):raise ValueError("The input must be a Gaussian basis function")

		convolution_weight=((pb.pi*self.width*In.width)/(self.width+In.width))**(self.dimension*0.5)
		convolution_width=self.width+In.width
		convolution_mu=self.mu+In.mu
		return basis(convolution_mu,convolution_width,self.dimension,constant=convolution_weight*self.constant*In.constant)

	def __mul__(self,In):

		'''calculates the Inner product of two gaussian basis functions, multiply a number with gaussian basis functions

			Argument
			--------
				In: gaussian basis function or float

			Returns
			--------
				float or basis instance
				Inner product of two gaussian basis functions or
				weighted Gaussian 
				'''


		if In.__class__ is basis:
			cij=(self.width+In.width)/(self.width*In.width)
			rij=(self.mu-In.mu).T*(1/(self.width+In.width))*(self.mu-In.mu)
			return float(self.constant*In.constant*(pb.pi)**(self.dimension*0.5)*(cij**(-0.5*self.dimension))*pb.exp(-rij))

		else:
			constant=float(self.constant*In)
			return basis(self.mu,self.width,self.dimension,constant)

	def __rmul__(self,In):

		'''calculates the Inner product of two gaussian basis functions, multiply a number with gaussian basis functions

			Argument
			--------
				In: gaussian basis function or float

			Returns
			--------
				float or basis instance
				Inner product of two gaussian basis functions or
				weighted Gaussian 
				'''

		if In.__class__ is basis:
			cij=(self.width+In.width)/(self.width*In.width)
			rij=(self.mu-In.mu).T*(1/(self.width+In.width))*(self.mu-In.mu)
			return float(self.constant*In.constant*(pb.pi)**(self.dimension*0.5)*(cij**(-0.5*self.dimension))*pb.exp(-rij))

		else:
			constant=float(self.constant*In)
			return basis(self.mu,self.width,self.dimension,constant)



