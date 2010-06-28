#Author Parham Aram
#Date 28-06-2010
'''This module defines each Gaussian basis function to be used in the non-linear IDE model. '''

#Standard library imports

from __future__ import division
import pylab as pb
import numpy as np

class Basis():
	'''This is a class to define Gaussian basis functions'''

	def __init__(self,center,width,dimension,constant=1):
		'''
		Arguments
		---------
			center: dimension X 1 matrix 
				center of the gaussian basis function	
			width: covariance of the gaussian sigma^2
				float
			dimension: int
					
			constant: float
				must be one, this is used to calculate the convolution and Inner product of two gaussian basis functions'''

		self.center=pb.matrix(center)
		self.width=width
		self.dimension=dimension
		self.constant=constant
		assert type(self.center) is pb.matrix, 'center must be a matrix'
		assert self.center.shape==(self.dimension,1), 'center shape mismatch'
		assert type(width) is float,'width must be float type'

	def __call__(self,s):

		assert type(s) is pb.matrix, 'input must be a matrix'
		assert s.shape==(self.dimension,1), 'input shape mismatch'
		return float(self.constant*pb.exp(-1/(self.width)*(s-self.center).T*(s-self.center)))

	def __repr__(self):

		return 'Gaussian basis function;'+'center='+str(self.center) +',width='+str(self.width) 

	def conv(self,In):

		'''calculates the convolution of two gaussian basis functions
			Argument
			--------
			In: gaussian basis function

			Returns
			--------
			convolution of two gaussian basis functions'''

		assert In.__class__ is Basis, 'The input must be a Gaussian basis function'
		
		convolution_weight=((pb.pi*self.width*In.width)/(self.width+In.width))**(self.dimension*0.5)
		convolution_width=self.width+In.width
		convolution_center=self.center+In.center
		return Basis(convolution_center,convolution_width,self.dimension,constant=convolution_weight)



	def __mul__(self,In):

		'''calculates the Inner product of two gaussian basis functions

			Argument
			--------
			In: gaussian basis function

			Returns
			--------
			Inner product of two gaussian basis functions'''

		assert In.__class__ is Basis, 'The input must be a Gaussian basis function'
		cij=(self.width+In.width)/(self.width*In.width)
		rij=(self.center-In.center).T*(1/(self.width+In.width))*(self.center-In.center)
		return float(self.constant*In.constant*(pb.pi)**(self.dimension*0.5)*(cij**(-0.5*self.dimension))*pb.exp(-rij))



